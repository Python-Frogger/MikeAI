# =============================================================================
# ppo_agent_v9.py  —  Actor-Critic network + PPO algorithm + rollout buffer
# =============================================================================
#
# Architecture: dual-path backbone → LSTM → Actor/Critic
#   CNN path   (70 → 128):  Conv2d(1→16) → Conv2d(16→32) → Linear(2240→128)
#   Dense path (24 → 64):   Linear(24→64)
#   Shared trunk (192→128): Linear → ReLU
#   LSTM       (128→128):   1 layer, hidden=128  ← NEW in v9
#   Actor head (128→4):     logits → Categorical distribution
#   Critic head (128→1):    state value V(s)
#
# Why LSTM?
#   v8 (and DQN v7) processed each frame independently — no memory of what
#   just happened. The aliens oscillate left↔right in a predictable pattern
#   (~320 steps per full cycle), but without memory the agent can only react
#   to where they ARE, not predict where they WILL BE.
#
#   LSTM carries a hidden state (h, c) forward through the entire episode.
#   After a few training cycles it will learn:
#     - Which direction aliens are moving (velocity from position sequence)
#     - How many steps until next bounce (distance from wall + remembered velocity)
#     - Bullet travel time (shoot now → hit in ~10 steps → time shots forward)
#
# BPTT (Backpropagation Through Time):
#   SEQ_LEN = 2560 — gradients flow back 2560 steps during training.
#   This covers eight full alien left↔right oscillation cycles (~full winning game).
#   The hidden state (h, c) carries forward for the WHOLE episode during play.
#   SEQ_LEN only controls how far gradients flow — not how far the agent remembers.
#
# Warm-start from v8:
#   CNN, trunk, actor head, critic head weights transfer from best_model_v8.pth.
#   LSTM initialises fresh. The spatial knowledge (what aliens look like, where
#   the player is, kill rewards) carries over. Only temporal reasoning is new.
#
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# =============================================================================
# MANUAL LSTM — ROCm/RX 9070 (gfx1201) compatibility
# =============================================================================
# nn.LSTM backward pass triggers a MIOpen reduction kernel that fails to
# compile on gfx1201 (RX 9070) due to a missing 'type_traits' header in the
# installed ROCm version. Fix: implement LSTM using basic ops only.
# Mathematically identical to nn.LSTM(input_size, hidden_size, batch_first=True).
# Uses only Linear + sigmoid + tanh — all have solid ROCm support.
# =============================================================================

class ManualLSTM(nn.Module):
    """
    Drop-in replacement for nn.LSTM(input_size, hidden_size, batch_first=True).

    Interface matches nn.LSTM exactly:
      forward(x, hidden) → (output, (h_n, c_n))
      x:      (batch, seq_len, input_size)
      hidden: (h, c) each (1, batch, hidden_size), or None
      output: (batch, seq_len, hidden_size)
      h_n, c_n: (1, batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Fused weights for all 4 gates: input (i), forget (f), cell (g), output (o)
        self.W_ih = nn.Linear(input_size,   4 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size,  4 * hidden_size, bias=False)
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'bias' in name:
                nn.init.zeros_(p.data)

    def forward(self, x, hidden=None):
        batch, seq_len, _ = x.shape
        if hidden is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h = hidden[0].squeeze(0)   # (1, batch, H) → (batch, H)
            c = hidden[1].squeeze(0)

        outputs = []
        for t in range(seq_len):
            gates = self.W_ih(x[:, t]) + self.W_hh(h)   # (batch, 4H)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)   # (batch, seq_len, H)
        return output, (h.unsqueeze(0), c.unsqueeze(0))

GRID_ROWS    = 7
GRID_COLS    = 10
GRID_SIZE    = GRID_ROWS * GRID_COLS   # 70
CONTEXT_SIZE = 24
STATE_SIZE   = GRID_SIZE + CONTEXT_SIZE  # 94

ACTION_NAMES  = ['Left', 'Right', 'Shoot', 'Nothing']

N_QUINTILES = 5   # percentile-based stratification — always 5 equal groups

# LSTM sequence length = eight full alien oscillation cycles (~full winning game).
# Maths from game_env_v7.py:
#   Rightmost alien starts at x=570, bounces at x=760 → 152 steps.
#   Formation then moves left 200px → 160 steps. Then right 200px → 160 steps.
#   Full cycle: 152 + 160 = 312 (first) then 320 per subsequent cycle.
#   Using 2560 (8 × 320) so gradients span a full winning game — agent can learn
#   whole-game strategies (early clear patterns → endgame, wave timing end-to-end).
SEQ_LEN     = 2560
LSTM_HIDDEN = 128


# =============================================================================
# NETWORK
# =============================================================================

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic with LSTM temporal memory.

    CNN + dense backbone identical to v8/v7 — spatial reasoning preserved.
    LSTM sits between trunk and heads — adds temporal reasoning on top.

    During rollout:   get_action() is called step-by-step, (h,c) passes forward.
    During training:  evaluate() receives (batch, seq_len, state) tensors.
    """

    def __init__(self):
        super().__init__()

        # ── CNN path (same as v8) ───────────────────────────────────────────
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.grid_fc = nn.Linear(32 * GRID_ROWS * GRID_COLS, 128)

        # ── Dense path (same as v8) ─────────────────────────────────────────
        self.ctx_fc = nn.Linear(CONTEXT_SIZE, 64)

        # ── Shared trunk (same as v8) ───────────────────────────────────────
        self.trunk = nn.Linear(128 + 64, 128)

        # ── LSTM: temporal memory ← NEW ────────────────────────────────────
        # ManualLSTM instead of nn.LSTM — identical maths, avoids the MIOpen
        # reduction kernel that crashes on RX 9070 (gfx1201) during backward.
        self.lstm = ManualLSTM(128, LSTM_HIDDEN)

        # ── Actor: 4 action logits (reads from LSTM output) ────────────────
        self.actor_head = nn.Linear(LSTM_HIDDEN, 4)

        # ── Critic: scalar state value (reads from LSTM output) ────────────
        self.critic_head = nn.Linear(LSTM_HIDDEN, 1)

        self.relu = nn.ReLU()

        # Small initial weights → near-uniform early policy + near-zero values
        nn.init.uniform_(self.actor_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.uniform_(self.critic_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.critic_head.bias)

    # ── Backbone: CNN + dense → trunk ──────────────────────────────────────

    def _backbone(self, x):
        """
        Process state(s) through CNN + dense → trunk.
        x can be (batch, STATE_SIZE) or (batch, seq_len, STATE_SIZE).
        Returns same leading shape with last dim = 128.
        """
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, STATE_SIZE)

        grid_flat = x_flat[:, :GRID_SIZE]
        context   = x_flat[:, GRID_SIZE:]

        g = grid_flat.view(-1, 1, GRID_ROWS, GRID_COLS)
        g = self.relu(self.conv1(g))
        g = self.relu(self.conv2(g))
        g = g.view(g.size(0), -1)
        g = self.relu(self.grid_fc(g))   # (N, 128)

        c = self.relu(self.ctx_fc(context))   # (N, 64)

        merged = torch.cat([g, c], dim=1)         # (N, 192)
        trunk  = self.relu(self.trunk(merged))    # (N, 128)

        return trunk.view(*leading, 128)

    # ── Utility: zero hidden state ──────────────────────────────────────────

    def init_hidden(self, batch_size=1, device=None):
        """Return (h, c) zeroed — call at episode start and rollout start."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, LSTM_HIDDEN, device=device)
        c = torch.zeros(1, batch_size, LSTM_HIDDEN, device=device)
        return h, c

    # ── Rollout collection: single step ────────────────────────────────────

    @torch.no_grad()
    def get_action(self, state_tensor, hidden):
        """
        Sample action for one step during rollout.
        state_tensor: (1, STATE_SIZE)
        hidden: (h, c) from previous step (or init_hidden at episode start)
        Returns: (action int, log_prob float, value float, new_hidden)
        """
        # Add seq_len=1 dimension for LSTM
        x = state_tensor.unsqueeze(1)       # (1, 1, 94)
        trunk = self._backbone(x)            # (1, 1, 128)
        lstm_out, new_hidden = self.lstm(trunk, hidden)   # (1, 1, 128)
        h_out = lstm_out.squeeze(1)          # (1, 128)

        logits = self.actor_head(h_out)      # (1, 4)
        value  = self.critic_head(h_out)     # (1, 1)

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), new_hidden

    # ── PPO update: sequence batch ──────────────────────────────────────────

    def evaluate(self, states_seq, actions_seq, hidden):
        """
        Re-evaluate stored (state, action) sequence pairs during PPO update.
        states_seq:  (batch, seq_len, STATE_SIZE)
        actions_seq: (batch, seq_len)
        hidden:      (h, c) zero-initialised at sequence start
        Returns: (log_probs, values, entropy) — all (batch * seq_len,) with gradients.
        """
        trunk = self._backbone(states_seq)              # (batch, seq_len, 128)
        lstm_out, _ = self.lstm(trunk, hidden)           # (batch, seq_len, 128)

        batch, seq_len = states_seq.shape[:2]
        h_flat = lstm_out.reshape(batch * seq_len, LSTM_HIDDEN)
        a_flat = actions_seq.reshape(batch * seq_len)

        logits = self.actor_head(h_flat)                # (batch*seq, 4)
        values = self.critic_head(h_flat).squeeze(-1)   # (batch*seq,)

        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(a_flat)
        entropy   = dist.entropy()

        return log_probs, values, entropy

    # ── Probe helpers (for p_probe_v9.py) ──────────────────────────────────

    @torch.no_grad()
    def action_probs(self, state_tensor, hidden=None):
        """Returns P(Left/Right/Shoot/Nothing) as numpy. For probing."""
        if hidden is None:
            hidden = self.init_hidden(1, state_tensor.device)
        x = state_tensor.unsqueeze(1)
        trunk = self._backbone(x)
        lstm_out, _ = self.lstm(trunk, hidden)
        logits = self.actor_head(lstm_out.squeeze(1))
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def state_value(self, state_tensor, hidden=None):
        """Returns V(s) scalar. For probing."""
        if hidden is None:
            hidden = self.init_hidden(1, state_tensor.device)
        x = state_tensor.unsqueeze(1)
        trunk = self._backbone(x)
        lstm_out, _ = self.lstm(trunk, hidden)
        return self.critic_head(lstm_out.squeeze(1)).item()


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """
    Stores ALL steps from a fixed-length rollout window (on-policy).

    v9 change vs v8:
      get_batches() → get_sequences()
      Instead of random frame sampling, yields contiguous sequences of SEQ_LEN steps.
      The LSTM needs to see steps in time order to build useful hidden states.
      Mini-batch = SEQS_PER_BATCH sequences × SEQ_LEN steps ≈ same step count as v8.

    Quintile sampling is preserved but operates at sequence level:
      Each episode's reward weight is spread uniformly to all its sequences.
      Top-quintile episodes still get 20% of gradient attention.

    Memory footprint (524,288 steps, SEQ_LEN=1280):
      States:              524,288 × 94 × 4 bytes  = ~197 MB
      Actions/rewards/etc: 524,288 × 6 × 4 bytes   = ~12.6 MB
      Total:                                        ~210 MB  (trivial on 17 GB GPU)
    """

    def __init__(self, n_steps, seq_len=SEQ_LEN):
        self.n_steps = n_steps
        self.seq_len = seq_len
        self.states    = np.zeros((n_steps, STATE_SIZE), dtype=np.float32)
        self.actions   = np.zeros(n_steps, dtype=np.int64)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, state, action, reward, value, log_prob, done):
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr]     = float(done)
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Backwards pass to compute GAE advantages and returns.
        Identical to v8 — GAE is frame-level, unaffected by LSTM.
        """
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0
                next_val = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]

            delta    = (self.rewards[t]
                        + gamma * next_val * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def _compute_sequence_weights(self, ep_records):
        """
        Build per-STEP quintile weights, then aggregate to per-SEQUENCE weights.

        Episode-level reward bucketing (same logic as v8):
          Sort episodes by reward → split into N_QUINTILES groups.
          Top quintile gets 1/N_QUINTILES of total gradient weight.
          Efficient 20-kill game beats spammy 20-kill game.

        Each sequence inherits the average step-weight of its constituent steps.
        Sequences spanning episode boundaries get a blended weight.
        """
        n_seqs = self.ptr // self.seq_len
        if n_seqs == 0:
            return np.ones(1, dtype=np.float32)

        # Build step-level weights (same as v8's compute_weights)
        step_weights = np.zeros(self.ptr, dtype=np.float32)
        valid = [(s, min(e, self.ptr), score)
                 for s, e, score in ep_records if s < min(e, self.ptr)]

        if valid:
            valid_sorted = sorted(valid, key=lambda x: x[2])
            n = len(valid_sorted)
            w_per_q = 1.0 / N_QUINTILES
            for q in range(N_QUINTILES):
                lo_idx = (q * n) // N_QUINTILES
                hi_idx = ((q + 1) * n) // N_QUINTILES
                q_eps  = valid_sorted[lo_idx:hi_idx]
                if not q_eps:
                    continue
                total_steps = sum(e - s for s, e, _ in q_eps)
                if total_steps > 0:
                    for s, e, _ in q_eps:
                        step_weights[s:min(e, self.ptr)] = w_per_q / total_steps
        else:
            step_weights[:self.ptr] = 1.0 / max(self.ptr, 1)

        # Aggregate step weights to sequence weights
        seq_weights = np.zeros(n_seqs, dtype=np.float32)
        for i in range(n_seqs):
            s = i * self.seq_len
            e = s + self.seq_len
            seq_weights[i] = step_weights[s:e].mean()

        total = seq_weights.sum()
        if total > 0:
            seq_weights /= total
        else:
            seq_weights[:] = 1.0 / n_seqs
        return seq_weights

    def get_sequences(self, seqs_per_batch, device, ep_records=None):
        """
        Yield mini-batches of contiguous sequences for LSTM training.

        Each batch: (states, actions, log_probs, advantages, returns)
        Shapes: (seqs_per_batch, seq_len, ...) except scalars (seqs_per_batch * seq_len,)

        Hidden state is zeroed at each sequence start (truncated BPTT).
        The LSTM learns to reconstruct context from the first few frames.
        The full-episode hidden state is maintained during ROLLOUT (not training).
        """
        n_seqs = self.ptr // self.seq_len
        if n_seqs == 0:
            return

        if ep_records is not None:
            seq_weights = self._compute_sequence_weights(ep_records)
            seq_indices = np.random.choice(n_seqs, size=n_seqs, replace=True, p=seq_weights)
        else:
            seq_indices = np.random.permutation(n_seqs)

        for start in range(0, len(seq_indices), seqs_per_batch):
            batch_seq_ids = seq_indices[start:start + seqs_per_batch]
            if len(batch_seq_ids) < seqs_per_batch // 2:
                continue

            B = len(batch_seq_ids)
            states_b   = np.zeros((B, self.seq_len, STATE_SIZE), dtype=np.float32)
            actions_b  = np.zeros((B, self.seq_len), dtype=np.int64)
            lp_b       = np.zeros((B, self.seq_len), dtype=np.float32)
            adv_b      = np.zeros((B, self.seq_len), dtype=np.float32)
            ret_b      = np.zeros((B, self.seq_len), dtype=np.float32)

            for bi, seq_id in enumerate(batch_seq_ids):
                s = seq_id * self.seq_len
                e = s + self.seq_len
                states_b[bi]  = self.states[s:e]
                actions_b[bi] = self.actions[s:e]
                lp_b[bi]      = self.log_probs[s:e]
                adv_b[bi]     = self.advantages[s:e]
                ret_b[bi]     = self.returns[s:e]

            yield (
                torch.FloatTensor(states_b).to(device),    # (B, seq_len, 94)
                torch.LongTensor(actions_b).to(device),    # (B, seq_len)
                torch.FloatTensor(lp_b).to(device),        # (B, seq_len)
                torch.FloatTensor(adv_b).to(device),       # (B, seq_len)
                torch.FloatTensor(ret_b).to(device),       # (B, seq_len)
            )

    def reset(self):
        self.ptr = 0
        self.advantages[:] = 0
        self.returns[:]    = 0

    def episode_stats(self):
        """Scan buffer for episode boundaries, return list of episode rewards."""
        ep_rewards = []
        current = 0.0
        for i in range(self.ptr):
            current += self.rewards[i]
            if self.dones[i]:
                ep_rewards.append(current)
                current = 0.0
        return ep_rewards


# =============================================================================
# HALL OF FAME
# =============================================================================

class HallOfFame:
    """
    Cross-rollout memory — hybrid kill + reward ranking. Same concept as v8.

    v9 change: get_batches() runs each episode through the LSTM sequentially
    to get correct hidden states at each sequence chunk boundary. This means
    HoF advantages are computed with proper temporal context, not just single
    frames as in v8.

    The LSTM state is zeroed at the episode start, then carried forward through
    each SEQ_LEN chunk — the same approach as the rollout training.
    """

    def __init__(self, max_episodes=40):
        self.max_per_group = max_episodes // 2
        self.kill_hof   = []
        self.reward_hof = []

    def offer(self, states, actions, log_probs, rewards, dones, kills, score):
        """Offer a complete episode. Assessed against both groups."""
        ep = {
            'states':    np.array(states,    dtype=np.float32),
            'actions':   np.array(actions,   dtype=np.int64),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'rewards':   np.array(rewards,   dtype=np.float32),
            'dones':     np.array(dones,     dtype=np.float32),
            'kills':     int(kills),
            'score':     float(score),
        }
        self.kill_hof.append(ep)
        self.kill_hof.sort(key=lambda e: (e['kills'], e['score']), reverse=True)
        self.kill_hof = self.kill_hof[:self.max_per_group]

        self.reward_hof.append(ep)
        self.reward_hof.sort(key=lambda e: (e['score'], e['kills']), reverse=True)
        self.reward_hof = self.reward_hof[:self.max_per_group]

    def _all_episodes(self):
        seen, combined = set(), []
        for ep in self.kill_hof + self.reward_hof:
            if id(ep) not in seen:
                seen.add(id(ep))
                combined.append(ep)
        return combined

    def get_batches(self, net, device, seq_len, seqs_per_batch,
                    gamma=0.99, gae_lambda=0.95):
        """
        Yield mini-batches from HoF episodes with advantages recomputed
        using the current LSTM-equipped critic.

        Each episode is run sequentially through the LSTM (zero initial state)
        to get values. The episode is then chopped into seq_len chunks.
        Hidden states at chunk boundaries are stored and reused for training.
        """
        episodes = self._all_episodes()
        if not episodes:
            return

        all_states  = []
        all_actions = []
        all_old_lp  = []
        all_adv     = []
        all_ret     = []

        for ep in episodes:
            T = len(ep['rewards'])
            if T < seq_len:
                continue   # episode too short for even one sequence

            states_np = ep['states']   # (T, 94)

            # ── Pass full episode through LSTM to get values ────────────────
            # Also record hidden state at each chunk boundary for training.
            h, c = net.init_hidden(1, device)
            ep_values  = np.zeros(T, dtype=np.float32)
            h_at_chunk = []   # (h, c) at start of each chunk

            n_chunks = T // seq_len
            with torch.no_grad():
                for ci in range(n_chunks):
                    s, e = ci * seq_len, (ci + 1) * seq_len
                    h_at_chunk.append((h.clone(), c.clone()))
                    chunk_t = torch.FloatTensor(states_np[s:e]).unsqueeze(0).to(device)
                    trunk   = net._backbone(chunk_t)
                    lstm_out, (h, c) = net.lstm(trunk, (h, c))
                    vals = net.critic_head(lstm_out.squeeze(0)).squeeze(-1).cpu().numpy()
                    ep_values[s:e] = vals

            # ── Compute GAE advantages ──────────────────────────────────────
            rewards = ep['rewards']
            dones   = ep['dones']
            n_used  = n_chunks * seq_len
            adv     = np.zeros(n_used, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(n_used)):
                nnt   = 1.0 - dones[t]
                nv    = ep_values[t + 1] if t < n_used - 1 else 0.0
                delta = rewards[t] + gamma * nv * nnt - ep_values[t]
                last_gae = delta + gamma * gae_lambda * nnt * last_gae
                adv[t]   = last_gae
            ret = adv + ep_values[:n_used]

            # ── Store chunks with their proper initial hidden states ─────────
            for ci in range(n_chunks):
                s, e = ci * seq_len, (ci + 1) * seq_len
                all_states.append((ep['states'][s:e], h_at_chunk[ci]))
                all_actions.append(ep['actions'][s:e])
                all_old_lp.append(ep['log_probs'][s:e])
                all_adv.append(adv[s:e])
                all_ret.append(ret[s:e])

        if not all_states:
            return

        n_chunks_total = len(all_states)
        idx = np.random.permutation(n_chunks_total)

        for start in range(0, n_chunks_total, seqs_per_batch):
            b_idx = idx[start:start + seqs_per_batch]
            if len(b_idx) < seqs_per_batch // 2:
                continue

            B = len(b_idx)
            states_batch  = np.zeros((B, seq_len, STATE_SIZE), dtype=np.float32)
            actions_batch = np.zeros((B, seq_len), dtype=np.int64)
            lp_batch      = np.zeros((B, seq_len), dtype=np.float32)
            adv_batch     = np.zeros((B, seq_len), dtype=np.float32)
            ret_batch     = np.zeros((B, seq_len), dtype=np.float32)

            # Batch the initial hidden states
            h_batch = torch.zeros(1, B, LSTM_HIDDEN, device=device)
            c_batch = torch.zeros(1, B, LSTM_HIDDEN, device=device)

            for bi, ii in enumerate(b_idx):
                states_batch[bi]  = all_states[ii][0]
                h_ep, c_ep        = all_states[ii][1]
                h_batch[:, bi, :] = h_ep.squeeze(1)
                c_batch[:, bi, :] = c_ep.squeeze(1)
                actions_batch[bi] = all_actions[ii]
                lp_batch[bi]      = all_old_lp[ii]
                adv_batch[bi]     = all_adv[ii]
                ret_batch[bi]     = all_ret[ii]

            yield (
                torch.FloatTensor(states_batch).to(device),
                torch.LongTensor(actions_batch).to(device),
                torch.FloatTensor(lp_batch).to(device),
                torch.FloatTensor(adv_batch).to(device),
                torch.FloatTensor(ret_batch).to(device),
                (h_batch, c_batch),
            )

    def summary(self):
        if not self.kill_hof and not self.reward_hof:
            return "HoF: empty"
        k_kills  = [e['kills'] for e in self.kill_hof]
        r_scores = [e['score']  for e in self.reward_hof]
        n        = len(self._all_episodes())
        return (f"HoF({n})  "
                f"kills:[{min(k_kills)}-{max(k_kills)}k  best={max(k_kills)}k]  "
                f"reward:[{min(r_scores):.0f}→{max(r_scores):.0f}  best={max(r_scores):.0f}r]")
