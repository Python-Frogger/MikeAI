# =============================================================================
# dqn_agent_v7.py  —  Double DQN with Event-Driven Replay Buffer
# =============================================================================
#
# Architecture unchanged from v6b — it was sound.
# Code cleaned up and comments streamlined.
#
# Network: dual-path CNN
#   CNN path (70 → 128):  Conv2d(1→16) → Conv2d(16→32) → Linear(2240→128)
#   Dense path (24 → 64): Linear(24→64)
#   Merged (192 → 64 → 4 Q-values)
#
# Key algorithm choices:
#   Double DQN     — online net selects action, target net evaluates it
#                    prevents Q-value overestimation
#   Soft updates   — target net drifts slowly toward online (tau=0.001)
#                    prevents "chasing a moving target" instability
#   Huber loss     — MSE for small errors, L1 for large ones
#                    stops early training from blowing up weights
#   Grad clipping  — belt-and-braces against gradient explosions
#
# [AGENT-TUNABLE] Hyperparameters are set in train_v7.py DQNAgent() call.
# Do not hardcode defaults here — keep them as constructor args so the
# training script controls everything in one place.
#
# =============================================================================

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_ROWS    = 7
GRID_COLS    = 10
GRID_SIZE    = GRID_ROWS * GRID_COLS   # 70
CONTEXT_SIZE = 24                       # must match game_env_v7.py


# =============================================================================
# NETWORK
# =============================================================================

class DQNNetwork(nn.Module):
    """
    Dual-path CNN → 4 Q-values (left / right / shoot / nothing).

    CNN path sees spatial patterns in the 7×10 grid (alien formation,
    bullet position, player position relative to formation).

    Dense path sees global statistics (column counts, row counts,
    swarm bounding box, relative player position).

    Combining both lets the network reason spatially AND statistically.
    """

    def __init__(self, grid_size, context_size, action_size):
        super().__init__()

        # CNN path
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.grid_fc = nn.Linear(32 * GRID_ROWS * GRID_COLS, 128)

        # Dense path
        self.ctx_fc = nn.Linear(context_size, 64)

        # Merge and output
        self.merge_fc = nn.Linear(128 + 64, 64)
        self.q_head   = nn.Linear(64, action_size)

        self.relu = nn.ReLU()

        # Small initial weights → Q-values start near zero → stable early training
        nn.init.uniform_(self.q_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, x):
        grid_flat = x[:, :GRID_SIZE]   # (batch, 70)
        context   = x[:, GRID_SIZE:]   # (batch, 24)

        g = grid_flat.view(-1, 1, GRID_ROWS, GRID_COLS)
        g = self.relu(self.conv1(g))
        g = self.relu(self.conv2(g))
        g = g.view(g.size(0), -1)
        g = self.relu(self.grid_fc(g))   # (batch, 128)

        c = self.relu(self.ctx_fc(context))   # (batch, 64)

        merged = torch.cat([g, c], dim=1)
        trunk  = self.relu(self.merge_fc(merged))
        return self.q_head(trunk)   # (batch, 4)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class EventDrivenReplayBuffer:
    """
    Stores only meaningful events — kills, misses, deaths, wins,
    drop events, wasted shots, and occasional movement steps.

    Random sampling breaks temporal correlations so training batches
    look like diverse slices of experience, not correlated episode chunks.
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# AGENT
# =============================================================================

class DQNAgent:

    def __init__(self, state_size, action_size, device,
                 lr=0.0001,
                 gamma=0.999,
                 tau=0.001,
                 epsilon_start=1.0,
                 epsilon_end=0.02,
                 epsilon_decay_episodes=800,
                 batch_size=64,
                 buffer_capacity=100_000,
                 warmup_events=2_000):

        self.device       = device
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_events = warmup_events

        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes

        self.online_net = DQNNetwork(GRID_SIZE, CONTEXT_SIZE, action_size).to(device)
        self.target_net = DQNNetwork(GRID_SIZE, CONTEXT_SIZE, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.loss_fn   = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.buffer      = EventDrivenReplayBuffer(buffer_capacity)
        self.train_steps = 0
        self.total_events = 0

    # -------------------------------------------------------------------------

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(s).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)
        self.total_events += 1

    def train_step(self):
        """One gradient update. Returns {'loss': float} or None if in warmup."""
        if len(self.buffer) < self.warmup_events:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a)
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: online picks action, target evaluates it
        with torch.no_grad():
            best_a   = self.online_net(ns).argmax(dim=1, keepdim=True)
            next_q   = self.target_net(ns).gather(1, best_a).squeeze(1)
            target_q = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft target update: θ_target = τ·θ_online + (1-τ)·θ_target
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        self.train_steps += 1
        return {'loss': loss.item()}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def is_warming_up(self):
        return len(self.buffer) < self.warmup_events

    # -------------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'online_net':   self.online_net.state_dict(),
            'target_net':   self.target_net.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epsilon':      self.epsilon,
            'train_steps':  self.train_steps,
            'total_events': self.total_events,
        }, path)
        print(f"  [Saved → {path}]")

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ck['online_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.epsilon      = ck.get('epsilon',      self.epsilon_end)
        self.train_steps  = ck.get('train_steps',  0)
        self.total_events = ck.get('total_events', 0)
        print(f"  [Loaded ← {path}]  ε={self.epsilon:.3f}  steps={self.train_steps:,}")
