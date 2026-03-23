# =============================================================================
# train_v9.py  —  PPO + LSTM training loop for Space Invaders v9
# =============================================================================
#
# What's new vs v8:
#
#  LSTM HIDDEN STATE MANAGEMENT
#    A hidden state (h, c) flows through the rollout step-by-step.
#    Reset to zeros at every episode boundary (done=True).
#    The LSTM accumulates memory across the ~2500 steps of each game.
#    During training, sequences of SEQ_LEN=320 steps are fed to the LSTM.
#    Gradients flow back 320 steps — exactly one full alien oscillation cycle.
#
#  SEQUENCE MINI-BATCHES
#    v8: random 2048 frames from 262k rollout.
#    v9: 8 contiguous sequences × 320 steps = 2560 steps per mini-batch.
#    Same total gradient intensity, but LSTM can now learn from temporal patterns.
#    262144 / 320 = 819 sequences per rollout → 819/8 ≈ 102 mini-batches per epoch.
#
#  WARM-START FROM v8 WEIGHTS
#    On first run: offered the chance to load best_model_v8.pth.
#    CNN, trunk, actor head, critic head all transfer (same architecture).
#    LSTM starts fresh (new layer — no v8 equivalent).
#    125 updates of spatial learning carry over. Only temporal reasoning is new.
#
#  EVERYTHING ELSE UNCHANGED
#    Same rollout size (262,144), epochs (6), GAE, clip, entropy coef.
#    Same reward values, alive bonus, wasted shot penalty.
#    Same Hall of Fame (adapted for LSTM sequences).
#    Same quintile sampling (adapted for sequence-level weights).
#    Same checkpoint/logging/rendering infrastructure.
#
# =============================================================================

import sys
sys.path.insert(0, 'D:/PythonProjects/MikeAI/spaceinvaders_AI')

import os
import csv
import time
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pygame

from game_env_v7 import SpaceInvadersEnv, REWARDS
from ppo_agent_v9 import (ActorCritic, RolloutBuffer, HallOfFame,
                           ACTION_NAMES, SEQ_LEN, LSTM_HIDDEN)

# =============================================================================
# CONFIG  — all tunable knobs in one place
# =============================================================================

# ── Rollout ───────────────────────────────────────────────────────────────────
ROLLOUT_STEPS   = 1_048_576 # steps per PPO update (~2× prev). From update 280.
SEQ_LEN         = SEQ_LEN   # 2560 — eight full alien oscillation cycles (~full game). From ppo_agent_v9.
SEQS_PER_BATCH  = 4         # sequences per mini-batch: 4 × 2560 = 10240 steps
                             # 1048576/2560 = 409 seqs → 409/4 ≈ 102 batches per epoch
PPO_EPOCHS      = 6         # gradient epochs over each rollout

# ── PPO algorithm ─────────────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.004        # same as final v8 run
MAX_GRAD_NORM = 0.5

# ── Optimiser ─────────────────────────────────────────────────────────────────
LR              = 2.5e-4
CRITIC_LR_MULT  = 4         # critic head gets 4× LR

# ── Reward shaping ────────────────────────────────────────────────────────────
ALIVE_BONUS     = 0.003     # per step
WASTED_SHOT_PEN = REWARDS['wasted_shot']   # -2.0

# ── Training control ──────────────────────────────────────────────────────────
RENDER_EVERY    = 10
SAVE_EVERY      = 5
MAX_UPDATES     = 10_000

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR        = 'D:/PythonProjects/MikeAI/spaceinvaders_AI'
BEST_PATH       = f'{SAVE_DIR}/best_model_v9.pth'
FINAL_PATH      = f'{SAVE_DIR}/final_model_v9.pth'
CKPT_PATTERN    = f'{SAVE_DIR}/checkpoint_v9_upd{{n}}.pth'
LOG_PATH        = f'{SAVE_DIR}/training_log_v9.csv'
DIAG_PATH       = f'{SAVE_DIR}/diag_log_v9.csv'

# v8 paths — for warm-start weight transfer
BEST_PATH_V8    = f'{SAVE_DIR}/best_model_v8.pth'
FINAL_PATH_V8   = f'{SAVE_DIR}/final_model_v8.pth'

# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name}  |  VRAM: {vram:.1f} GB")
        return torch.device('cuda')
    print("[CPU] No GPU detected — training on CPU")
    return torch.device('cpu')

device = get_device()

# =============================================================================
# INIT
# =============================================================================

env = SpaceInvadersEnv(render_mode=True)
net = ActorCritic().to(device)

# Separate LR for critic head
_critic_ids  = {id(p) for p in net.critic_head.parameters()}
_actor_group = [p for p in net.parameters() if id(p) not in _critic_ids]
opt = optim.Adam([
    {'params': _actor_group,                 'lr': LR},
    {'params': net.critic_head.parameters(), 'lr': LR * CRITIC_LR_MULT},
])

buf = RolloutBuffer(ROLLOUT_STEPS, seq_len=SEQ_LEN)
hof = HallOfFame(max_episodes=40)

# Tracking
ep_num           = 0
update_num       = 0
total_steps      = 0
best_avg50       = -999.0
score_history    = collections.deque(maxlen=50)
kill_history     = collections.deque(maxlen=50)
best_avg50_kills = 0.0
running          = True

print(f"\n{'='*65}")
print(f"  Space Invaders PPO v9 — LSTM({LSTM_HIDDEN}) temporal memory")
print(f"  Rollout: {ROLLOUT_STEPS:,} steps/update  |  SeqLen: {SEQ_LEN}  |  Epochs: {PPO_EPOCHS}")
print(f"  Seqs/batch: {SEQS_PER_BATCH} × {SEQ_LEN} = {SEQS_PER_BATCH*SEQ_LEN} steps")
print(f"  Alive bonus: {ALIVE_BONUS}/step  |  Wasted shot: {WASTED_SHOT_PEN}")
print(f"  Render every {RENDER_EVERY} episodes  |  Press Q to quit")
print(f"{'='*65}\n")

# =============================================================================
# HELPERS
# =============================================================================

def save_checkpoint(path, tag=''):
    torch.save({
        'net':                   net.state_dict(),
        'optimizer':             opt.state_dict(),
        'ep_num':                ep_num,
        'update_num':            update_num,
        'total_steps':           total_steps,
        'best_avg50':            best_avg50,
        'score_history':         list(score_history),
        'kill_history':          list(kill_history),
        'best_avg50_kills':      best_avg50_kills,
        'hof_kill_episodes':     hof.kill_hof,
        'hof_reward_episodes':   hof.reward_hof,
    }, path)
    print(f"  [Saved{tag} → {os.path.basename(path)}]")


def load_checkpoint(path):
    global ep_num, update_num, total_steps, best_avg50, best_avg50_kills
    ck = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ck['net'])
    try:
        opt.load_state_dict(ck['optimizer'])
    except Exception:
        print("  [Optimizer structure changed — fresh optimizer, weights kept]")
    ep_num           = ck.get('ep_num',           0)
    update_num       = ck.get('update_num',        0)
    total_steps      = ck.get('total_steps',       0)
    best_avg50       = ck.get('best_avg50',        -999.0)
    best_avg50_kills = ck.get('best_avg50_kills',  0.0)
    for s in ck.get('score_history', []):
        score_history.append(s)
    for k in ck.get('kill_history', []):
        kill_history.append(k)
    hof.kill_hof   = ck.get('hof_kill_episodes',  [])
    hof.reward_hof = ck.get('hof_reward_episodes', [])
    print(f"  [Loaded ← {os.path.basename(path)}]  "
          f"ep={ep_num}  update={update_num}  steps={total_steps:,}  "
          f"best_avg50={best_avg50:.1f}  best_avg50_kills={best_avg50_kills:.1f}")


def warm_start_from_v8(v8_path):
    """
    Transfer compatible weights from a v8 checkpoint to v9.
    CNN, trunk, actor head, critic head all have identical shapes.
    LSTM is new — initialises fresh (orthogonal init already applied in __init__).
    """
    ck = torch.load(v8_path, map_location=device, weights_only=False)
    v8_state = ck['net']
    v9_state = net.state_dict()
    transferred, skipped = [], []
    for key in v8_state:
        if key in v9_state and v9_state[key].shape == v8_state[key].shape:
            v9_state[key] = v8_state[key]
            transferred.append(key)
        else:
            skipped.append(key)
    net.load_state_dict(v9_state)
    print(f"  [Warm-start] Transferred {len(transferred)} tensors from v8.")
    if skipped:
        print(f"  [Warm-start] Skipped (shape mismatch / new): {skipped}")
    print(f"  [Warm-start] LSTM starts fresh with orthogonal init.")


def check_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            print("\n  [Q pressed — stopping cleanly]")
            return False
    return True


def console_ep(ep_score, ep_kills, ep_steps, is_rendered):
    avg50 = np.mean(score_history) if score_history else 0.0
    tag   = ' [WATCH]' if is_rendered else ''
    print(f"  Ep {ep_num:>5}  score={ep_score:>7.1f}  kills={ep_kills:>2}  "
          f"steps={ep_steps:>4}  avg50={avg50:>7.1f}{tag}")


def console_update(ep_scores, pl, vl, ent, secs_r, secs_u):
    avg50 = np.mean(score_history) if score_history else 0.0
    star  = ' *** BEST ***' if avg50 >= best_avg50 else ''
    print(f"\n{'─'*65}")
    print(f"  UPDATE {update_num:>4}  |  eps this rollout: {len(ep_scores):>3}  "
          f"|  total eps: {ep_num:>5}  |  steps: {total_steps:,}")
    if ep_scores:
        print(f"  Scores this rollout:  mean={np.mean(ep_scores):>7.1f}  "
              f"min={np.min(ep_scores):>7.1f}  max={np.max(ep_scores):>7.1f}")
    print(f"  avg50={avg50:>7.1f}  best_avg50={best_avg50:>7.1f}{star}")
    print(f"  policy_loss={pl:.4f}  value_loss={vl:.4f}  entropy={ent:.4f}")
    print(f"  rollout={secs_r:.1f}s  update={secs_u:.1f}s")
    print(f"{'─'*65}\n")

# =============================================================================
# CHECKPOINT LOADING — v9 first, then offer v8 warm-start
# =============================================================================

_has_v9_final = os.path.exists(FINAL_PATH)
_has_v9_best  = os.path.exists(BEST_PATH)
_has_v8_best  = os.path.exists(BEST_PATH_V8)
_has_v8_final = os.path.exists(FINAL_PATH_V8)

if _has_v9_final and _has_v9_best:
    print(f"  Found v9 checkpoints:")
    print(f"    [F] {os.path.basename(FINAL_PATH)} — last clean save")
    print(f"    [B] {os.path.basename(BEST_PATH)}  — best avg50 ever")
    ans = input("  Load which? [F/b/N] ").strip().lower()
    if ans == 'b':
        load_checkpoint(BEST_PATH)
    elif ans not in ('n', ''):
        load_checkpoint(FINAL_PATH)

elif _has_v9_final:
    ans = input(f"Found {os.path.basename(FINAL_PATH)}. Resume? [y/N] ").strip().lower()
    if ans == 'y':
        load_checkpoint(FINAL_PATH)

elif _has_v9_best:
    ans = input(f"Found {os.path.basename(BEST_PATH)}. Resume? [y/N] ").strip().lower()
    if ans == 'y':
        load_checkpoint(BEST_PATH)

else:
    # No v9 checkpoint — offer v8 warm-start
    print("  No v9 checkpoint found. Starting fresh.")
    v8_source = None
    if _has_v8_best:
        v8_source = BEST_PATH_V8
    elif _has_v8_final:
        v8_source = FINAL_PATH_V8

    if v8_source:
        print(f"  Found v8 weights: {os.path.basename(v8_source)}")
        ans = input("  Warm-start v9 from v8 weights? (CNN/trunk/heads transfer, LSTM fresh) [Y/n] ").strip().lower()
        if ans not in ('n',):
            warm_start_from_v8(v8_source)
    else:
        print("  No v8 weights found either — training from scratch.")

# =============================================================================
# LOG FILE — append if resuming, fresh write if new run
# =============================================================================

_log_mode  = 'a' if update_num > 0 else 'w'
log_file   = open(LOG_PATH, _log_mode, newline='')
log_writer = csv.writer(log_file)
if _log_mode == 'w':
    log_writer.writerow([
        'update', 'episodes', 'total_steps',
        'ep_mean', 'ep_min', 'ep_max',
        'avg50', 'best_avg50',
        'mean_kills', 'max_kills', 'avg50_kills', 'best_avg50_kills',
        'policy_loss', 'value_loss', 'rel_value_loss_pct', 'entropy',
        'secs_rollout', 'secs_update',
    ])
    log_file.flush()

# Diagnostic CSV — always append across runs, one row per episode
_diag_new  = not (os.path.exists(DIAG_PATH) and os.path.getsize(DIAG_PATH) > 0)
diag_file  = open(DIAG_PATH, 'a', newline='')
diag_writer = csv.writer(diag_file)
if _diag_new:
    diag_writer.writerow(['update', 'ep_num', 'ep_score', 'ep_kills', 'ep_steps',
                          'start_p_x', 'start_alien_dir', 'start_swarm_drift'])
    diag_file.flush()

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

state = env.reset()
ep_start_p_x    = env.p_x
ep_start_dir    = 1 if env.aliens[0]['speed'] > 0 else -1
ep_start_drift  = env._swarm_drift()

ep_score      = 0.0
ep_kills      = 0
ep_steps      = 0
ep_start_step = 0
ep_records    = []
ep_kills_list = []

# ── Hidden state — carries through the episode, resets at done ────────────────
hidden = net.init_hidden(batch_size=1, device=device)

while running and update_num < MAX_UPDATES:

    # ── Rollout collection ────────────────────────────────────────────────────
    buf.reset()
    ep_records    = []
    ep_kills_list = []
    ep_start_step = 0
    # Reset hidden at start of rollout (clean slate — not mid-episode)
    hidden = net.init_hidden(batch_size=1, device=device)
    t_rollout_start = time.time()

    for step in range(ROLLOUT_STEPS):

        if step % 500 == 0:
            pygame.event.pump()
        if step % 5000 == 0:
            if not check_quit():
                running = False
                break

        # Get action — hidden state flows forward step-by-step
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, value, hidden = net.get_action(state_t, hidden)

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Reward shaping
        if info['wasted_shot']:
            reward += WASTED_SHOT_PEN
        reward += ALIVE_BONUS

        buf.add(state, action, reward, value, log_prob, done)
        state = next_state
        total_steps += 1

        ep_score += reward
        ep_steps += 1
        if info.get('resolution_type') == 'kill':
            ep_kills += 1

        # Render
        is_rendered = (RENDER_EVERY > 0 and ep_num % RENDER_EVERY == 0)
        if is_rendered:
            col_counts = [
                sum(1 for a in env.aliens if a['alive'] and a['col'] == c)
                for c in range(8)
            ]
            overlay = {
                'episode':       ep_num,
                'epsilon':       0.0,
                'buffer_events': buf.ptr,
                'warmup_done':   True,
                'train_steps':   total_steps,
                'best_score':    int(best_avg50),
                'avg50':         np.mean(score_history) if score_history else 0.0,
                'col_counts':    col_counts,
                'recent_scores': list(score_history),
            }
            env.render(fps_cap=0, overlay=overlay)

        # Episode end
        if done:
            s, e = ep_start_step, buf.ptr
            ep_records.append((s, e, ep_score))
            ep_kills_list.append(ep_kills)
            hof.offer(buf.states[s:e], buf.actions[s:e], buf.log_probs[s:e],
                      buf.rewards[s:e], buf.dones[s:e], ep_kills, ep_score)
            ep_start_step = buf.ptr
            score_history.append(ep_score)
            kill_history.append(ep_kills)
            console_ep(ep_score, ep_kills, ep_steps, is_rendered)
            diag_writer.writerow([update_num, ep_num, round(ep_score, 1), ep_kills, ep_steps,
                                   round(ep_start_p_x, 1), ep_start_dir, round(ep_start_drift, 1)])
            diag_file.flush()
            ep_num   += 1
            ep_score  = 0.0
            ep_kills  = 0
            ep_steps  = 0
            state          = env.reset()
            ep_start_p_x   = env.p_x
            ep_start_dir   = 1 if env.aliens[0]['speed'] > 0 else -1
            ep_start_drift = env._swarm_drift()
            # ── Reset LSTM hidden state at episode boundary ─────────────────
            hidden = net.init_hidden(batch_size=1, device=device)

    if not running:
        break

    secs_rollout = time.time() - t_rollout_start

    # Partial episode at rollout end
    if ep_start_step < buf.ptr:
        ep_records.append((ep_start_step, buf.ptr, ep_score))
        ep_kills_list.append(ep_kills)

    # ── GAE ───────────────────────────────────────────────────────────────────
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        # Bootstrap with a single-step LSTM forward (hidden carries from rollout)
        x = state_t.unsqueeze(1)
        trunk = net._backbone(x)
        lstm_out, _ = net.lstm(trunk, hidden)
        last_value = net.critic_head(lstm_out.squeeze(1)).item()

    buf.compute_gae(last_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    # ── Calculating screen ────────────────────────────────────────────────────
    t_update_start = time.time()
    net.train()

    # Pre-calculate total batches so we can show a progress bar
    _n_seqs         = buf.ptr // SEQ_LEN
    _batches_per_ep = max(1, _n_seqs // SEQS_PER_BATCH)
    _total_batches  = PPO_EPOCHS * _batches_per_ep
    _avg  = float(np.mean(score_history)) if score_history else 0.0
    _avgk = float(np.mean(kill_history))  if kill_history  else 0.0

    _disp = pygame.display.get_surface()
    _f1   = pygame.font.SysFont(None, 52) if _disp else None
    _f2   = pygame.font.SysFont(None, 30) if _disp else None
    _bar_w = 680

    def _draw_progress(batch_num, epoch_num):
        if not _disp:
            return
        pct     = batch_num / max(_total_batches, 1)
        elapsed = time.time() - t_update_start
        eta_s   = (elapsed / max(pct, 0.001)) * (1.0 - pct)
        eta_min = int(eta_s // 60)
        eta_sec = int(eta_s % 60)

        _disp.fill((10, 10, 20))
        _disp.blit(_f1.render(f"PPO v9  —  Update {update_num + 1}  calculating...",
                              True, (80, 200, 100)), (60, 220))
        _disp.blit(_f2.render(f"avg50={_avg:.1f}   avg50_kills={_avgk:.1f}   ep={ep_num}",
                              True, (160, 160, 160)), (60, 285))
        _disp.blit(_f2.render(f"LSTM({LSTM_HIDDEN})  SeqLen={SEQ_LEN}  {_n_seqs} seqs",
                              True, (100, 100, 180)), (60, 315))
        # Progress bar
        bar_fill = int(_bar_w * pct)
        pygame.draw.rect(_disp, (40, 40, 40),  (60, 360, _bar_w, 28))
        pygame.draw.rect(_disp, (60, 180, 90), (60, 360, bar_fill, 28))
        pygame.draw.rect(_disp, (80, 80, 80),  (60, 360, _bar_w, 28), 1)
        _disp.blit(_f2.render(
            f"Epoch {epoch_num}/{PPO_EPOCHS}   batch {batch_num}/{_total_batches}"
            f"   {pct*100:.1f}%   ETA {eta_min}m {eta_sec:02d}s",
            True, (200, 200, 200)), (60, 400))
        _disp.blit(_f2.render(
            f"elapsed {int(elapsed//60)}m {int(elapsed%60):02d}s",
            True, (120, 120, 120)), (60, 430))
        pygame.display.flip()

    _draw_progress(0, 1)
    pygame.event.pump()

    # ── PPO update — sequence mini-batches ────────────────────────────────────
    policy_losses, value_losses, entropies = [], [], []
    _batch_count = 0

    for epoch in range(PPO_EPOCHS):
        for batch in buf.get_sequences(SEQS_PER_BATCH, device, ep_records=ep_records):
            states_b, actions_b, old_lp_b, adv_b, returns_b = batch

            # Flatten seq dimension for advantage normalisation
            # adv_b shape: (B, seq_len) → normalise over all B*seq_len values
            adv_flat = adv_b.reshape(-1)
            adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            adv_b_norm = adv_norm.reshape(adv_b.shape)

            # Zero hidden state at sequence start (truncated BPTT)
            init_h = net.init_hidden(states_b.shape[0], device)

            # Re-evaluate stored sequences with current policy + LSTM
            new_lp, values_b, entropy = net.evaluate(states_b, actions_b, init_h)

            # Flatten for PPO loss
            old_lp_flat   = old_lp_b.reshape(-1)
            adv_flat_norm = adv_b_norm.reshape(-1)
            returns_flat  = returns_b.reshape(-1)

            # PPO clipped surrogate
            ratio  = (new_lp - old_lp_flat).exp()
            surr1  = ratio * adv_flat_norm
            surr2  = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat_norm
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_b, returns_flat)

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
            _batch_count += 1
            _draw_progress(_batch_count, epoch + 1)
            pygame.event.pump()

    # ── Hall of Fame pass ─────────────────────────────────────────────────────
    hof_pl, hof_vl = [], []
    for batch in hof.get_batches(net, device, SEQ_LEN, SEQS_PER_BATCH,
                                  gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        states_b, actions_b, old_lp_b, adv_b, returns_b, hof_hidden = batch

        adv_flat = adv_b.reshape(-1)
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        adv_b_norm = adv_norm.reshape(adv_b.shape)

        new_lp, values_b, entropy = net.evaluate(states_b, actions_b, hof_hidden)

        old_lp_flat   = old_lp_b.reshape(-1)
        adv_flat_norm = adv_b_norm.reshape(-1)
        returns_flat  = returns_b.reshape(-1)

        ratio  = (new_lp - old_lp_b.reshape(-1)).exp()
        surr1  = ratio * adv_flat_norm
        surr2  = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat_norm
        pl     = -torch.min(surr1, surr2).mean()
        vl     = F.mse_loss(values_b, returns_flat)
        loss   = pl + VALUE_COEF * vl - ENTROPY_COEF * entropy.mean()

        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
        opt.step()
        hof_pl.append(pl.item())
        hof_vl.append(vl.item())

    net.eval()
    secs_update = time.time() - t_update_start

    # ── Logging ───────────────────────────────────────────────────────────────
    update_num += 1
    ep_scores_this_rollout = buf.episode_stats()
    avg50  = float(np.mean(score_history)) if score_history else 0.0
    pl     = float(np.mean(policy_losses))
    vl     = float(np.mean(value_losses))
    ent    = float(np.mean(entropies))
    ep_mean_this  = float(np.mean(ep_scores_this_rollout)) if ep_scores_this_rollout else 0.0
    rel_vl = float(np.sqrt(vl) / ep_mean_this * 100) if ep_mean_this > 0 else 0.0
    mean_kills    = float(np.mean(ep_kills_list))   if ep_kills_list else 0.0
    max_kills     = float(np.max(ep_kills_list))    if ep_kills_list else 0.0
    avg50_kills   = float(np.mean(kill_history))    if kill_history  else 0.0
    if avg50_kills > best_avg50_kills:
        best_avg50_kills = avg50_kills

    hof_str = hof.summary()
    console_update(ep_scores_this_rollout, pl, vl, ent, secs_rollout, secs_update)
    print(f"  {hof_str}")

    log_writer.writerow([
        update_num, ep_num, total_steps,
        round(np.mean(ep_scores_this_rollout), 2) if ep_scores_this_rollout else 0,
        round(np.min(ep_scores_this_rollout),  2) if ep_scores_this_rollout else 0,
        round(np.max(ep_scores_this_rollout),  2) if ep_scores_this_rollout else 0,
        round(avg50, 2), round(best_avg50, 2),
        round(mean_kills, 2), round(max_kills, 1), round(avg50_kills, 2), round(best_avg50_kills, 2),
        round(pl, 5), round(vl, 5), round(rel_vl, 3), round(ent, 5),
        round(secs_rollout, 1), round(secs_update, 1),
    ])
    log_file.flush()

    # ── Save best ─────────────────────────────────────────────────────────────
    if avg50 > best_avg50:
        best_avg50 = avg50
        save_checkpoint(BEST_PATH, tag=' BEST')

    # ── Periodic checkpoint ───────────────────────────────────────────────────
    if update_num % SAVE_EVERY == 0:
        ckpt = CKPT_PATTERN.format(n=update_num)
        save_checkpoint(ckpt)

# =============================================================================
# CLEANUP
# =============================================================================

print("\nTraining ended.")
save_checkpoint(FINAL_PATH, tag=' FINAL')
log_file.close()
diag_file.close()
pygame.quit()
print("Done.")
print(f"  Resume: run train_v9.py — it will find {os.path.basename(FINAL_PATH)} automatically.")
