# =============================================================================
# watch_v9.py  —  Watch the trained PPO v9 agent play Space Invaders
#
# No training. No gradients. No rollout buffer. Just the agent playing.
# Press Q or close the window to quit.
# =============================================================================

import os
import torch
import pygame

from game_env_v7 import SpaceInvadersEnv
from ppo_agent_v9 import ActorCritic

# ── Config ────────────────────────────────────────────────────────────────────

SAVE_DIR    = 'D:/PythonProjects/MikeAI/spaceinvaders_AI'
FINAL_PATH  = f'{SAVE_DIR}/final_model_v9.pth'
BEST_PATH   = f'{SAVE_DIR}/best_model_v9.pth'

FPS    = 240     # viewing speed — lower = slower, 0 = unlimited
GREEDY = False   # False = sample from policy (more natural), True = always pick highest-prob action

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load model — prefer final (latest save), fall back to best ────────────────

if os.path.exists(FINAL_PATH):
    MODEL_PATH = FINAL_PATH
elif os.path.exists(BEST_PATH):
    MODEL_PATH = BEST_PATH
else:
    raise FileNotFoundError(f"No v9 model found in {SAVE_DIR}")

net = ActorCritic().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
net.load_state_dict(ckpt['net'])
net.eval()

print(f"Loaded : {MODEL_PATH}")
print(f"  update={ckpt.get('update_num', '?')}  "
      f"best_avg50={ckpt.get('best_avg50', '?'):.1f}")
print(f"  Device: {DEVICE}  |  Mode: {'greedy' if GREEDY else 'stochastic'}")
print("─" * 60)
print("Q / close window = quit")
print("─" * 60)

# ── Environment ───────────────────────────────────────────────────────────────

env = SpaceInvadersEnv(render_mode=True)

# ── Tracking ──────────────────────────────────────────────────────────────────

ep_num        = 0
running_kills = 0

# ── Main loop ─────────────────────────────────────────────────────────────────

running = True
while running:
    state     = env.reset()
    ep_reward = 0.0
    done      = False

    # Reset LSTM hidden state at the start of each episode
    hidden = net.init_hidden(batch_size=1, device=DEVICE)

    while not done:

        # Quit / key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = done = True

        if not running:
            break

        # Pick action — get_action handles LSTM state internally
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if GREEDY:
            with torch.no_grad():
                x = state_t.unsqueeze(1)
                trunk = net._backbone(x)
                lstm_out, hidden = net.lstm(trunk, hidden)
                logits = net.actor_head(lstm_out.squeeze(1))
                action = int(logits.argmax(dim=-1).item())
        else:
            action, _, _, hidden = net.get_action(state_t, hidden)

        state, reward, done, info = env.step(action)
        ep_reward += reward
        env.render(fps_cap=FPS)

    if not running:
        break

    ep_num        += 1
    kills          = 40 - env.alien_count
    running_kills += kills

    result = "CLEAR!" if kills == 40 else f"{kills}/40"
    print(f"  Ep {ep_num:4d}  |  kills={result:8s}  |  "
          f"game_score={env.score:5d}  |  rl_reward={ep_reward:8.1f}  |  "
          f"avg_kills={running_kills / ep_num:5.1f}")

pygame.quit()
print("─" * 60)
print(f"Watched {ep_num} episodes  |  "
      f"avg kills: {running_kills / max(ep_num, 1):.1f}")
