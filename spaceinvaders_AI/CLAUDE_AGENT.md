# CLAUDE_AGENT.md — Space Invaders DQN v7 Tuning Agent
## How to use this with Claude Code

### Setup
Install Claude Code if you haven't:
```bash
npm install -g @anthropic-ai/claude-code
```
Then from your project folder:
```bash
claude
```
You're now in an agentic session. Claude Code can read files, propose edits,
and run commands. Give it this file as context with:
```
/read CLAUDE_AGENT.md
```

---

## Your role as the agent

You are monitoring a Space Invaders DQN training run. Your job is to:

1. **Read the metrics** from `training_log_v7.csv` and `training_progress_v7.png`
2. **Diagnose problems** using the patterns below
3. **Propose specific changes** to the config blocks in `train_v7.py` or `game_env_v7.py`
4. **Always ask James for approval** before editing any file
5. **Never add new reward types** — only tune existing values

---

## What you can tune (and where)

### In `train_v7.py` — `HYPERPARAMS` dict:
| Key | Safe range | Effect |
|-----|-----------|--------|
| `lr` | 0.00005 – 0.0003 | Higher = faster learning, more instability |
| `gamma` | 0.99 – 0.9995 | Higher = longer credit chain, slower convergence |
| `epsilon_decay_episodes` | 500 – 2000 | Longer = more exploration |
| `batch_size` | 32 – 128 | Larger = more stable, slower |
| `warmup_events` | 1000 – 5000 | More = better initial diversity |

### In `train_v7.py` — other tunable values:
| Variable | Effect |
|----------|--------|
| `EPSILON_CYCLE_RESET` | How much to re-explore after convergence |
| `STORE_MOVEMENT_EVERY` | How often to store movement steps (lower = more signal) |
| `MOVEMENT_REWARD` | Size of alignment reward for stored movement steps |

### In `game_env_v7.py` — `REWARDS` dict:
| Key | Current | Notes |
|-----|---------|-------|
| `kill_base` | 5.0 | Base kill reward (escalates +1 per kill) |
| `miss` | -3.0 | Flat miss penalty |
| `wasted_shot` | -2.0 | Shooting while bullet active |
| `death` | -5.0 | Keep LESS than invasion |
| `invasion` | -20.0 | Keep MORE than death |
| `drop` | -3.0 | Per bounce-and-drop event |
| `win` | 50.0 | Clearing all 40 aliens |
| `alignment` | 0.02 | Stored movement reward scale |

**Rule**: `abs(invasion) > abs(death)` always. If you raise death, raise invasion proportionally.

---

## Diagnosing problems from the CSV

The CSV columns are:
`episode, score, avg50, best_score, epsilon, buffer_size, train_steps, events_this_ep, kills, misses, drops, deaths, wasted_shots`

### Pattern 1: Shooting too eagerly (high wasted_shots, low kills/misses ratio)
- `wasted_shots` consistently > `kills` → agent is mashing shoot
- Try: increase `REWARDS['wasted_shot']` (make it more negative, e.g. -3.0 → -4.0)

### Pattern 2: Not moving toward swarm (low alignment, dying to invasion)
- `deaths` dominated by invasion (aliens reaching bottom) not contact
- avg50 plateaus, drops column rarely clearing
- Try: increase `MOVEMENT_REWARD` (0.05 → 0.1), decrease `STORE_MOVEMENT_EVERY` (5 → 3)
- OR: increase `REWARDS['drop']` (make it more negative)

### Pattern 3: Misses too high relative to kills
- `misses > kills * 2` persistently after ep 500
- Agent shooting wildly
- Try: increase `REWARDS['miss']` (more negative) moderately (-3 → -4)
- Do NOT make miss too punishing or agent stops shooting entirely

### Pattern 4: avg50 plateauing below 200
- First check: is epsilon still decaying? (epsilon column in CSV)
- If epsilon < 0.05 and avg50 still < 200 → genuine learning failure
- Try: reset epsilon to 0.5 (via `EPSILON_CYCLE_RESET`), then diagnose
- If still stuck after another 300 eps: raise `REWARDS['alignment']` (0.02 → 0.04)

### Pattern 5: High variance, occasional 400s but low avg50
- Agent knows HOW to win but doesn't do it consistently
- This is an exploration/exploitation balance problem
- Try: slow down epsilon decay (`epsilon_decay_episodes` 800 → 1200)

### Pattern 6: avg50 rising but drop events very rare
- Agent clearing aliens but not learning to avoid bounces
- Check `drops` column — if near zero, drop events aren't triggering
- This suggests aliens aren't reaching the walls (good?) or something is broken
- If avg50 > 300 and drops are rare, don't touch it — it's working without

### Pattern 7: entropy collapse symptoms (was PPO issue, but)
- avg50 suddenly drops to near zero after being decent
- Agent "forgot" how to play
- Try: increase `EPSILON_CYCLE_RESET` to 0.7, reduce `lr` slightly

---

## How to propose a change

When you've identified a problem, format your proposal like this and ask James:

```
PROPOSED CHANGE:
  File: game_env_v7.py
  Variable: REWARDS['miss']
  Current: -3.0
  Proposed: -4.5
  Reason: wasted_shots running at 2x kills since ep 300, agent mashing.
  Risk: agent may stop shooting if too punishing — monitor kills/ep after change.
  Approve? [y/n]
```

Wait for approval before editing. After approval, use str_replace to make the change.

---

## What you must NEVER do without explicit approval:
- Add new reward types or events
- Change the state vector structure
- Switch algorithm (DQN → PPO)
- Modify game physics (alien speed, bullet speed, grid layout)
- Delete or restructure the event handling logic
- Change which events are stored in the buffer

---

## Useful commands during a session

Read the last 50 lines of the log to see current progress:
```bash
tail -50 training_log_v7.csv
```

Check avg50 trend:
```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('training_log_v7.csv')))[-100:]
for r in rows[::10]:
    print(f'ep {r[\"episode\"]:>5}  avg50={r[\"avg50\"]:>6}  K={r[\"kills\"]} M={r[\"misses\"]} D={r[\"drops\"]}')
"
```

Open the latest plot:
```bash
open training_progress_v7.png   # Mac
xdg-open training_progress_v7.png  # Linux
```

---

## Understanding the event types (for diagnosis)

| Event | What it means | Buffer entry |
|-------|--------------|--------------|
| Kill | Bullet hit alien | (pre-shot state, shoot, +kill_reward, post-kill state) |
| Miss | Bullet off screen | (pre-shot state, shoot, -miss, empty state) |
| Death | Alien touches player | (pre-death state, last_action, -5, terminal) |
| Invasion | Alien reaches y=700 | (pre-invasion state, last_action, -20, terminal) |
| Drop | Swarm bounces+drops | (pre-bounce state, last_action, -3, post-drop state) |
| Wasted shot | Shoot while active | (state, shoot, -2, same state) |
| Movement | Every 5 steps if closing gap | (prev state, last_action, +small, state) |

The drop event is the "Pavlov bell": pre-bounce state shows edge column present
and swarm near wall. Post-drop shows aliens 40px lower. The network learns
to associate edge columns + wall proximity with the drop penalty.

---

## Reference: v6b performance
- v4 plateau: avg50 ≈ 200
- v5 best: avg50 ≈ 310
- v6b overnight: avg50 declined to ~120 (movement problem, directional bias)
- v7 target: avg50 > 310 consistently
