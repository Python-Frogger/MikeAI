# =============================================================================
# game_env_v7.py  —  Space Invaders environment (clean rewrite)
# =============================================================================
#
# What changed from v6b:
#
#  1. DROP EVENTS — the "Pavlov bell"
#     When aliens bounce off a wall and drop 40px, the training loop gets
#     a pre_drop_state and post_drop_state so the agent learns:
#     "edge column alive + swarm at wall → drop happens → penalty"
#     Previously, the agent only saw the saliva (lower aliens) not the bell
#     (edge column causing the bounce).
#
#  2. WASTED SHOT stored as a real event
#     Pressing shoot while bullet is already flying now stores a proper
#     (state, shoot, -WASTED_SHOT, same_state, False) transition.
#     Previously it was -0.05 and silently discarded.
#
#  3. MOVEMENT TOWARD SWARM tracked per step
#     step() returns info['alignment_delta'] — positive means player
#     moved closer to swarm centre this step.  Training loop uses this
#     to store occasional movement events with small alignment rewards.
#
#  4. INVASION penalty is worse than DEATH penalty
#     Invasion -20, Death -5.  Encourages fighting over hiding.
#
#  5. Reward values live in REWARDS dict — easy for agent to tune.
#
# State vector (94 numbers — same structure as v6b, 2 new context features):
#
#   [0..69]   7×10 unified grid → CNN path
#               rows 0-4: aliens (1=alive, 0=dead)
#               row 5:    bullet column (one-hot)
#               row 6:    player column (one-hot)
#   [70]      player_x normalised
#   [71]      bullet_active (0/1)
#   [72]      bullet_x normalised
#   [73]      bullet_y normalised
#   [74..77]  swarm bounding box (left, right, top, bottom) normalised
#   [78]      alien_direction (0=left, 1=right)
#   [79..86]  column_counts[0..7] normalised /5
#   [87..91]  row_counts[0..4] normalised /8
#   [92]      player_x - swarm_centre_x (relative horizontal)
#   [93]      p_y - swarm_bottom_y (relative vertical / threat distance)
#
# =============================================================================

import pygame
import numpy as np
import random

# --- Colours ---
BG     = (0,   0,   0)
RED    = (255, 0,   0)
BLUE   = (0,   0,   255)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN  = (0,   200, 0)
ORANGE = (255, 140, 0)

SCREEN_W = 800
SCREEN_H = 800

ALIEN_COLS = 8
ALIEN_ROWS = 5
MAX_ALIENS = ALIEN_COLS * ALIEN_ROWS   # 40

GRID_ROWS = 7    # 5 alien + 1 bullet + 1 player
GRID_COLS = 10   # 8 alien cols + 2 overflow zones

# =============================================================================
# REWARD CONFIG — agent can tune these values, not the structure
# =============================================================================
# [AGENT-TUNABLE] Adjust values here. Do not add or remove keys.

REWARDS = {
    'kill_base':    8.0,    # Base kill reward; total = kill_base + kills_so_far
    'miss':         0.0,    # No penalty for missing — reward for hitting is enough signal
    'wasted_shot': -2.0,    # Shoot while bullet active (stored in buffer) — reduced from -4.0
    'death':       -5.0,    # Alien touches player
    'invasion':   -20.0,    # Alien reaches player y-level (worse than death)
    'drop':        -3.0,    # Base drop penalty — scaled by proximity (close swarm = costlier)
    'win':         50.0,    # All 40 aliens cleared
    'alignment':    0.02,   # Reward per step for moving toward swarm centre
    'mash':        -0.05,   # Shoot while bullet active — NOT stored (UI feel only)
}

# =============================================================================
# ENVIRONMENT
# =============================================================================

class SpaceInvadersEnv:

    def __init__(self, render_mode=False):
        self.render_mode = render_mode

        self.p_width       = 40
        self.p_height      = 35
        self.bullet_width  = 10
        self.bullet_height = 30

        self.action_space = 4   # 0=left, 1=right, 2=shoot, 3=nothing
        self.grid_size    = GRID_ROWS * GRID_COLS   # 70
        self.context_size = 24
        self.state_size   = self.grid_size + self.context_size  # 94

        if not pygame.get_init():
            pygame.init()

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Space Invaders — DQN v7")
            self.font_large = pygame.font.SysFont(None, 36)
            self.font_small = pygame.font.SysFont(None, 22)
        else:
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))

        self.clock = pygame.time.Clock()
        self.reset()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _make_aliens(self):
        aliens = []
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                aliens.append({
                    "row": row, "col": col,
                    "x":   10 + col * 80,
                    "y":   10 + row * 70,
                    "width": 40, "height": 40,
                    "speed": 1,
                    "alive": True,
                })
        return aliens

    def _swarm_drift(self):
        """How far has the swarm drifted from its start x?"""
        for a in self.aliens:
            if a["alive"]:
                return a["x"] - (10 + a["col"] * 80)
        return 0.0

    def _map_x_to_grid_col(self, x_pos, drift):
        """Map screen x to grid column 0-9 (0 and 9 are overflow zones)."""
        centers = [10 + c * 80 + 20 + drift for c in range(ALIEN_COLS)]
        if x_pos < centers[0] - 40:
            return 0
        if x_pos > centers[-1] + 40:
            return 9
        return int(np.argmin([abs(x_pos - cc) for cc in centers])) + 1

    def _swarm_centre_x(self):
        live = [a for a in self.aliens if a["alive"]]
        if not live:
            return SCREEN_W / 2
        return (min(a["x"] for a in live) + max(a["x"] + a["width"] for a in live)) / 2

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self):
        self.aliens      = self._make_aliens()
        self.alien_count = MAX_ALIENS

        # Fair random start: teleport player, fast-forward aliens same distance
        centre_x = SCREEN_W // 2 - self.p_width // 2
        target_x = random.randint(0, SCREEN_W - self.p_width)
#         target_x = centre_x # REMOVE THE RANDOM START
        walk_steps = int(abs(target_x - centre_x) / 4.5)
        for _ in range(walk_steps):
            for a in self.aliens:
                if a["alive"]:
                    a["x"] += 1.25 * a["speed"]
                    if a["x"] >= 760 or a["x"] <= 0:
                        for b in self.aliens:
                            if b["alive"]:
                                b["speed"] *= -1
                                b["y"]     += 40
                        break
        self.p_x = target_x

# REMOVED FOR FIXED START TEST
        # Randomise direction after fast-forward
        if random.random() < 0.5:
            for a in self.aliens:
                a["speed"] = -1

        self.p_y           = 700
        self.bullet_x      = 0
        self.bullet_y      = -100
        self.bullet_active = False
        self.done          = False
        self.score         = 0
        self.steps         = 0
        self.last_reward   = 0.0

        return self.get_state()

    # =========================================================================
    # STATE
    # =========================================================================

    def get_state(self):
        state = np.zeros(self.state_size, dtype=np.float32)
        grid  = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        live  = [a for a in self.aliens if a["alive"]]

        # Alien rows (0-4): alive = 1.0
        for a in self.aliens:
            if a["alive"]:
                grid[a["row"], a["col"] + 1] = 1.0

        # Player (row 6) and bullet (row 5) relative to formation
        if live:
            drift = self._swarm_drift()
            grid[6, self._map_x_to_grid_col(self.p_x + self.p_width / 2, drift)] = 1.0
            if self.bullet_active:
                grid[5, self._map_x_to_grid_col(self.bullet_x + self.bullet_width / 2, drift)] = 1.0
        else:
            grid[6, 5] = 1.0

        state[:self.grid_size] = grid.flatten()

        # Context features
        o = self.grid_size
        state[o+0] = self.p_x / SCREEN_W
        state[o+1] = 1.0 if self.bullet_active else 0.0
        state[o+2] = self.bullet_x / SCREEN_W if self.bullet_active else 0.0
        state[o+3] = self.bullet_y / SCREEN_H if self.bullet_active else 0.0

        if live:
            state[o+4] = min(a["x"] for a in live) / SCREEN_W
            state[o+5] = max(a["x"] + a["width"] for a in live) / SCREEN_W
            state[o+6] = min(a["y"] for a in live) / SCREEN_H
            state[o+7] = max(a["y"] + a["height"] for a in live) / SCREEN_H
            state[o+8] = 1.0 if live[0]["speed"] > 0 else 0.0

        for col in range(ALIEN_COLS):
            count = sum(1 for a in self.aliens if a["alive"] and a["col"] == col)
            state[o+9+col] = count / ALIEN_ROWS

        for row in range(ALIEN_ROWS):
            count = sum(1 for a in self.aliens if a["alive"] and a["row"] == row)
            state[o+17+row] = count / ALIEN_COLS

        if live:
            cx = self._swarm_centre_x()
            state[o+22] = (self.p_x + self.p_width / 2 - cx) / SCREEN_W
            state[o+23] = (self.p_y - max(a["y"] + a["height"] for a in live)) / SCREEN_H

        return state

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action):
        """
        Returns: (next_state, reward, done, info)

        info keys:
          bullet_fired       True if bullet was just launched
          bullet_resolved    True if bullet hit alien OR flew off screen
          resolution_type    'kill' | 'miss' | None
          event_reward       reward for buffer storage (kill/miss amount)
          drop_event         True if aliens dropped this step
          pre_drop_state     state captured BEFORE the drop
          drop_penalty       penalty amount for drop event
          wasted_shot        True if agent shot while bullet was active
          alignment_delta    positive = player moved toward swarm centre this step
        """
        self.steps += 1
        reward = 0.0

        info = {
            'bullet_fired':    False,
            'bullet_resolved': False,
            'resolution_type': None,
            'event_reward':    0.0,
            'drop_event':      False,
            'pre_drop_state':  None,
            'drop_penalty':    0.0,
            'wasted_shot':     False,
            'alignment_delta': 0.0,
        }

        # Track alignment before action (for movement reward)
        prev_gap = abs((self.p_x + self.p_width / 2) - self._swarm_centre_x())

        # ---------------------------------------------------------------
        # ACTION
        # ---------------------------------------------------------------
        if action == 0:
            self.p_x -= 4.5
        elif action == 1:
            self.p_x += 4.5
        elif action == 2:
            if not self.bullet_active:
                self.bullet_x      = self.p_x + self.p_width / 2 - self.bullet_width / 2
                self.bullet_y      = self.p_y - 5
                self.bullet_active = True
                info['bullet_fired'] = True
            else:
                # Wasted shot — bullet already in flight
                # Small mash penalty for feel (not stored); wasted_shot flag
                # triggers storage of a proper event in the training loop
                reward += REWARDS['mash']
                info['wasted_shot'] = True

        self.p_x = max(0, min(self.p_x, SCREEN_W - self.p_width))

        # Alignment delta: did we close the gap to the swarm?
        new_gap = abs((self.p_x + self.p_width / 2) - self._swarm_centre_x())
        info['alignment_delta'] = prev_gap - new_gap   # positive = closer

        # ---------------------------------------------------------------
        # MOVE BULLET
        # ---------------------------------------------------------------
        if self.bullet_active:
            self.bullet_y -= 7

        # ---------------------------------------------------------------
        # BULLET OFF SCREEN — MISS
        # ---------------------------------------------------------------
        if self.bullet_active and self.bullet_y <= 0:
            miss_penalty = REWARDS['miss']
            reward += miss_penalty
            self.bullet_active = False
            info['bullet_resolved'] = True
            info['resolution_type'] = 'miss'
            info['event_reward']    = miss_penalty

        # ---------------------------------------------------------------
        # DETECT IMPENDING BOUNCE — capture pre-drop state BEFORE moving aliens
        # This is the "bell" — edge column present, swarm at wall.
        # We store this state so the agent learns the cause, not just the effect.
        # ---------------------------------------------------------------
        if any(a["alive"] and (a["x"] + 1.25 * a["speed"] >= 760 or
                                a["x"] + 1.25 * a["speed"] <= 0)
               for a in self.aliens):
            info['pre_drop_state'] = self.get_state()

        # ---------------------------------------------------------------
        # MOVE ALIENS
        # ---------------------------------------------------------------
        bounce = False
        for a in self.aliens:
            if a["alive"]:
                a["x"] += 1.25 * a["speed"]
                if a["x"] >= 760 or a["x"] <= 0:
                    bounce = True

        if bounce:
            for a in self.aliens:
                if a["alive"]:
                    a["speed"] *= -1
                    a["y"]     += 40
            # Distance-scaled drop penalty: closer swarm = costlier drop.
            # Normalised so penalty = REWARDS['drop'] at game-start distance (~370px).
            # Teaches: clear bottom row (raises swarm) → future drops cheaper.
            # Teaches: don't let swarm drift down → penalty grows exponentially.
            _live     = [a for a in self.aliens if a["alive"]]
            _bot      = max(a["y"] + a["height"] for a in _live) if _live else self.p_y
            _distance = max(10, self.p_y - _bot)
            _drop_pen = -(abs(REWARDS['drop']) * 370.0) / _distance
            info['drop_event']   = True
            info['drop_penalty'] = _drop_pen
            reward += _drop_pen

        # ---------------------------------------------------------------
        # COLLISION DETECTION
        # ---------------------------------------------------------------
        bullet_rect = pygame.Rect(self.bullet_x, self.bullet_y,
                                  self.bullet_width, self.bullet_height)
        p_rect = pygame.Rect(self.p_x, self.p_y, self.p_width, self.p_height)

        for a in self.aliens:
            if not a["alive"]:
                continue
            a_rect = pygame.Rect(a["x"], a["y"], a["width"], a["height"])

            # Kill
            if self.bullet_active and bullet_rect.colliderect(a_rect):
                a["alive"]         = False
                self.alien_count  -= 1
                self.score        += 10
                self.bullet_active = False

                kills_so_far = MAX_ALIENS - self.alien_count
                kill_reward  = REWARDS['kill_base'] + kills_so_far
                reward      += kill_reward

                info['bullet_resolved'] = True
                info['resolution_type'] = 'kill'
                info['event_reward']    = kill_reward

            # Death — check alive again (bullet may have just killed this alien)
            if a["alive"] and p_rect.colliderect(a_rect):
                reward    += REWARDS['death']
                self.done  = True

        # Invasion
        for a in self.aliens:
            if a["alive"] and a["y"] >= self.p_y:
                reward    += REWARDS['invasion']
                self.done  = True

        # Win
        if self.alien_count == 0:
            reward    += REWARDS['win']
            self.done  = True

        self.last_reward = reward
        return self.get_state(), reward, self.done, info

    # =========================================================================
    # RENDER
    # =========================================================================

    def render(self, fps_cap=0, overlay=None):
        if not self.render_mode:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BG)

        for a in self.aliens:
            if a["alive"]:
                pygame.draw.rect(self.screen, BLUE,
                                 (a["x"], a["y"], a["width"], a["height"]))

        pygame.draw.rect(self.screen, RED,
                         (self.p_x, self.p_y, self.p_width, self.p_height))

        gun_x = self.p_x + self.p_width / 2
        pygame.draw.line(self.screen, YELLOW,
                         (gun_x, self.p_y + 10), (gun_x, self.p_y - 5))

        if self.bullet_active:
            pygame.draw.rect(self.screen, WHITE,
                             (self.bullet_x, self.bullet_y,
                              self.bullet_width, self.bullet_height))

        # Column guides
        live = [a for a in self.aliens if a["alive"]]
        if live:
            drift = self._swarm_drift()
            for c in range(ALIEN_COLS):
                if any(a["alive"] and a["col"] == c for a in self.aliens):
                    cx = 10 + c * 80 + 20 + drift
                    pygame.draw.line(self.screen, (30, 30, 30),
                                     (cx, 600), (cx, 690), 1)

        # HUD bar
        pygame.draw.rect(self.screen, (20, 20, 20), (0, 755, SCREEN_W, 45))
        hud = self.font_small.render(
            f"Score: {self.score}  |  Aliens: {self.alien_count}  |  "
            f"Steps: {self.steps}  |  Reward: {self.last_reward:+.2f}",
            True, WHITE)
        self.screen.blit(hud, (10, 768))

        if overlay:
            self._draw_overlay(overlay)

        pygame.display.flip()
        if fps_cap > 0:
            self.clock.tick(fps_cap)
        return True

    def _draw_overlay(self, info):
        panel = pygame.Surface((295, 260), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 175))
        self.screen.blit(panel, (5, 5))
        y = 12

        def txt(text, colour=WHITE):
            nonlocal y
            t = self.font_small.render(text, True, colour)
            self.screen.blit(t, (15, y))
            y += 20

        eps = info.get('epsilon', 1.0)
        eps_col = (255, int(100 + 155 * (1 - eps)), 50)

        txt(f"DQN v7 — Ep {info.get('episode', '?')}")
        txt(f"ε: {eps:.3f}  |  Buffer: {info.get('buffer_events', 0):,}", eps_col)
        txt(f"Best: {info.get('best_score', 0)}  |  Avg50: {info.get('avg50', 0.0):.1f}", YELLOW)

        warmup_done = info.get('warmup_done', False)
        if warmup_done:
            txt(f"Training  |  Steps: {info.get('train_steps', 0):,}", GREEN)
        else:
            txt(f"Warmup  {info.get('buffer_events',0)}/{info.get('warmup_size', 2000)}", (150,150,150))

        # Column bar chart
        txt("Column counts (edge = orange):", (180,180,180))
        y -= 4
        col_counts = info.get('col_counts', [5]*8)
        bar_w, gap, max_h = 24, 3, 28
        for i, cnt in enumerate(col_counts):
            h = int((cnt / 5) * max_h)
            colour = ORANGE if i in (0, 7) else (80, 120, 220)
            bx = 15 + i * (bar_w + gap)
            if h > 0:
                pygame.draw.rect(self.screen, colour, (bx, y + max_h - h, bar_w, h))
            pygame.draw.rect(self.screen, (60,60,60), (bx, y, bar_w, max_h), 1)
        y += max_h + 6

        txt("S = slow   F = fast", (150,150,150))

        # Recent score bars
        recent = info.get('recent_scores', [])
        if recent:
            txt("Last 20 scores:", (180,180,180))
            y -= 4
            max_val = max(max(recent[-20:]), 1)
            for i, s in enumerate(recent[-20:]):
                h = int((s / max_val) * 40)
                ratio = i / max(len(recent[-20:]) - 1, 1)
                colour = (30, int(80 + 175 * ratio), int(200 * (1-ratio) + 55))
                bx = 15 + i * 12
                if h > 0:
                    pygame.draw.rect(self.screen, colour, (bx, y + 40 - h, 10, h))
