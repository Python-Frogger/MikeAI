import numpy as np
import random
import matplotlib.pyplot as plt


# --- THE BRAIN CLASS ---
# This class acts as the AI's memory. It's like a notebook where it writes down
# how many points it gets for every possible guess.
class ToolKitAI:
    def __init__(self):
        # The Q-Table is a grid. 7 rows (dice 1-6) and 2 columns (Lower or Higher).
        # We start with all zeros because the AI knows nothing at the beginning.
        self.q_table = np.zeros((7, 2))

        # Exploration is the "Curiosity" level.
        # 1.0 means it will guess randomly 100% of the time to explore the rules.
        self.exploration = 1.0

    def get_action(self, base):
        # This is where the AI decides: "Should I explore or use my brain?"
        if random.random() < self.exploration:
            return random.randint(0, 1)  # Pick a random guess (0=Lower, 1=Higher)

        # If not exploring, look at the notebook (Q-table) and pick the side
        # that has the higher number (the most points).
        return np.argmax(self.q_table[base])

    def train(self, base, action, points):
        # 1. We look at what we THOUGHT we would get (the old value in the notebook)
        old_prediction = self.q_table[base, action]

        # 2. We calculate the "Error" (The difference between the result and our guess)
        # If we got 3 points but expected 0, error is +3 (A happy surprise!)
        # If we got 0 points but expected 3, error is -3 (A disappointment!)
        error = points - old_prediction

        # 3. We update the notebook by a small step (Learning Rate = 0.2)
        # This is the "Pencil" or "Eraser" line:
        self.q_table[base, action] = old_prediction + (0.2 * error)


# --- VISUAL SETUP ---
plt.ion()  # "Interactive On" - lets the graph update while the code runs.
fig, (ax_chart, ax_brain) = plt.subplots(1, 2, figsize=(12, 5))
brain = ToolKitAI()
streak = 0
max_streak = 0
base = 0


# --- THE MAIN GAME LOOP ---
keep_running = True
while keep_running:
    try:
        # STEP 1: The Setup
        # Roll the first die (the Base).
        #base = random.randint(1, 6)
        if base < 6:
            base += 1
        else:
            base = 1

        # Ask the AI for its guess.
        action = brain.get_action(base)
        guess_text = "HIGHER" if action == 1 else "LOWER"

        # STEP 2: Show the AI's Guess on Screen
        ax_chart.clear()
        ax_chart.set_xlim(0, 10);
        ax_chart.set_ylim(0, 10);
        ax_chart.axis('off')

        # This shows how much the AI is using its logic vs. guessing.
        focus_percent = 100 - (brain.exploration * 100)
        ax_chart.text(5, 9, f"AI BRAIN FOCUS: {focus_percent:.0f}%", fontsize=10, ha='center')
        ax_chart.text(5, 7.5, f"BASE DIE: {base}", fontsize=22, ha='center', fontweight='bold')
        ax_chart.text(5, 6, f"AI GUESS: {guess_text}", fontsize=16, ha='center', color='blue')

        plt.pause(0.8)  # Wait so we can see the guess before the rolls happen

        # STEP 3: The Challenge (Roll 3 dice)
        points = 0
        rolls = []
        for i in range(3):
            r = random.randint(1, 6)
            rolls.append(r)
            # If AI guessed Higher and the roll is higher, give a point.
            if (action == 1 and r > base) or (action == 0 and r < base):
                points += 1

        # Track the win streak (Winning is getting 2 or 3 points).
        if points >= 2:
            streak += 1
            max_streak = max(streak, max_streak)
        else:
            streak = 0

        # STEP 4: Show the Results
        ax_chart.text(5, 4, f"ROLLS: {rolls}", fontsize=15, ha='center')
        ax_chart.text(5, 2.5, f"POINTS: {points}/3", fontsize=24, ha='center', fontweight='bold')
        ax_chart.text(5, 1, f"STREAK: {streak} (MAX: {max_streak})", fontsize=12, ha='center')

        # STEP 5: The "Aha!" Moment
        # The AI compares its guess to the points it got and updates its notebook.
        brain.train(base, action, points)

        # STEP 6: Update the Heatmap (The AI's Mental Map)
        ax_brain.clear()
        # 'gray_r' makes 0 points White and 3 points Black.
        # This shows us exactly what the AI thinks is the best move for every die.
        ax_brain.imshow(brain.q_table[1:], cmap='gray_r', aspect='auto', vmin=0, vmax=3)
        ax_brain.set_title("AI MEMORY (Black = Best Move)")
        ax_brain.set_xticks([0, 1])
        ax_brain.set_xticklabels(["LOWER", "HIGHER"])
        ax_brain.set_yticks(range(6))
        ax_brain.set_yticklabels([1, 2, 3, 4, 5, 6])
        ax_brain.set_ylabel("If the Base Die is:")

        plt.pause(0.6)  # Pause so the brain doesn't flicker too fast

    except Exception as e:
        print(f"Dojo Closed: {e}")
        keep_running = False