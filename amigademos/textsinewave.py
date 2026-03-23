import pygame
import math

# Initialize pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Amiga Demo")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up font - using a larger font size for big letters
font = pygame.font.SysFont(None, 100)  # Size 100 for large letters

# --- CONFIGURABLE WORD ---
# The word we want to display
word = "AMIGA DEMO"

# sine wave counter for letter animation
counter = 0
counter_speed = 4

# Create individual letters with their own coordinates
# Each letter will have its own position and will be moved independently later
letters = []
start_x = WIDTH // 2 - (len(word) * 30)  # Center the text horizontally
start_y = HEIGHT // 2 - 50  # Center vertically

# Create each letter with its own starting coordinates
for i, char in enumerate(word):
    letter_x = start_x + i * 60  # Space each letter 100 pixels apart
    letter_y = start_y
    letters.append({
        'char': char,
        'original_x': letter_x,
        'original_y': letter_y,
        'x': letter_x,
        'y': letter_y,
        'angle': 0  # For circular motion
    })

# Set up the clock for controlling frame rate
    clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(BLACK)

    # Update counter for animation
    counter += counter_speed
    if counter >= 360:
        counter = 0

    # Update each letter's position with sine wave animation
    for letter_data in letters:
        letter_data['y'] = letter_data['original_y'] + math.sin(math.radians(counter + 36 * letters.index(letter_data))) * 50

    # Draw each letter at its current position
    for letter_data in letters:
        # Render the letter
        text_surface = font.render(letter_data['char'], True, WHITE)
        # Draw the letter at its current position
        screen.blit(text_surface, (letter_data['x'], letter_data['y']))

    # Update the display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

# Clean up
pygame.quit()