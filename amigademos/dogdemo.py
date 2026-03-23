import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Amiga Demo")

# Set up colors
background_color = (0, 0, 0)
text_color = (255, 255, 255)

# Set up font
font_size = 100
font = pygame.font.Font(None, font_size)

# Create the text surface
text = "AMIGA"
text_surface = font.render(text, True, text_color)

# set counter to 0
counter = 0

# Get the dimensions of the text
text_width = text_surface.get_width()
text_height = text_surface.get_height()

# Calculate the starting position to center the text
start_x = (screen_width - text_width) // 2
start_y = (screen_height - text_height) // 2

# Create individual letter surfaces and positions
letters = []
letter_positions = []
letter_widths = []

# Calculate letter positions
for i, letter in enumerate(text):
    letter_surface = font.render(letter, True, text_color)
    letters.append(letter_surface)
    letter_widths.append(letter_surface.get_width())

    # Calculate position for each letter
    x = start_x + sum(letter_widths[:i])
    y = start_y
    letter_positions.append((x, y))

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # incrase counter
    counter += 1

    # adjust each letter's Y by a sine wave
    for i, letter_surface in enumerate(letters):
        letter_positions[i] = (letter_positions[i][0], letter_positions[i][1] + math.sin(counter/10 + i) * 10)

    # Fill the screen with black
    screen.fill(background_color)

    # Draw each letter at its position
    for i, letter_surface in enumerate(letters):
        screen.blit(letter_surface, letter_positions[i])

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()