
import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pygame Template")

# Set up the clock for controlling frame rate
clock = pygame.time.Clock()

# Create a bouncing box
box_width = 50
box_height = 50
box_x = 100
box_y = 100
box_speed_x = 3
box_speed_y = 3

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update game state here

    # Draw everything
    screen.fill((0, 0, 0))  # Fill screen with black


    # Update box position
    box_x += box_speed_x
    box_y += box_speed_y

    # Bounce off edges
    if box_x <= 0 or box_x >= SCREEN_WIDTH - box_width:
        box_speed_x = -box_speed_x
    if box_y <= 0 or box_y >= SCREEN_HEIGHT - box_height:
        box_speed_y = -box_speed_y

    # Draw the box
    pygame.draw.rect(screen, (255, 255, 255), (box_x, box_y, box_width, box_height))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)

# Clean up and exit
pygame.quit()
sys.exit()
