import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dog Moving in Figure 8")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)

# Dog properties
dog_width = 40
dog_height = 25
dog_x = WIDTH // 2
dog_y = HEIGHT // 2

# Figure 8 parameters
figure8_width = 300
figure8_height = 200
speed = 0.002  # Slowed down speed

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(WHITE)
    
    # Calculate new position using figure 8 (lemniscate) parametric equations
    t = pygame.time.get_ticks() * speed
    x = figure8_width * math.sin(t)
    y = figure8_height * math.sin(2 * t) / 2

    # Update dog position
    dog_x = WIDTH // 2 + x
    dog_y = HEIGHT // 2 + y

    # Draw figure 8 path (for visualization)
    for i in range(0, 1000):
        t = i * 0.01
        x = figure8_width * math.sin(t)
        y = figure8_height * math.sin(2 * t) / 2
        if i == 0:
            prev_point = (WIDTH // 2 + x, HEIGHT // 2 + y)
        else:
            current_point = (WIDTH // 2 + x, HEIGHT // 2 + y)
            pygame.draw.line(screen, BLACK, prev_point, current_point, 1)
            prev_point = current_point

    # Draw dog with more dog-like shape
    # Body
    pygame.draw.ellipse(screen, BROWN, (int(dog_x - dog_width//2), int(dog_y - dog_height//2), dog_width, dog_height))

    # Head
    pygame.draw.circle(screen, BROWN, (int(dog_x), int(dog_y - dog_height//2 - 10)), 15)

    # Ears
    pygame.draw.polygon(screen, BROWN, [
        (int(dog_x - 10), int(dog_y - dog_height//2 - 20)),
        (int(dog_x - 5), int(dog_y - dog_height//2 - 30)),
        (int(dog_x - 15), int(dog_y - dog_height//2 - 25))
    ])
    pygame.draw.polygon(screen, BROWN, [
        (int(dog_x + 10), int(dog_y - dog_height//2 - 20)),
        (int(dog_x + 5), int(dog_y - dog_height//2 - 30)),
        (int(dog_x + 15), int(dog_y - dog_height//2 - 25))
    ])

    # Tail
    pygame.draw.line(screen, BROWN, (int(dog_x + dog_width//2), int(dog_y)), (int(dog_x + dog_width//2 + 20), int(dog_y - 10)), 3)

    # Legs
    pygame.draw.line(screen, BROWN, (int(dog_x - dog_width//4), int(dog_y + dog_height//2)), (int(dog_x - dog_width//4), int(dog_y + dog_height//2 + 15)), 3)
    pygame.draw.line(screen, BROWN, (int(dog_x + dog_width//4), int(dog_y + dog_height//2)), (int(dog_x + dog_width//4), int(dog_y + dog_height//2 + 15)), 3)

    # Eyes
    pygame.draw.circle(screen, YELLOW, (int(dog_x - 5), int(dog_y - dog_height//2 - 5)), 3)
    pygame.draw.circle(screen, YELLOW, (int(dog_x + 5), int(dog_y - dog_height//2 - 5)), 3)

    # Update display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()