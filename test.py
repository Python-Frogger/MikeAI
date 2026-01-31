import pygame

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((200, 100))

# Define yellow color
yellow = (255, 255, 0)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill((0, 0, 0))

    # Set font and render text
    font = pygame.font.Font(None, 36)
    text = font.render("Hello World", True, yellow)
    text_rect = text.get_rect(center=(100, 50))
    screen.blit(text, text_rect)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
