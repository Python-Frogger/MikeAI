import pygame
import math
import random

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Amiga Demo - Rotating Graffiti Text")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GREY = (128, 128, 128)

# Cube vertices (8 points)
vertices = [
    (-50, -50, -50),  # 0
    (50, -50, -50),   # 1
    (50, 50, -50),     # 2
    (-50, 50, -50),     # 3
    (-50, -50, 50),    # 4
    (50, -50, 50),     # 5
    (50, 50, 50),      # 6
    (-50, 50, 50)      # 7
]

# Edges connecting vertices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# Rotation speeds (faster rotation)
rotation_speeds = [0.03, 0.02, 0.04]  # Faster rotation on all axes

# Star system
stars = []
for _ in range(100):
    # Random position in 3D space (z from 0 to 1000)
    x = random.randint(-WIDTH, WIDTH)
    y = random.randint(-HEIGHT, HEIGHT)
    z = random.randint(0, 1000)
    # Random brightness (0 = black, 255 = white)
    brightness = random.randint(100, 255)
    # Random twinkling speed
    twinkle_speed = random.uniform(0.01, 0.05)
    stars.append([x, y, z, brightness, twinkle_speed])

# Animation parameters
animation_start_time = pygame.time.get_ticks()
animation_duration = 5000  # 5 seconds

# Rotation angles
angle_x = 0
angle_y = 0
angle_z = 0

# Text rotation
text_angle = 0
text_rotation_speed = 0.02

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    current_time = pygame.time.get_ticks()
    elapsed_time = current_time - animation_start_time

    # Calculate movement in Z space and scaling
    # Use sine wave for smooth movement and scaling
    z_movement = math.sin(elapsed_time / 1000) * 100  # Move 100 units forward/backward
    scale_factor = 1 - 0.5 * math.sin(elapsed_time / 1000)  # Scale from 1 to 1.5

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(BLACK)

    # Update rotation angles
    angle_x += rotation_speeds[0]
    angle_y += rotation_speeds[1]
    angle_z += rotation_speeds[2]

    # Update text rotation
    text_angle += text_rotation_speed

    # Rotate vertices
    rotated_vertices = []
    for vertex in vertices:
        x, y, z = vertex

        # Rotate around X axis
        new_y = y * math.cos(angle_x) - z * math.sin(angle_x)
        new_z = y * math.sin(angle_x) + z * math.cos(angle_x)
        y, z = new_y, new_z

        # Rotate around Y axis
        new_x = x * math.cos(angle_y) + z * math.sin(angle_y)
        new_z = -x * math.sin(angle_y) + z * math.cos(angle_y)
        x, z = new_x, new_z

        # Rotate around Z axis
        new_x = x * math.cos(angle_z) - y * math.sin(angle_z)
        new_y = x * math.sin(angle_z) + y * math.cos(angle_z)
        x, y = new_x, new_y

        # Apply movement in Z space and scaling
        z += z_movement
        x *= scale_factor
        y *= scale_factor

        rotated_vertices.append((x, y, z))

    # Project 3D vertices to 2D screen
    projected_vertices = []
    for vertex in rotated_vertices:
        x, y, z = vertex
        # Simple perspective projection
        factor = 500 / (500 + z)
        x = x * factor + WIDTH // 2
        y = y * factor + HEIGHT // 2
        projected_vertices.append((x, y))

    # Draw edges with green color
    for edge in edges:
        start = projected_vertices[edge[0]]
        end = projected_vertices[edge[1]]
        pygame.draw.line(screen, GREEN, start, end, 2)

    # Draw stars with twinkling effect
    for i, star in enumerate(stars):
        x, y, z, brightness, twinkle_speed = star

        # Move star toward viewer (decrease z)
        z -= 2

        # Reset star if it goes too close
        if z <= 0:
            z = 1000
            x = random.randint(-WIDTH, WIDTH)
            y = random.randint(-HEIGHT, HEIGHT)

        # Update brightness for twinkling effect
        brightness = 100 + int(155 * math.sin(pygame.time.get_ticks() * twinkle_speed * 0.001))
        # Clamp brightness
        brightness = max(0, min(255, brightness))

        # Simple perspective projection for stars
        factor = 500 / (500 + z)
        screen_x = x * factor + WIDTH // 2
        screen_y = y * factor + HEIGHT // 2

        # Draw star with varying brightness
        star_color = (brightness, brightness, brightness)
        pygame.draw.circle(screen, star_color, (int(screen_x), int(screen_y)), 1)

        # Update star position
        stars[i] = [x, y, z, brightness, twinkle_speed]

    # Draw rotating graffiti text "A" and "I"

    # Draw rotating "A"
    # Define points for the "A" relative to center (center_x, center_y)
    a_center_x = WIDTH // 2 - 100  # Position to the left of "I"
    a_center_y = HEIGHT // 2

    # Base points for triangle (without rotation) - A shape
    a_points = [
        (-25, 50),   # Bottom left
        (0, -50),    # Top center
        (25, 50)     # Bottom right
    ]

    # Apply rotation to A points
    rotated_a_points = []
    a_rotation_angle = text_angle  # Use the same rotation angle for consistency

    for px, py in a_points:
        # Rotate around center (0, 0) first
        new_x = px * math.cos(a_rotation_angle) - py * math.sin(a_rotation_angle)
        new_y = px * math.sin(a_rotation_angle) + py * math.cos(a_rotation_angle)

        # Then translate to center position
        final_x = new_x + a_center_x
        final_y = new_y + a_center_y

        rotated_a_points.append((final_x, final_y))

    # Draw the "A" using lines to create the triangle shape
    pygame.draw.line(screen, WHITE, rotated_a_points[0], rotated_a_points[1], 3)  # Left slanted line
    pygame.draw.line(screen, WHITE, rotated_a_points[1], rotated_a_points[2], 3)  # Right slanted line
    pygame.draw.line(screen, WHITE, rotated_a_points[0], rotated_a_points[2], 3)  # Bottom line

    # Draw "I" with rotation (already implemented)
    i_width, i_height = 10, 60  # Thinner I
    i_center_x, i_center_y = WIDTH // 2 + 50, HEIGHT // 2

    # Vertical rectangle for I
    i_rect = pygame.Rect(0, 0, i_width, i_height)
    points = [
        (i_rect.left, i_rect.top),
        (i_rect.right, i_rect.top),
        (i_rect.right, i_rect.bottom),
        (i_rect.left, i_rect.bottom)
    ]
    # Apply rotation and translation
    rotated_i_points = []
    for px, py in points:
        new_x = px * math.cos(text_angle) - py * math.sin(text_angle)
        new_y = px * math.sin(text_angle) + py * math.cos(text_angle)
        final_x = new_x + i_center_x
        final_y = new_y + i_center_y
        rotated_i_points.append((final_x, final_y))

    # Draw I with rectangle
    pygame.draw.polygon(screen, WHITE, rotated_i_points)

    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
