import pygame
from pygame.locals import *
import sys
import math

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("3D Cube Demo")

# Define colors
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# Define the cube vertices
vertices = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1)
]

# Define the cube edges
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Define the rotation angles
x_angle = y_angle = z_angle = 0

# Function to rotate a point around the X-axis
def rotate_x(point, angle):
    x, y, z = point
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    new_y = y * cos_angle - z * sin_angle
    new_z = y * sin_angle + z * cos_angle
    return (x, new_y, new_z)

# Function to rotate a point around the Y-axis
def rotate_y(point, angle):
    x, y, z = point
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    new_x = x * cos_angle + z * sin_angle
    new_z = -x * sin_angle + z * cos_angle
    return (new_x, y, new_z)

# Function to rotate a point around the Z-axis
def rotate_z(point, angle):
    x, y, z = point
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle
    return (new_x, new_y, z)

# Function to project a 3D point onto the 2D screen
def project(point):
    x, y, z = point
    scale = 500 / (z + 10)
    return (int(screen_width / 2 + x * scale), int(screen_height / 2 - y * scale))

# Main loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    # Rotate the vertices
    rotated_vertices = []
    for vertex in vertices:
        x_rotated = rotate_x(vertex, x_angle)
        y_rotated = rotate_y(x_rotated, y_angle)
        z_rotated = rotate_z(y_rotated, z_angle)
        rotated_vertices.append(z_rotated)

    # Draw the edges
    for edge in edges:
        point1 = project(rotated_vertices[edge[0]])
        point2 = project(rotated_vertices[edge[1]])
        pygame.draw.line(screen, white, point1, point2, 2)

    # Update the angles
    x_angle += 0.05
    y_angle += 0.03
    z_angle += 0.07

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
