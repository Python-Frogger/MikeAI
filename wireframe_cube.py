"""
3D Rotating Wireframe Cube - Old School Style
Using manual sin/cos calculations for rotation with perspective projection
"""

import pygame
import math

# Initialize pygame
pygame.init()

# Screen settings
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Wireframe Cube - Old School")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Classic green phosphor look

# Clock for framerate
clock = pygame.time.Clock()

# Define the 8 vertices of a unit cube centered at origin
# Each vertex is (x, y, z)
cube_vertices = [
    [-1, -1, -1],  # 0: back-bottom-left
    [ 1, -1, -1],  # 1: back-bottom-right
    [ 1,  1, -1],  # 2: back-top-right
    [-1,  1, -1],  # 3: back-top-left
    [-1, -1,  1],  # 4: front-bottom-left
    [ 1, -1,  1],  # 5: front-bottom-right
    [ 1,  1,  1],  # 6: front-top-right
    [-1,  1,  1],  # 7: front-top-left
]

# Define the 12 edges of the cube (pairs of vertex indices)
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]

# Projection settings
FOV_DISTANCE = 5  # Distance from camera to projection plane
SCALE = 150       # Scale factor for screen coordinates


def rotate_x(point, angle):
    """Rotate a 3D point around the X axis by angle (in radians)"""
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    new_y = y * cos_a - z * sin_a
    new_z = y * sin_a + z * cos_a

    return [x, new_y, new_z]


def rotate_y(point, angle):
    """Rotate a 3D point around the Y axis by angle (in radians)"""
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    new_x = x * cos_a + z * sin_a
    new_z = -x * sin_a + z * cos_a

    return [new_x, y, new_z]


def rotate_z(point, angle):
    """Rotate a 3D point around the Z axis by angle (in radians)"""
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a

    return [new_x, new_y, z]


def project_3d_to_2d(point):
    """
    Project a 3D point onto 2D screen using perspective projection.
    The camera is looking down the Z axis from a distance.
    """
    x, y, z = point

    # Move the cube away from the camera
    z_offset = z + FOV_DISTANCE

    # Prevent division by zero
    if z_offset == 0:
        z_offset = 0.001

    # Perspective projection formula:
    # projected_x = x * fov / z
    # projected_y = y * fov / z
    factor = FOV_DISTANCE / z_offset

    screen_x = x * factor * SCALE + WIDTH // 2
    screen_y = -y * factor * SCALE + HEIGHT // 2  # Flip Y for screen coords

    return (int(screen_x), int(screen_y))


def main():
    # Rotation angles (in radians)
    angle_x = 0
    angle_y = 0
    angle_z = 0

    # Rotation speeds
    speed_x = 0.02
    speed_y = 0.03
    speed_z = 0.01

    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear screen
        screen.fill(BLACK)

        # Transform all vertices
        transformed_vertices = []
        for vertex in cube_vertices:
            # Apply rotations in order: X, Y, Z
            point = vertex.copy()
            point = rotate_x(point, angle_x)
            point = rotate_y(point, angle_y)
            point = rotate_z(point, angle_z)
            transformed_vertices.append(point)

        # Project vertices to 2D
        projected_vertices = []
        for vertex in transformed_vertices:
            projected = project_3d_to_2d(vertex)
            projected_vertices.append(projected)

        # Draw edges
        for edge in cube_edges:
            start_vertex = projected_vertices[edge[0]]
            end_vertex = projected_vertices[edge[1]]
            pygame.draw.line(screen, GREEN, start_vertex, end_vertex, 2)

        # Draw vertices as small circles for that authentic look
        for vertex in projected_vertices:
            pygame.draw.circle(screen, GREEN, vertex, 4)

        # Update rotation angles
        angle_x += speed_x
        angle_y += speed_y
        angle_z += speed_z

        # Keep angles in range [0, 2*PI] to prevent overflow
        angle_x = angle_x % (2 * math.pi)
        angle_y = angle_y % (2 * math.pi)
        angle_z = angle_z % (2 * math.pi)

        # Update display
        pygame.display.flip()

        # Cap framerate at 60 FPS
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
