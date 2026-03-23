"""
MIKE 3D - Classic Amiga Demo Style
Rotating 3D letters in the spirit of 1980s demoscene
Press ESC or Q to quit
"""

import pygame
import math
import sys

# ==================== INITIALIZE ====================
pygame.init()

# Screen setup - classic Amiga-ish resolution scaled up
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MIKE 3D - Amiga Demo Style")
clock = pygame.time.Clock()

# ==================== COLORS (Amiga Palette Style) ====================
BLACK = (0, 0, 0)
# Copper bar gradient colors
COPPER_COLORS = [
    (255, 0, 0),
    (255, 128, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 128, 255),
    (128, 0, 255),
    (255, 0, 255),
]


# ==================== 3D LETTER DEFINITIONS ====================
# Each letter is defined as vertices and edges (wireframe style)
# Letters are defined in a grid roughly 3 units wide, 5 units tall

def make_letter_M():
    """Create 3D vertices and edges for letter M"""
    vertices = [
        # Left vertical bar
        (-1.5, -2.5, -0.3), (-1.5, 2.5, -0.3), (-1.0, 2.5, -0.3), (-1.0, -2.5, -0.3),
        (-1.5, -2.5, 0.3), (-1.5, 2.5, 0.3), (-1.0, 2.5, 0.3), (-1.0, -2.5, 0.3),
        # Right vertical bar
        (1.0, -2.5, -0.3), (1.0, 2.5, -0.3), (1.5, 2.5, -0.3), (1.5, -2.5, -0.3),
        (1.0, -2.5, 0.3), (1.0, 2.5, 0.3), (1.5, 2.5, 0.3), (1.5, -2.5, 0.3),
        # Left diagonal
        (-1.0, 2.5, -0.3), (0.0, 0.5, -0.3),
        (-1.0, 2.5, 0.3), (0.0, 0.5, 0.3),
        # Right diagonal
        (1.0, 2.5, -0.3), (0.0, 0.5, -0.3),
        (1.0, 2.5, 0.3), (0.0, 0.5, 0.3),
    ]
    edges = [
        # Left bar front
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Left bar back
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Left bar connections
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Right bar front
        (8, 9), (9, 10), (10, 11), (11, 8),
        # Right bar back
        (12, 13), (13, 14), (14, 15), (15, 12),
        # Right bar connections
        (8, 12), (9, 13), (10, 14), (11, 15),
        # Diagonals
        (16, 17), (18, 19), (16, 18), (17, 19),
        (20, 21), (22, 23), (20, 22), (21, 23),
    ]
    return vertices, edges


def make_letter_I():
    """Create 3D vertices and edges for letter I"""
    vertices = [
        # Vertical bar (thicker)
        (-0.4, -2.5, -0.3), (-0.4, 2.5, -0.3), (0.4, 2.5, -0.3), (0.4, -2.5, -0.3),
        (-0.4, -2.5, 0.3), (-0.4, 2.5, 0.3), (0.4, 2.5, 0.3), (0.4, -2.5, 0.3),
        # Top serif
        (-1.0, 2.5, -0.3), (-1.0, 2.0, -0.3), (1.0, 2.0, -0.3), (1.0, 2.5, -0.3),
        (-1.0, 2.5, 0.3), (-1.0, 2.0, 0.3), (1.0, 2.0, 0.3), (1.0, 2.5, 0.3),
        # Bottom serif
        (-1.0, -2.0, -0.3), (-1.0, -2.5, -0.3), (1.0, -2.5, -0.3), (1.0, -2.0, -0.3),
        (-1.0, -2.0, 0.3), (-1.0, -2.5, 0.3), (1.0, -2.5, 0.3), (1.0, -2.0, 0.3),
    ]
    edges = [
        # Vertical bar
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Top serif
        (8, 9), (9, 10), (10, 11), (11, 8),
        (12, 13), (13, 14), (14, 15), (15, 12),
        (8, 12), (9, 13), (10, 14), (11, 15),
        # Bottom serif
        (16, 17), (17, 18), (18, 19), (19, 16),
        (20, 21), (21, 22), (22, 23), (23, 20),
        (16, 20), (17, 21), (18, 22), (19, 23),
    ]
    return vertices, edges


def make_letter_K():
    """Create 3D vertices and edges for letter K"""
    vertices = [
        # Vertical bar
        (-1.5, -2.5, -0.3), (-1.5, 2.5, -0.3), (-1.0, 2.5, -0.3), (-1.0, -2.5, -0.3),
        (-1.5, -2.5, 0.3), (-1.5, 2.5, 0.3), (-1.0, 2.5, 0.3), (-1.0, -2.5, 0.3),
        # Upper diagonal (pointing up-right from middle)
        (-1.0, 0.0, -0.3), (1.5, 2.5, -0.3), (1.5, 1.8, -0.3), (-1.0, -0.5, -0.3),
        (-1.0, 0.0, 0.3), (1.5, 2.5, 0.3), (1.5, 1.8, 0.3), (-1.0, -0.5, 0.3),
        # Lower diagonal (pointing down-right from middle)
        (-1.0, -0.2, -0.3), (1.5, -2.5, -0.3), (1.5, -1.8, -0.3), (-1.0, 0.3, -0.3),
        (-1.0, -0.2, 0.3), (1.5, -2.5, 0.3), (1.5, -1.8, 0.3), (-1.0, 0.3, 0.3),
    ]
    edges = [
        # Vertical bar
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Upper diagonal
        (8, 9), (9, 10), (10, 11), (11, 8),
        (12, 13), (13, 14), (14, 15), (15, 12),
        (8, 12), (9, 13), (10, 14), (11, 15),
        # Lower diagonal
        (16, 17), (17, 18), (18, 19), (19, 16),
        (20, 21), (21, 22), (22, 23), (23, 20),
        (16, 20), (17, 21), (18, 22), (19, 23),
    ]
    return vertices, edges


def make_letter_E():
    """Create 3D vertices and edges for letter E"""
    vertices = [
        # Vertical bar
        (-1.5, -2.5, -0.3), (-1.5, 2.5, -0.3), (-1.0, 2.5, -0.3), (-1.0, -2.5, -0.3),
        (-1.5, -2.5, 0.3), (-1.5, 2.5, 0.3), (-1.0, 2.5, 0.3), (-1.0, -2.5, 0.3),
        # Top horizontal bar
        (-1.5, 2.5, -0.3), (-1.5, 2.0, -0.3), (1.5, 2.0, -0.3), (1.5, 2.5, -0.3),
        (-1.5, 2.5, 0.3), (-1.5, 2.0, 0.3), (1.5, 2.0, 0.3), (1.5, 2.5, 0.3),
        # Middle horizontal bar
        (-1.0, 0.25, -0.3), (-1.0, -0.25, -0.3), (1.2, -0.25, -0.3), (1.2, 0.25, -0.3),
        (-1.0, 0.25, 0.3), (-1.0, -0.25, 0.3), (1.2, -0.25, 0.3), (1.2, 0.25, 0.3),
        # Bottom horizontal bar
        (-1.5, -2.0, -0.3), (-1.5, -2.5, -0.3), (1.5, -2.5, -0.3), (1.5, -2.0, -0.3),
        (-1.5, -2.0, 0.3), (-1.5, -2.5, 0.3), (1.5, -2.5, 0.3), (1.5, -2.0, 0.3),
    ]
    edges = [
        # Vertical bar
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Top bar
        (8, 9), (9, 10), (10, 11), (11, 8),
        (12, 13), (13, 14), (14, 15), (15, 12),
        (8, 12), (9, 13), (10, 14), (11, 15),
        # Middle bar
        (16, 17), (17, 18), (18, 19), (19, 16),
        (20, 21), (21, 22), (22, 23), (23, 20),
        (16, 20), (17, 21), (18, 22), (19, 23),
        # Bottom bar
        (24, 25), (25, 26), (26, 27), (27, 24),
        (28, 29), (29, 30), (30, 31), (31, 28),
        (24, 28), (25, 29), (26, 30), (27, 31),
    ]
    return vertices, edges


# ==================== 3D MATH FUNCTIONS ====================
def rotate_x(vertex, angle):
    """Rotate a vertex around the X axis"""
    x, y, z = vertex
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x, y * cos_a - z * sin_a, y * sin_a + z * cos_a)


def rotate_y(vertex, angle):
    """Rotate a vertex around the Y axis"""
    x, y, z = vertex
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a)


def rotate_z(vertex, angle):
    """Rotate a vertex around the Z axis"""
    x, y, z = vertex
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)


def project(vertex, fov=256, viewer_distance=8):
    """Project 3D vertex to 2D screen coordinates"""
    x, y, z = vertex
    factor = fov / (viewer_distance + z)
    return (int(x * factor + WIDTH // 2), int(-y * factor + HEIGHT // 2))


# ==================== LETTER CLASS ====================
class Letter3D:
    def __init__(self, vertices, edges, x_offset=0):
        self.original_vertices = [(v[0] + x_offset, v[1], v[2]) for v in vertices]
        self.vertices = list(self.original_vertices)
        self.edges = edges
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0

    def rotate(self, dx, dy, dz):
        """Update rotation angles"""
        self.angle_x += dx
        self.angle_y += dy
        self.angle_z += dz

    def get_transformed_vertices(self):
        """Apply rotations and return transformed vertices"""
        transformed = []
        for v in self.original_vertices:
            # Apply rotations
            v = rotate_x(v, self.angle_x)
            v = rotate_y(v, self.angle_y)
            v = rotate_z(v, self.angle_z)
            transformed.append(v)
        return transformed

    def draw(self, surface, color):
        """Draw the letter wireframe"""
        transformed = self.get_transformed_vertices()
        projected = [project(v) for v in transformed]

        for edge in self.edges:
            start = projected[edge[0]]
            end = projected[edge[1]]
            pygame.draw.line(surface, color, start, end, 2)


# ==================== STARFIELD ====================
class Starfield:
    def __init__(self, num_stars=150):
        self.stars = []
        for _ in range(num_stars):
            self.stars.append([
                pygame.math.Vector3(
                    (pygame.time.get_ticks() % 1000) / 1000 * WIDTH - WIDTH // 2,
                    (pygame.time.get_ticks() % 777) / 777 * HEIGHT - HEIGHT // 2,
                    pygame.time.get_ticks() % 500 + 1
                ),
                (pygame.time.get_ticks() * 17) % 256  # Random seed for position
            ])
        # Reinitialize with actual random positions
        import random
        self.stars = []
        for _ in range(num_stars):
            self.stars.append({
                'x': random.randint(-WIDTH, WIDTH),
                'y': random.randint(-HEIGHT, HEIGHT),
                'z': random.randint(1, 500)
            })

    def update(self):
        for star in self.stars:
            star['z'] -= 3
            if star['z'] <= 0:
                star['x'] = pygame.math.Vector2(WIDTH, HEIGHT).x * (0.5 - __import__('random').random())
                star['y'] = pygame.math.Vector2(WIDTH, HEIGHT).y * (0.5 - __import__('random').random())
                star['z'] = 500

    def draw(self, surface):
        for star in self.stars:
            if star['z'] > 0:
                factor = 256 / star['z']
                x = int(star['x'] * factor + WIDTH // 2)
                y = int(star['y'] * factor + HEIGHT // 2)
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    brightness = min(255, int(255 * (1 - star['z'] / 500)))
                    size = max(1, int(3 * (1 - star['z'] / 500)))
                    pygame.draw.circle(surface, (brightness, brightness, brightness), (x, y), size)


# ==================== COPPER BARS (Classic Amiga Effect) ====================
def draw_copper_bars(surface, offset):
    """Draw horizontal copper gradient bars"""
    bar_height = 4
    for i in range(0, HEIGHT, bar_height * 2):
        # Calculate color based on position and offset (creates movement)
        color_index = (i // 20 + int(offset)) % len(COPPER_COLORS)
        next_index = (color_index + 1) % len(COPPER_COLORS)

        # Interpolate between colors
        t = ((i // 20 + offset) % 1)
        c1 = COPPER_COLORS[color_index]
        c2 = COPPER_COLORS[next_index]

        color = (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

        # Draw with reduced alpha effect (darker)
        dark_color = (color[0] // 4, color[1] // 4, color[2] // 4)
        pygame.draw.rect(surface, dark_color, (0, i, WIDTH, bar_height))


# ==================== CREATE LETTERS ====================
# Position letters side by side: M I K E
letter_spacing = 4.5
letters = [
    Letter3D(*make_letter_M(), x_offset=-letter_spacing * 1.5),  # M
    Letter3D(*make_letter_I(), x_offset=-letter_spacing * 0.5),  # I
    Letter3D(*make_letter_K(), x_offset=letter_spacing * 0.5),   # K
    Letter3D(*make_letter_E(), x_offset=letter_spacing * 1.5),   # E
]

# Create starfield
starfield = Starfield(200)

# ==================== MAIN LOOP ====================
running = True
time_counter = 0
copper_offset = 0

# Font for credits
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 36)

print("MIKE 3D Demo Started!")
print("Press ESC or Q to quit")

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

    # Clear screen
    screen.fill(BLACK)

    # Draw copper bars background
    draw_copper_bars(screen, copper_offset)
    copper_offset += 0.1

    # Update and draw starfield
    starfield.update()
    starfield.draw(screen)

    # Update letter rotations (each letter rotates slightly differently for effect)
    time_counter += 0.02

    for i, letter in enumerate(letters):
        # Different rotation speeds/phases for each letter
        phase = i * 0.5
        letter.angle_x = math.sin(time_counter + phase) * 0.3
        letter.angle_y = time_counter * 0.8 + phase
        letter.angle_z = math.cos(time_counter * 0.5 + phase) * 0.2

    # Draw letters with cycling colors
    for i, letter in enumerate(letters):
        # Cycle through copper colors
        color_index = (int(time_counter * 2) + i) % len(COPPER_COLORS)
        color = COPPER_COLORS[color_index]
        letter.draw(screen, color)

    # Draw title text (Amiga style)
    title_text = title_font.render("*** MIKE 3D ***", True, (255, 255, 0))
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))

    # Draw scrolling credits at bottom
    scroll_text = "GREETINGS TO ALL DEMOSCENE CODERS! ... CLASSIC AMIGA STYLE 3D ROTATING LETTERS ... CODED IN PYTHON WITH PYGAME ... PRESS ESC TO EXIT ...          "
    scroll_offset = int(time_counter * 50) % (len(scroll_text) * 10)
    display_text = scroll_text * 3
    credits = font.render(display_text, True, (0, 255, 255))
    screen.blit(credits, (-scroll_offset, HEIGHT - 30))

    # Update display
    pygame.display.flip()

    # Cap at 60 FPS
    clock.tick(60)

# ==================== CLEANUP ====================
pygame.quit()
print("Demo ended. Thanks for watching!")
sys.exit()
