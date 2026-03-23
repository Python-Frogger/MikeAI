import pygame
import math
import sys

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Amiga Cube Demo - Painter's Dilemma Solved")
clock = pygame.time.Clock()

# Bright Amiga color palette (classic hot colors)
AMIGA_COLORS = {
    'front': (255, 0, 0),  # Red
    'back': (0, 0, 255),  # Blue
    'right': (0, 255, 0),  # Green
    'left': (0, 255, 255),  # Cyan
    'top': (255, 0, 255),  # Magenta
    'bottom': (255, 255, 0)  # Yellow
}

# Cube vertices (centered at origin, side length 4.0)
vertices = [
    (-2.0, -2.0, -2.0),  # 0: bottom-back-left
    (-2.0, -2.0, 2.0),  # 1: bottom-front-left
    (-2.0, 2.0, -2.0),  # 2: top-back-left
    (-2.0, 2.0, 2.0),  # 3: top-front-left
    (2.0, -2.0, -2.0),  # 4: bottom-back-right
    (2.0, -2.0, 2.0),  # 5: bottom-front-right
    (2.0, 2.0, -2.0),  # 6: top-back-right
    (2.0, 2.0, 2.0)  # 7: top-front-right
]

# CORRECTED FACE DEFINITIONS (unique vertices, proper winding)
faces = [
    ([1, 3, 7, 5], 'front'),  # z=2: Red
    ([0, 2, 6, 4], 'back'),  # z=-2: Blue
    ([4, 6, 7, 5], 'right'),  # x=2: Green
    ([0, 1, 3, 2], 'left'),  # x=-2: Cyan
    ([1, 3, 7, 6], 'top'),  # y=2: Magenta
    ([0, 1, 5, 4], 'bottom')  # y=-2: Yellow
]

# Rotation angles
angle_x = 0.0
angle_y = 0.0
angle_z = 0.0
CAM_DIST = 5.0  # Camera distance from origin


def rotate_vertex(v, ax, ay, az):
    """Apply rotations: X -> Y -> Z (standard Euler order)"""
    x, y, z = v
    # Rotate around X-axis
    cos_x = math.cos(ax);
    sin_x = math.sin(ax)
    x_temp = x * cos_x - y * sin_x
    y_temp = y * cos_x + z * sin_x
    z_temp = z

    # Rotate around Y-axis
    cos_y = math.cos(ay);
    sin_y = math.sin(ay)
    x2 = x_temp * cos_y - z_temp * sin_y
    y2 = y_temp * cos_y + z_temp * sin_y
    z2 = z_temp

    # Rotate around Z-axis
    cos_z = math.cos(az);
    sin_z = math.sin(az)
    x3 = x2 * cos_z - y2 * sin_z
    y3 = x2 * sin_z + y2 * cos_z
    z3 = z2
    return (x3, y3, z3)


def project(vertex):
    """Perspective projection with proper centering and y-flip"""
    x, y, z = vertex
    depth = CAM_DIST - z
    if depth <= 0 or abs(depth) < 0.1:  # Skip behind camera
        return None

    # CRITICAL FIX: Proper centering using screen dimensions
    scale = WIDTH * 0.45  # Adjusts cube size to screen
    # Flip Y: Pygame Y grows downward, 3D Y grows upward
    screen_x = (x * scale) / depth - WIDTH // 2
    screen_y = (HEIGHT // 2) - (y * scale) / depth  # FLIP Y-AXIS
    return (screen_x, screen_y)


def solve_painter_dilemma(faces):
    """SOLVES PAINTER'S DILEMMA:
    - For each face: compute MINIMUM depth (closest vertex to camera)
    - Sort faces by min_depth ASCENDING (closest faces drawn LAST)
    - Draw in reverse order → ensures visible faces show correct color
    """
    face_data = []

    for face_indices, face_type in faces:
        min_depth = float('inf')
        projected_verts = []
        valid = True

        for idx in face_indices:
            orig_v = vertices[idx]
            rot_v = rotate_vertex(orig_v, angle_x, angle_y, angle_z)
            depth = CAM_DIST - rot_v[2]  # Depth = camera_dist - z
            if depth <= 0:
                valid = False
                break
            if depth < min_depth:
                min_depth = depth
            proj_v = project(rot_v)
            if proj_v is None:
                valid = False
                break
            projected_verts.append(proj_v)

        if not valid:
            continue

        color = AMIGA_COLORS[face_type]
        face_data.append((min_depth, projected_verts, color))

    # SORT BY MIN_DEPTH ASCENDING → DRAW REVERSE ORDER (painter algorithm)
    face_data.sort(key=lambda x: x[0])
    return [(color, vertices) for (_, vertices, color) in face_data]


# Main loop
running = True
font = pygame.font.SysFont(None, 24)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill((0, 0, 0))  # Black background

    # SOLVE PAINTER'S DILEMMA: depth-sort faces
    sorted_faces = solve_painter_dilemma(faces)

    # Draw each face in sorted order (backfaces first → frontfaces last)
    for face_color, vertices_proj in sorted_faces:
        # Convert to integers for drawing
        pts = [tuple(map(int, v)) for v in vertices_proj]
        pygame.draw.polygon(screen, face_color, pts)

    # =============== FIXED LABEL DRAWING (floats, not tuples) ===============
    # X-axis labels (left/right)
    x_positions = [-2.5, 2.5, 0.0]
    y_positions = [2.5, -2.5, 0.0]

    # Y-axis label
    z_pos = 0.0

    axis_labels = [
        ("X", AMIGA_COLORS['right']),  # Right axis color
        ("Y", AMIGA_COLORS['left']),  # Left axis color
        ("Z", AMIGA_COLORS['front']),  # Front axis color
        ("Center", (128, 128, 128))
    ]

    for i, (label, color) in enumerate(axis_labels):
        x = x_positions[i] if i < 3 else z_pos
        y = y_positions[i] if i < 3 else z_pos
        # Scale color for text readability
        r = color[0] // 128
        g = color[1] // 128
        b = color[2] // 128
        txt = font.render(f"{label}: R{r} G{g} B{b}", True, color)
        screen.blit(txt, (x + 0.5, y + 0.5))

    # Update rotation
    angle_x += 0.08
    angle_y += 0.05
    angle_z += 0.03

    pygame.display.flip()
    clock.tick(60)  # Stable 60 FPS

pygame.quit()
sys.exit()
