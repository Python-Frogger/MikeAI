import pygame
import random
import math
eL
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Welcome Back!")
clock = pygame.time.Clock()

class Particle:
    def __init__(self, x, y, color, is_rocket=False):
        self.x, self.y = x, y
        self.color = color
        self.is_rocket = is_rocket
        if is_rocket:
            self.vx = random.uniform(-1, 1)
            self.vy = random.uniform(-12, -8)
            self.size = 4
        else:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.size = random.randint(2, 4)
        self.life = 255
        self.decay = random.uniform(3, 6) if not is_rocket else 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15  # gravity
        if not self.is_rocket:
            self.life -= self.decay
            self.vx *= 0.98
            self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = max(0, min(255, int(self.life)))
            color = (*self.color[:3], alpha)
            s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (self.size, self.size), self.size)
            surface.blit(s, (self.x - self.size, self.y - self.size))

def create_explosion(x, y):
    color = random.choice([
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 200, 50), (255, 150, 200)
    ])
    return [Particle(x, y, color) for _ in range(60)]

def launch_rocket():
    x = random.randint(100, WIDTH - 100)
    color = (255, 200, 100)
    return Particle(x, HEIGHT, color, is_rocket=True)

particles = []
rockets = []
launch_timer = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            particles.extend(create_explosion(*event.pos))

    # Auto-launch rockets
    launch_timer += 1
    if launch_timer > 30:
        rockets.append(launch_rocket())
        launch_timer = random.randint(-20, 0)

    # Update rockets
    for rocket in rockets[:]:
        rocket.update()
        if rocket.vy > -2:  # explode near peak
            particles.extend(create_explosion(rocket.x, rocket.y))
            rockets.remove(rocket)

    # Update particles
    for p in particles[:]:
        p.update()
        if p.life <= 0:
            particles.remove(p)

    # Draw
    screen.fill((10, 10, 30))

    # Draw welcome text
    font = pygame.font.Font(None, 74)
    text = font.render("Welcome Back!", True, (255, 255, 255))
    screen.blit(text, (WIDTH//2 - text.get_width()//2, 50))

    for rocket in rockets:
        rocket.draw(screen)
    for p in particles:
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
