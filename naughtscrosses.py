import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 700
GRID_SIZE = 3
CELL_SIZE = 150
LINE_WIDTH = 15
CIRCLE_WIDTH = 15
CROSS_WIDTH = 20
CIRCLE_RADIUS = 60
SPACE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Naughts and Crosses")

class Game:
    def __init__(self):
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.player = 1  # Player 1 starts
        self.game_over = False
        self.font = pygame.font.SysFont('Arial', 30)
        self.small_font = pygame.font.SysFont('Arial', 24)
        self.winner = None
        
    def make_move(self, row, col):
        if self.board[row][col] == 0 and not self.game_over:
            self.board[row][col] = self.player
            self.check_game_state()
            if not self.game_over:
                self.player = 2 if self.player == 1 else 1
    
    def check_game_state(self):
        # Check rows
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] != 0:
                self.game_over = True
                self.winner = self.board[row][0]
                return
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != 0:
                self.game_over = True
                self.winner = self.board[0][col]
                return
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.game_over = True
            self.winner = self.board[0][0]
            return
            
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.game_over = True
            self.winner = self.board[0][2]
            return
            
        # Check for draw
        if all(self.board[row][col] != 0 for row in range(3) for col in range(3)):
            self.game_over = True
            self.winner = 0  # Draw
    
    def reset_game(self):
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.player = 1
        self.game_over = False
        self.winner = None
    
    def draw_board(self):
        # Draw background
        screen.fill(WHITE)
        
        # Calculate grid position to center it
        grid_width = CELL_SIZE * 3
        grid_height = CELL_SIZE * 3
        grid_x = (WIDTH - grid_width) // 2
        grid_y = HEIGHT - 150 - grid_height
        
        # Draw grid lines
        pygame.draw.line(screen, BLACK, (grid_x, grid_y), 
                          (grid_x + grid_width, grid_y), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x, grid_y + CELL_SIZE), 
                          (grid_x + grid_width, grid_y + CELL_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x, grid_y + CELL_SIZE * 2), 
                          (grid_x + grid_width, grid_y + CELL_SIZE * 2), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x, grid_y), 
                          (grid_x, grid_y + grid_height), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x + CELL_SIZE, grid_y), 
                          (grid_x + CELL_SIZE, grid_y + grid_height), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x + CELL_SIZE * 2, grid_y), 
                          (grid_x + CELL_SIZE * 2, grid_y + grid_height), LINE_WIDTH)
        # Add the missing lines:
        pygame.draw.line(screen, BLACK, (grid_x + grid_width, grid_y), 
                        (grid_x + grid_width, grid_y + grid_height), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (grid_x, grid_y + grid_height), 
                        (grid_x + grid_width, grid_y + grid_height), LINE_WIDTH)
        
        # Draw numbers on grid (1-9)
        number = 1
        for row in range(3):
            for col in range(3):
                text = self.small_font.render(str(number), True, BLACK)
                text_rect = text.get_rect(center=(grid_x + col * CELL_SIZE + CELL_SIZE//2, 
                                              grid_y + row * CELL_SIZE + CELL_SIZE//2))
                screen.blit(text, text_rect)
                number += 1
        
        # Draw X's and O's
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 1:  # Player 1 (O)
                    pygame.draw.circle(screen, RED, (grid_x + col * CELL_SIZE + CELL_SIZE//2, 
                                                         grid_y + row * CELL_SIZE + CELL_SIZE//2), 
                                       CIRCLE_RADIUS, CIRCLE_WIDTH)
                elif self.board[row][col] == 2:  # Player 2 (X)
                    pygame.draw.line(screen, BLUE, 
                                (grid_x + col * CELL_SIZE + SPACE, 
                                 grid_y + row * CELL_SIZE + SPACE),
                                (grid_x + (col + 1) * CELL_SIZE - SPACE, 
                                 grid_y + (row + 1) * CELL_SIZE - SPACE), 
                                CROSS_WIDTH)
                    pygame.draw.line(screen, BLUE, 
                                  (grid_x + (col + 1) * CELL_SIZE - SPACE, 
                                   grid_y + row * CELL_SIZE + SPACE),
                                  (grid_x + col * CELL_SIZE + SPACE, 
                                   grid_y + (row + 1) * CELL_SIZE - SPACE), 
                                  CROSS_WIDTH)
        
        # Draw status text
        if self.game_over:
            if self.winner == 0:
                text = self.font.render("Game Over: Draw!", True, BLACK)
            else:
                text = self.font.render(f"Player {self.winner} Wins!", True, BLACK)
        else:
            text = self.font.render(f"Player {self.player}'s Turn", True, BLACK)
        
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 50))
        screen.blit(text, text_rect)
        
        # Draw replay button
        pygame.draw.rect(screen, GREEN, (WIDTH//2 - 100, HEIGHT - 100, 200, 50))
        button_text = self.small_font.render("Play Again", True, BLACK)
        button_rect = button_text.get_rect(center=(WIDTH//2, HEIGHT - 75))
        screen.blit(button_text, button_rect)
        
        # Draw player indicators
        player1_text = self.small_font.render("Player 1 (O): RED", True, RED)
        player2_text = self.small_font.render("Player 2 (X): BLUE", True, BLUE)
        screen.blit(player1_text, (20, 20))
        screen.blit(player2_text, (WIDTH - player2_text.get_width() - 20, 20))

def main():
    game = Game()
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and not game.game_over:
                # Handle keyboard inputs for moves (1-9)
                if event.key == pygame.K_1:
                    game.make_move(0, 0)
                elif event.key == pygame.K_2:
                    game.make_move(0, 1)
                elif event.key == pygame.K_3:
                    game.make_move(0, 2)
                elif event.key == pygame.K_4:
                    game.make_move(1, 0)
                elif event.key == pygame.K_5:
                    game.make_move(1, 1)
                elif event.key == pygame.K_6:
                    game.make_move(1, 2)
                elif event.key == pygame.K_7:
                    game.make_move(2, 0)
                elif event.key == pygame.K_8:
                    game.make_move(2, 1)
                elif event.key == pygame.K_9:
                    game.make_move(2, 2)
            
            if event.type == pygame.KEYDOWN and game.game_over:
                # Handle restart with 'R' key
                if event.key == pygame.K_r:
                    game.reset_game()
        
        # Draw everything
        game.draw_board()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
