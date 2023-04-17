import pygame
from board import Board, Point
import time

SPEED = 4

# rgb colors
WHITE = (200, 200, 200)
RED = (255, 50, 30)
BLUE1 = (70, 130, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGame:
    def __init__(self, window_w=640, window_h=640, board_w=11, board_h=11):
        self.window_w = window_w
        self.window_h = window_h
        self.board_w = board_w
        self.board_h = board_h
        self.display = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Snake')

        self.board = Board(board_w, board_h)
        self.clock = pygame.time.Clock()
        turn = 0

    def get_input(self, wait=False):
        w = True
        while w:
            w = wait
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    w = False
                    if event.key == pygame.K_LEFT:
                        self.board.direction = 'W'
                    elif event.key == pygame.K_RIGHT:
                        self.board.direction = 'E'
                    elif event.key == pygame.K_UP:
                        self.board.direction = 'N'
                    elif event.key == pygame.K_DOWN:
                        self.board.direction = 'S'

    def make_step(self):
        self.board.make_move()
        self.update_ui()

    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.board.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * (self.window_w/self.board_w),
                                                              pt.y * (self.window_h/self.board_h),
                                                              self.window_w/self.board_w,
                                                              self.window_h/self.board_h))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.board.food.x * (self.window_w/self.board_w),
                                                        self.board.food.y * (self.window_h/self.board_h),
                                                        self.window_w/self.board_w, self.window_h/self.board_h))

        text = font.render("Score: " + str(self.board.score), True, WHITE)

        display_surface = pygame.display.get_surface()
        self.display.blit(pygame.transform.flip(display_surface, False, True), dest=(0, 0))

        self.display.blit(text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    font = pygame.font.SysFont('arial', 25)
    game = SnakeGame()

    # await for first input
    game.update_ui()
    game.get_input(wait=True)
    game.clock.tick(SPEED)

    # game loop
    while True:
        game.get_input()
        game.make_step()
        game.clock.tick(SPEED)

        if game.board.is_over:
            break

    print('Final Score', game.board.score)

    pygame.quit()
