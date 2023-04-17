import random
from collections import namedtuple
import pygame
import time
from enum import Enum


Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    North = 0
    East = 1
    South = 2
    West = 3


class Board:
    def __init__(self, width=11, height=11, display=False):
        self.show_display = display
        if self.show_display:
            self.display = BoardDisplay()
        self.height = height
        self.width = width
        self.is_over = False
        self.snake = [Point(self.height//2, self.width//2)] * 3
        self.direction = 'E'
        self.food = self.spawn_food()
        self.score = 0
        self.frame_iteration = 0

    def spawn_food(self):
        all_points = {Point(x, y) for x in range(self.width) for y in range(self.height)}
        available_points = list(all_points - set(self.snake))

        if not available_points:
            return None

        return random.choice(available_points)

    def check_collision(self, point):
        if point in self.snake[1:]:
            return True
        elif not self.width > point.x >= 0:
            return True
        elif not self.height > point.y >= 0:
            return True
        else:
            return False

    def make_move(self):
        self.frame_iteration += 1
        reward = 0
        # move head forward
        head = self.snake[0]
        if self.direction == 'E':
            self.snake.insert(0, Point(head.x+1, head.y))
        if self.direction == 'N':
            self.snake.insert(0, Point(head.x, head.y+1))
        if self.direction == 'W':
            self.snake.insert(0, Point(head.x-1, head.y))
        if self.direction == 'S':
            self.snake.insert(0, Point(head.x, head.y-1))

        # check food and remove tail
        new_head = self.snake[0]
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward += 10
        else:
            self.snake.pop()

        # check for collisions
        if self.check_collision(new_head):
            self.is_over = True
            reward -= 10
        # self.is_over = self.check_collision(new_head)

        # check for max frame iteration
        if self.frame_iteration > 100 * len(self.snake):
            self.is_over = True

        if self.is_over:
            reward -= 5

        if self.show_display:
            self.display.update_display(self)

        return reward, self.is_over, self.score

    def reset(self):
        self.is_over = False
        self.snake = [Point(self.height//2, self.width//2)] * 3
        self.direction = 'E'
        self.food = self.spawn_food()
        self.score = 0
        self.frame_iteration = 0

    def get_moves(self):
        head = self.snake[0]
        valid_moves = [0, 0, 0, 0]
        if not self.check_collision(Point(head.x, head.y + 1)):
            valid_moves[0] = 1
        if not self.check_collision(Point(head.x + 1, head.y)):
            valid_moves[1] = 1
        if not self.check_collision(Point(head.x, head.y - 1)):
            valid_moves[2] = 1
        if not self.check_collision(Point(head.x - 1, head.y)):
            valid_moves[3] = 1
        return valid_moves

    def is_opposite(self, dir):
        if dir == 0 and self.direction == 'S':
            return True
        if dir == 2 and self.direction == 'N':
            return True
        if dir == 1 and self.direction == 'W':
            return True
        if dir == 3 and self.direction == 'E':
            return True
        else:
            return False




class BoardDisplay:
    def __init__(self):
        pygame.init()
        self.window_w = 640
        self.window_h = 640
        self.display = pygame.display.set_mode((self.window_w, self.window_h))

        # rgb colors
        self.WHITE = (200, 200, 200)
        self.RED = (255, 50, 30)
        self.BLUE1 = (70, 130, 255)
        self.BLUE2 = (0, 100, 255)
        self.BLACK = (0, 0, 0)

        self.font = pygame.font.SysFont('arial', 25)

    def update_display(self, board):
        self.display.fill(self.BLACK)

        for pt in board.snake:
            pygame.draw.rect(self.display, self.BLUE1, pygame.Rect(pt.x * (self.window_w / board.width),
                                                                   pt.y * (self.window_h / board.height),
                                                                   self.window_w / board.width,
                                                                   self.window_h / board.height))

        pygame.draw.rect(self.display, self.RED, pygame.Rect(board.food.x * (self.window_w / board.width),
                                                        board.food.y * (self.window_h / board.height),
                                                        self.window_w / board.width, self.window_h / board.height))

        text = self.font.render("Score: " + str(board.score), True, self.WHITE)

        display_surface = pygame.display.get_surface()
        self.display.blit(pygame.transform.flip(display_surface, False, True), dest=(0, 0))

        self.display.blit(text, [0, 0])
        pygame.display.flip()
        time.sleep(.1)

