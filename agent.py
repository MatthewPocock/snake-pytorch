import torch
import random
import numpy as np
from collections import deque
from board import Board, Point
from model import CNN_QNet, QTrainer
from helper import plot
import time
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, saved_weights=None):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.001
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = CNN_QNet()
        if saved_weights:
            self.model.load_state_dict(saved_weights)
        # self.model = self.model.load_state_dict(saved_weights)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y + 1)
        point_d = Point(head.x, head.y - 1)

        dir_l = game.direction == 'W'
        dir_r = game.direction == 'E'
        dir_u = game.direction == 'N'
        dir_d = game.direction == 'S'

        body_array = np.zeros((11, 11), dtype=int)
        for p in game.snake:
            try:
                body_array[p.x, p.y] = 1
            except IndexError:
                pass

        head_array = np.zeros((11, 11), dtype=int)
        try:
            head_array[game.snake[0].x, game.snake[0].y] = 1
        except IndexError:
            pass

        neck_array = np.zeros((11, 11), dtype=int)
        try:
            neck_array[game.snake[1].x, game.snake[1].y] = 1
        except IndexError:
            pass

        food_array = np.zeros((11, 11), dtype=int)
        try:
            food_array[game.food.x, game.food.y] = 1
        except IndexError:
            pass

        collisions_li = []
        for x in range(-3, 4):
            for y in range(-3, 4):
                if not (x == 0 and y == 0):
                    p = Point(head.x + x, head.y + y)
                    collisions_li.append(game.check_collision(p))


        state = [
            # # check danger
            # game.check_collision(point_u),
            # game.check_collision(point_r),
            # game.check_collision(point_d),
            # game.check_collision(point_l),

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        # flat_head_array = head_array.ravel()
        # flat_body_array = body_array.ravel()
        # flat_food_array = food_array.ravel()
        # flat_food_array = np.array(state, dtype=int)
        # return np.concatenate((flat_head_array, flat_body_array, flat_food_array))
        # return np.array(state, dtype=int)
        # return np.concatenate((collisions_li, state))
        return np.stack((head_array, neck_array, body_array, food_array), axis=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, board):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * self.n_games)
        final_move = [0, 0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            # possible_moves = board.get_moves()
            move = random.randint(0, 3)
            while board.is_opposite(move):
                move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # print('using NN to generate move...')
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state0)
            # probabilities = torch.softmax(prediction, dim=0)
            # move = torch.multinomial(probabilities, 1).item()
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    try:
        saved_weights = torch.load('model/model.pth')
        print('loading saved model')
        agent = Agent(saved_weights=saved_weights)
    except FileNotFoundError:
        print('no stored model found, will create new model')
        agent = Agent()
    game = Board(display=False)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game)
        if final_move[0] == 1:
            move = 'N'
        elif final_move[1] == 1:
            move = 'E'
        elif final_move[2] == 1:
            move = 'S'
        else:  # final_move[3] == 1:
            move = 'W'

        # perform move and get new state
        game.direction = move
        reward, done, score = game.make_move()
        state_new = agent.get_state(game)


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()
                print('storing model...')
                torch.save(agent.model.state_dict(), f'model/model.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            # time.sleep(.5)


if __name__ == '__main__':
    train()
