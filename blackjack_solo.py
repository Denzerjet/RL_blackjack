import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import time


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


# i think I'll use pytorch to make the neural network
class LinearQNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=4, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input): #required for pytorch
        hidden_out = F.relu(self.linear1(input))
        output_out = self.linear2(hidden_out)
        return output_out

class QTrainer:
    def __init__(self, model, lr=.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # tuples
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        #1: predicted Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        #2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward() #backpropagation

        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #param to control randomness
        self.gamma = 0.9 #discount rate (can be changed, must be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = LinearQNet() #size of state is 11, output layer is 3, and the hiden layer has 256 nodes (can be changed)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) #learn how the built-in zip function works
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        #for state, action, reward, next_state, done in mini_sample:
        #   self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games #you can play around with this, the more games the smaller the epsilon
        final_move = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # .item() converts it to one number
            final_move = move

        return final_move
    
# game loop
network = LinearQNet()
trainer = QTrainer(model=network)
agent = Agent()

# initial setup
env = gym.make('Blackjack-v1', natural=False, sab=False)

observation, info = env.reset() # info is metrics and debug info etc...
done = False

# train model

print('training now')
for i in range(1000):
    state_old = observation

    action = agent.get_action(observation)

    observation, reward, terminated, truncated, info = env.step(action)

    state_new = observation

    agent.train_short_memory(state_old, action, reward, state_new, done)

    agent.remember(state_old, action, reward, state_new, done)

    if terminated or truncated:
        done = True
        observation, info = env.reset()
        continue


# env = gym.make('Blackjack-v1', natural=False, sab=False)
# observation, info = env.reset()
# done = False
# num_games = 0
# num_wins = 0
# num_losses = 0
# num_draws = 0
# for i in range(100):
#     action = agent.get_action(observation)
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         if reward == -1:
#             num_losses += 1
#         elif reward == 1:
#             num_wins += 1
#         elif reward == 0:
#             num_draws += 1
#         num_games += 1
#         done = True
#         observation, info = env.reset()

# print("Wins: " + str(num_wins) + " | Losses: " + str(num_losses) + " | Draws: " + str(num_draws))

# 64 hidden layers: Wins: 28 | Losses: 54 | Draws: 6
# 16: Wins: 36 | Losses: 55 | Draws: 2
# 4: Wins: 40 | Losses: 41 | Draws: 4
# 2: Wins: 34 | Losses: 51 | Draws: 5
# 1: Wins: 24 | Losses: 61 | Draws: 


print('testing now')
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
observation, info = env.reset()
done = False
num_games = 0
num_wins = 0
num_losses = 0
num_draws = 0
for i in range(100):
    env.render()
    time.sleep(2)
    action = agent.get_action(observation)
    if action == 0:
        print("spoingus stuck")
    elif action == 1:
        print("spoingus hit dat")
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        if reward == -1:
            print("spoingus lost")
            num_losses += 1
        elif reward == 1:
            print("SPOINGUS HIT THE BIG WIN")
            num_wins += 1
        elif reward == 0:
            print("Draw :/")
            num_draws += 1
        time.sleep(3)
        num_games += 1
        print("game " + str(num_games) + " ended")
        print("Dubs: " + str(num_wins) + " | Ls: " + str(num_losses) + " | Draws: " + str(num_draws))
        done = True
        observation, info = env.reset()



####
# snake game loop
# while True:
#     #get old state
#     state_old = agent.get_state(game)

#     #get move
#     final_move = agent.get_action(state_old)

#     #perform move and get new state
#     reward, done, score = game.play_step(final_move)
#     state_new = agent.get_state(game)

#     #train short memory
#     agent.train_short_memory(state_old, final_move, reward, state_new, done)

#     #remember
#     agent.remember(state_old, final_move, reward, state_new, done)

#     if done:
#         #train long memory (also called experience replay), plot result
#         game.reset()
#         agent.n_games += 1
#         agent.train_long_memory()

#         if score > record:
#             record = score
#             agent.model.save()

#         print('Game', agent.n_games, 'Score', score, 'Record:', record)
####


####
# see env as human
    # env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
    # observation, info = env.reset()
    # env.render()
####

####
# ex prog
    # import gymnasium as gym
    # env = gym.make("LunarLander-v2", render_mode="human")
    # observation, info = env.reset()

    # for _ in range(1000):
    #     action = env.action_space.sample()  # agent policy that uses the observation and info
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset()

    # env.close()
####

# i can render the env later and pass this to the trained agent later
# using something along the lines of 
# env = gym.make("LunarLander-v2", render_mode="human")