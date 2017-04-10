from lib.env import MultiAgent
import numpy as np
import time
import curses
from lib.agent import *
from lib.utils import close_window
import os

n_landmarks= 2
n_episode = 1000
max_episode_length = 1000
timeout = 100
path = 'results/'
grid_size = (5,5)
action_space = ['left', 'right', 'up', 'down']
epsilon = 0.1

agent1 = QLearning(action_space, grid_size*(2 + n_landmarks), goal=0)
agent2 = QLearning(action_space, grid_size*(2 + n_landmarks), goal=1)

env = MultiAgent([agent1, agent2], grid_size=grid_size, n_landmarks=n_landmarks)

for i, a in enumerate(env.agents):
    a.load(path + 'q_agent_' + str(i)+'.npy')

for i_episode in range(n_episode):
    env.reset()
    for t in range(max_episode_length):
        env.render(timeout=timeout)
        c = env.window.getch()
        if c == ord('q'):
            close_window(env.window)
            exit()
        done_all_agent = []
        observation = env.get_observation()
        for a in env.agents:
            action = a.get_action(observation, epsilon)
            reward, done, info = env.step(a, action)
            done_all_agent.append(done)
        if all(done_all_agent):
            s = 'done in %d step.'%(t)
            env.window.addstr(15,0, s)
            break

q = env.agents[0].q[:,:,0,0,4,4,4,4]
env.plot_policy(q)
