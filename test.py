from lib.env import MultiAgent
import numpy as np
import time
import curses
from lib.agent import *
from lib.utils import close_window
import os

n_landmarks = 2
n_agents = 2
n_episode = 1000
max_episode_length = 1000
timeout = 100
path = 'results/'
grid_size = (5,5)
action_space = ['left', 'right', 'up', 'down']
epsilon = 0.1

agent1 = QLearning(action_space, grid_size*(1 + n_landmarks))
agent2 = QLearning(action_space, grid_size*(1 + n_landmarks))
agents = [agent1, agent2]

env = MultiAgent(action_space, n_agents, grid_size=grid_size, n_landmarks=n_landmarks)

for i, a in enumerate(agents):
    a.load(path + 'q_agent_' + str(i)+'.npy')

for i_episode in range(n_episode):
    observations = env.reset()
    for t in range(max_episode_length):
        env.render(timeout=timeout)
        c = env.window.getch()
        if c == ord('q'):
            close_window(env.window)
            exit()
        actions = []
        for i, a in enumerate(agents):
            action = a.get_action(observations[i], epsilon)
            actions.append(action)
        observations, reward, done, info = env.step(actions)
        if all(done):
            s = 'done in %d step.'%(t)
            env.window.addstr(15,0, s)
            break

q = env.agents[0].q[:,:,4,4,4,4,0]
env.plot_policy(q)
