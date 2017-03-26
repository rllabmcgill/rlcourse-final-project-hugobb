from lib.env import MultiAgent
import numpy as np
import time
import curses
from lib.agent import *
from lib.utils import close_window
import os

n_agents = 2
n_landmarks= 2
n_episode = 1000
max_episode_length = 1000
rendering = True
timeout = 100
path = 'results/'
env = MultiAgent(agent=QLearning, n_agents=n_agents, n_landmarks=n_landmarks, rendering=rendering)
for i, a in enumerate(env.agents):
    a.load(path + 'q_agent_' + str(i)+'.npy')

for i_episode in range(n_episode):
    env.reset()
    for t in range(max_episode_length):
        if rendering:
            env.render(timeout=timeout)
            c = env.window.getch()
            if c == ord('q'):
                rendering=False
                close_window(env.window)
                exit()
        done_all_agent = []
        observation = env.get_observation()
        for a in env.agents:
            action = a.get_action(observation)
            reward, done, info = env.step(a, action)
            done_all_agent.append(done)
        if all(done_all_agent):
            s = 'done in %d step.'%(t)
            if rendering:
                env.window.addstr(15,0, s)
            break

q = env.agents[0].q[:,:,0,0,4,4,4,4]
env.plot_policy(q)
