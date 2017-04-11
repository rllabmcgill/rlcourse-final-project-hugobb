from lib.env import MultiAgent
import numpy as np
from tqdm import tqdm
from lib.agent import *
from lib.utils import close_window
import os

n_landmarks= 2
n_agents = 2
n_episode = 100000
max_episode_length = 1000
path = 'results/'
grid_size = (5,5)
action_space = ['left', 'right', 'up', 'down']
epsilon = 1.

agent1 = QLearning(action_space, grid_size*(1 + n_landmarks) + (2,))
agent2 = QLearning(action_space, grid_size*(1 + n_landmarks) + (2,))
agents = [agent1, agent2]

env = MultiAgent(action_space, n_agents, grid_size=grid_size, n_landmarks=n_landmarks)

for i_episode in tqdm(range(n_episode)):
    observations = env.reset()
    for t in range(max_episode_length):
        actions = []
        for i, a in enumerate(agents):
            action = a.get_action(observations[i], epsilon)
            actions.append(action)
        new_observations, reward, done, info = env.step(actions)
        for i, a in enumerate(agents):
            a.update_q(observations[i], actions[i], new_observations[i], reward[i])
        observations = new_observations
        if all(done):
            s = 'done in %d step.'%(t)
            break
    epsilon = max(epsilon-1./n_episode, 0.1)

if not os.path.exists(path):
    os.makedirs(path)

for i, a in enumerate(agents):
    a.save(path + 'q_agent_' + str(i))
q = agents[0].q[:,:,4,4,4,4,0]
env.plot_policy(q)
