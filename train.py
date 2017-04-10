from lib.env import MultiAgent
import numpy as np
from tqdm import tqdm
from lib.agent import *
from lib.utils import close_window
import os

n_landmarks= 2
n_episode = 1000000
max_episode_length = 1000
rendering = False
path = 'results/'
grid_size = (5,5)
action_space = ['left', 'right', 'up', 'down']
epsilon = 1

def network():
    from lasagne.layers import InputLayer, DenseLayer

    l_in = InputLayer(shape=(None, 2*(n_landmarks+2)))
    l_dense1 = DenseLayer(l_in, 128)
    #l_dense2 = DenseLayer(l_dense1, 128)
    l_out = DenseLayer(l_dense1, len(action_space), nonlinearity=None)

    return l_out

agent1 = DeepQAgent(action_space, network(), goal=0)
agent2 = DeepQAgent(action_space, network(), goal=1)

env = MultiAgent([agent1, agent2], grid_size=grid_size, n_landmarks=n_landmarks)

for i_episode in tqdm(range(n_episode)):
    env.reset()
    observation = env.get_observation()
    for t in range(max_episode_length):
        if rendering:
            env.render()
            c = env.window.getch()
            if c == ord('q'):
                rendering=False
                close_window(env.window)
        done_all_agent = []
        reward_all_agent = []
        action_all_agent = []
        for a in env.agents:
            action = a.get_action(observation, epsilon)
            reward, done, info = env.step(a, action)
            action_all_agent.append(action)
            reward_all_agent.append(reward)
            done_all_agent.append(done)
        new_observation = env.get_observation()
        for i, a in enumerate(env.agents):
            a.update_q(observation, action_all_agent[i], new_observation, reward_all_agent[i], done_all_agent[i])
        observation = new_observation
        if all(done_all_agent):
            s = 'done in %d step.'%(t)
            if rendering:
                env.window.addstr(15,0, s)
            break
    epsilon = max(epsilon-1/n_episode, 0.1)

if not os.path.exists(path):
    os.makedirs(path)
