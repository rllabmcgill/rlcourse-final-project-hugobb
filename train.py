from lib.env import MultiAgent
import numpy as np
from tqdm import tqdm
from lib.agent import *
from lib.utils import close_window
import os

n_agents = 2
n_landmarks= 2
n_episode = 100000
max_episode_length = 1000
rendering = False
path = 'results/'
env = MultiAgent(agent=QLearning, n_agents=n_agents, n_landmarks=n_landmarks, rendering=rendering)

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
            action = a.get_action(observation)
            reward, done, info = env.step(a, action)
            action_all_agent.append(action)
            reward_all_agent.append(reward)
            done_all_agent.append(done)
        new_observation = env.get_observation()
        for i, a in enumerate(env.agents):
            a.update(observation, action_all_agent[i], new_observation, reward_all_agent[i])
        observation = new_observation
        if all(done_all_agent):
            s = 'done in %d step.'%(t)
            if rendering:
                env.window.addstr(15,0, s)
            break

if not os.path.exists(path):
    os.makedirs(path)

for i, a in enumerate(env.agents):
    a.save(path + 'q_agent_' + str(i))
q = env.agents[0].q[:,:,0,0,4,4,4,4]
env.plot_policy(q)
