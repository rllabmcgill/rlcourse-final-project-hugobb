#!/usr/bin/env python

import numpy as np

from env.multiagent_coop import MultiAgentCoop
from lib.rpg_agent.random_agent import RandomAgent

n_agents = 2
n_landmarks = 2
grid_size = (2,2)

env = MultiAgentCoop(n_agents, n_landmarks, grid_size)
agents = [RandomAgent(env.action_space_size()) for _ in range(n_agents)]

done = [False for i in range(n_agents)]
for t in range(1000):
    actions = [a.sample_action(None, None) for a in agents]
    o, r, done, _ = env.step(actions)
    print o
    print r
    print done
    if np.all(done):
        break
print done
print "stopped at", t
