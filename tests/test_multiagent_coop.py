#!/usr/bin/env python

import numpy as np

from lib.env.multiagent_coop import MultiAgentCoop
from lib.rpg_agent.random_agent import RandomAgent

def test_multiagent_coop():
    n_agents = 4
    n_landmarks = 4
    grid_size = (5,5)
    V = 5

    env = MultiAgentCoop(n_agents, n_landmarks, grid_size, V)
    obs_size = env.observation_space_size()
    # obs space: agent id + position of itself + pos of all landmarks + goal
    # where each pos is 2 values and goal is 2 values (agent id, landmark id)

    print "env observation space size:", obs_size
    assert(obs_size == n_agents + 2 + n_landmarks * 2 + n_agents + n_landmarks)

    print "env action space size:", env.action_space_size()
    agents = [RandomAgent(env.action_space_size()) for _ in range(n_agents)]
    env.render()

    for j in range(20):
        env.reset()
        cum_return = 0

        for t in range(100):
            actions = [a.sample_action(None, None) for a in agents]
            speeches = [np.random.choice(V) for _ in agents]
            o, r, done, _ = env.step(actions, speeches)
            assert(len(set(r)) == 1) # all rewards are shared by agents
            assert(len(set(done)) == 1) # episode ends at the same time for all 
            cum_return += r[0]
            #env.render()
            if np.all(done):
                break
        assert(cum_return == n_agents or not np.all(done) and cum_return < n_agents)
