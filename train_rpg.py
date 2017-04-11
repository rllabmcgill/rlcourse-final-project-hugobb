from lib.env import MultiAgent
import numpy as np
from tqdm import tqdm
from lib.rpg_agent.rpg import RPG
from lib.utils import close_window
import os
from collections import Counter

n_landmarks= 2
n_agents = 2
n_episode = 20000
max_episode_length = 50
path = 'results/'
grid_size = (5,5)
action_space = ['left', 'right', 'up', 'down']
gamma = 0.99
lr = 0.001
bs = 128
freq_train = 25
freq_print = 200
experience_memory = 1000

action_map = {0: 'left',
              1: 'right',
              2: 'up',
              3: 'down'}

obs_space_size = grid_size*(1 + n_landmarks) + (2,) # for 2agents 2 landmarks
obs_space_size = 7 # 2 for other agent pos, 2 for 2 landmarks, 1 for own goal.
agents = [RPG(obs_space_size, len(action_space), 2, 'softmax', gamma, lr, bs,
              freq_train, experience_memory) for _ in range(n_agents)]

env = MultiAgent(action_space, n_agents, grid_size=grid_size, n_landmarks=n_landmarks)
reward_mapping = {0: -1,
                  1: 0}

# Some preprocessing functions for RPG
preproc_observations = lambda O: [np.asarray(obs_i) for obs_i in O]
postproc_action = lambda a: action_map[int(a)]
preproc_rewards = lambda R: [reward_mapping[r] for r in R]
#reward_counts = np.zeros((n_episode, n_agents))
episode_len = np.zeros(n_episode)

for i_episode in tqdm(range(n_episode)):
    observations = env.reset()
    observations = preproc_observations(observations)
    for i in range(n_agents):
        agents[i].reset(observations[i])

    r_t = [0 for _ in range(n_agents)]
    r_t = preproc_rewards(r_t)
    done = [False for _ in range(n_agents)]
    for t in range(max_episode_length):
        actions, actions_env = [], []
        # actions_env stores action in the env format
        for i, a in enumerate(agents):
            # we don't care if done is true for one agent
            action = a.sample_action(observations[i], r_t[i])
            actions.append(action)
            actions_env.append(postproc_action(action))
        new_observations, r_t_1, done, info = env.step(actions_env)
        r_t_1 = preproc_rewards(r_t_1)
        new_observations = preproc_observations(new_observations)
        for i, a in enumerate(agents):
            if done[i]:
                break
            a.update(observations[i], actions[i], new_observations[i], r_t_1[i], done[i])
        if all(done):
            s = 'done in %d step.'%(t)
            break
        observations = new_observations
        r_t = r_t_1
            
        #for i in range(n_agents):
        #    if r_t[i] > 0:
        #        reward_counts[i_episode, i] += r_t[i]
    episode_len[i_episode] = t

    # end of episode: compute average reward
    #for i_agent in range(n_agents):
        #reward_counts[i_episode, i_agent] /= float(t+1)
        # divide by 0 should not happen: problem in env
    if i_episode%freq_print == 0:
        
        #print "from", i_episode-freq_print, "to", i_episode, ";", reward_counts.shape
        #print reward_counts[i_episode-freq_print:i_episode,:].mean(axis=0)
        print episode_len[i_episode-freq_print:i_episode].mean()


if not os.path.exists(path):
    os.makedirs(path)

#for i, a in enumerate(agents):
#    a.save(path + 'q_agent_' + str(i))
#q = agents[0].q[:,:,4,4,4,4,0]
#env.plot_policy(q)
