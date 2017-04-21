#!/usr/bin/env python

from lib.env.multiagent_coop import MultiAgentCoop
import numpy as np
import os
from collections import Counter
import pickle
import argparse
import json
import copy


def load_trajectory(filename, i):
    exp = pickle.load(open(filename, "r"))
    b = exp.get_batch(i, random_order=False)
    print "loaded traj of size ", len(b)
    return b

filename_1 = "res_coop/a1_vocab_size-10_n_landmarks-3_grid_size-5_experience_memory-10000_bs-128_n_agents-2_lr-0.001_algo-rpg_baseline_rec_freq_train-250_n_iter_per_train-10_n_hidden-20_gamma-0.9_max_episode_len-50_/final_experience_agent_0.p"
filename_2 = "res_coop/a1_vocab_size-10_n_landmarks-3_grid_size-5_experience_memory-10000_bs-128_n_agents-2_lr-0.001_algo-rpg_baseline_rec_freq_train-250_n_iter_per_train-10_n_hidden-20_gamma-0.9_max_episode_len-50_/final_experience_agent_1.p"

i=3

h1 = load_trajectory(filename_1, i)
h2 = load_trajectory(filename_2, i)
print "h1", h1
print "h2", h2



n_agents = 2
n_landmarks = 3
grid_size = (5,5)
V = 10

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


#reward_counts = np.zeros((n_episode, n_agents))
episode_len = np.ones((n_episode, n_agents)) * -1

save = lambda obj, fname: pickle.dump(obj, open(model_dir + "/" + fname + ".p", "w"))

def save_json(data, json_filename):
    with open(model_dir + "/" + json_filename + ".json", "w") as f:
        f.write(json.dumps(data, indent=4))
 
for i_episode in tqdm(range(n_episode)):
    observations = env.reset()
    for i in range(n_agents):
        agents[i].reset(observations[i])

    r_t = [0 for _ in range(n_agents)]
    done = [False for _ in range(n_agents)]
    last_actions  = [0 for _ in range(n_agents)]
    last_speeches  = [0 for _ in range(n_agents)]
    for t in range(1, max_episode_length):
        actions = []
        speeches = []
        # actions_env stores action in the env format
        #print "obs:", observations
        #print "r_t", r_t
        #print "last a", last_actions
        #print "last s", last_speeches
        for i, a in enumerate(agents): 
            # we don't care if done is true for one agent
            action, speech = a.sample_action(observations[i], r_t[i],
                                             last_actions[i], last_speeches[i])
            actions.append(action)
            speeches.append(speech)
        new_observations, r_t_1, done, info = env.step(actions, speeches)
        #print "rew:", r_t_1
        for i, a in enumerate(agents):
            if done[i] and episode_len[i_episode, i] < 0:
                # in this case, we have just seen done.
                # we want to update with this last time step.
                episode_len[i_episode, i] = t
            elif done[i] and episode_len[i_episode, i] <= t-2:
                continue
            a.update(observations[i], actions[i], speeches[i],
                     new_observations[i], r_t[i], done[i])
            #print "ep:", i_episode, " t:", t, " agent", i, " rew:", r_t[i], " done:", done[i]
        if all([done[i] for i in range(n_agents)]) and all([0 < episode_len[i_episode, i] <= t-2 for i in range(n_agents)]):
            #print "all done at time", t
            break
        observations = new_observations
        r_t = r_t_1
        last_actions = actions
        last_speeches = speeches
            
    for i in range(n_agents):
        if episode_len[i_episode, i] < 0:
            episode_len[i_episode, i] = t
    # end of episode: compute average reward
    #for i_agent in range(n_agents):
        #reward_counts[i_episode, i_agent] /= float(t+1)
        # divide by 0 should not happen: problem in env
    if i_episode%freq_print == 0:
        #print "from", i_episode-freq_print, "to", i_episode, ";", reward_counts.shape
        #print reward_counts[i_episode-freq_print:i_episode,:].mean(axis=0)
        print episode_len[i_episode-freq_print:i_episode,:].mean(axis=0)
    if i_episode%20000 == 0 and i_episode > 1:
        for i, a in enumerate(agents):
            filename = "debug_infos_a" + str(i) + "_e" + str(i_episode) + ".p"
            save(a.get_debug_info(), filename)
       
save_json(params, "params")
save(episode_len, "episode_len")
for i, a in enumerate(agents):
    save(a.experience, "final_experience_agent_" + str(i))
for i, a in enumerate(agents):
    if args.algo=='rpg':
        batch = a.experience.get_batch(args.bs, random_order=True)
        save(a._process_history(batch), "final_batch_agent_" + str(i))

#for i, a in enumerate(agents):
#    a.save(path + 'q_agent_' + str(i))
#q = agents[0].q[:,:,4,4,4,4,0]
#env.plot_policy(q)
