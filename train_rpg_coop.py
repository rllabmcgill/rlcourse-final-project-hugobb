#!/usr/bin/env python

from lib.env.multiagent_coop import MultiAgentCoop
import numpy as np
from tqdm import tqdm
from lib.rpg_agent.random_agent import RandomAgent

from lib.rpg_agent.rpg_ep_com import RPGCommunicate
from lib.rpg_agent.rpg import RPG

import os
from collections import Counter
import pickle
import argparse
import json
import copy


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-m', dest='model_dir', type=str, required=True,
                    help='Directory of the gym model')
parser.add_argument('-n', dest='n_episodes', type=int, required=True,
                    help='# episodes')
parser.add_argument('--freq_train', dest='freq_train', type=int, 
                    help='train every "freq_train" episodes', default=200)
parser.add_argument('--bs', dest='bs', type=int, 
                    help='Batch size', default=128)
parser.add_argument('--n_hidden', dest='n_hidden', type=int, 
                    help='n hidden units', default=18)
parser.add_argument('--n_iter_per_train', dest='n_iter_per_train', type=int, 
                    help='# iterations per training', default=10)
parser.add_argument('--max_episode_len', dest='max_episode_len', type=int,
                    help='max episode length', default=300)
parser.add_argument('--n_agents', dest='n_agents', type=int,
                    help='# agents', default=2)
parser.add_argument('--n_landmarks', dest='n_landmarks', type=int,
                    help='# landmarks', default=2)
parser.add_argument('-V', dest='vocab_size', type=int, required=True,
                    help='vocabulary size') 
parser.add_argument('--grid_size', dest='grid_size', type=int,
                    help='size of a side of the square grid', default=5) 
parser.add_argument('--exp_replay_size', dest='experience_memory', type=int,
                    help='# of trajectories stored into exp replay memory',
                    default=20000)
parser.add_argument('--lr', dest='lr', type=float,default=0.0003,
                    help='learning rate')
parser.add_argument('--gamma', dest='gamma', type=float,default=0.99,
                    help='discounting factor')
parser.add_argument('--algo', dest='algo', type=str, default='rpg_baseline_rec',
                    help='either "random", "rpg" or "rpg_baseline_rec"')

args = parser.parse_args()

def suffix_dir(params):
    r = ''
    for p, v in params.iteritems():
        r += str(p) + '-' + str(v) + '_'
    return r

n_landmarks= args.n_landmarks
n_agents = args.n_agents
V=args.vocab_size
grid_size=args.grid_size
n_episode = args.n_episodes
max_episode_length = args.max_episode_len
gamma = args.gamma
lr = args.lr
bs = args.bs
freq_train = args.freq_train
hid_size = args.n_hidden
experience_memory = args.experience_memory
n_iter_per_train = args.n_iter_per_train
grid_size = (args.grid_size,args.grid_size)

params = copy.deepcopy(vars(args)) # namespace to dict
del params['model_dir']
del params['n_episodes']
model_dir = args.model_dir + '_' + suffix_dir(params)
if os.path.exists(model_dir):
    print "model dir exists... appending a character"
    model_dir = model_dir + "_"

agent = None

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

params['reward_map'] = {0: 0,
                        1: 1}

freq_print = 500

action_space = ['left', 'right', 'up', 'down']
action_map = {0: 'left',
              1: 'right',
              2: 'up',
              3: 'down'}

env = MultiAgentCoop(n_agents, n_landmarks, grid_size, V)
obs_space_size = env.observation_space_size()

if args.algo=='rpg_baseline_rec':
    print "Baseline computed by a LSTM"
    agents = [RPGCommunicate(obs_space_size, len(action_space), V, hid_size,
                             'softmax', gamma, lr, bs, freq_train,
                             experience_memory, n_iter_per_train) for _ in range(n_agents)]
elif args.algo=='random':
    agents = [RandomAgent(len(action_space)) for _ in range(n_agents)]
else:
    raise NotImplementedError

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
