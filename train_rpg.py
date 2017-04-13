from lib.env import MultiAgent
import numpy as np
from tqdm import tqdm
from lib.rpg_agent.rpg import RPG
from lib.rpg_agent.random_agent import RandomAgent
from lib.utils import close_window
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
                    help='Batch size', default=64)
parser.add_argument('--n_hidden', dest='n_hidden', type=int, 
                    help='n hidden units', default=8)
parser.add_argument('--n_iter_per_train', dest='n_iter_per_train', type=int, 
                    help='# iterations per training', default=1)
parser.add_argument('--max_episode_len', dest='max_episode_len', type=int,
                    help='max episode length', default=50)
parser.add_argument('--n_agents', dest='n_agents', type=int,
                    help='# agents', default=2)
parser.add_argument('--n_landmarks', dest='n_landmarks', type=int,
                    help='# landmarks', default=2)
parser.add_argument('--exp_replay_size', dest='experience_memory', type=int,
                    help='# of trajectories stored into exp replay memory',
                    default=2000)

parser.add_argument('--lr', dest='lr', type=float,default=0.001,
                    help='learning rate')
parser.add_argument('--gamma', dest='gamma', type=float,default=0.999,
                    help='discounting factor')
parser.add_argument('--algo', dest='algo', type=str, default='rpg',
                    help='either "rpg" or "random"')

args = parser.parse_args()

def suffix_dir(params):
    r = ''
    for p, v in params.iteritems():
        r += str(p) + '-' + str(v) + '_'
    return r

n_landmarks= args.n_landmarks
n_agents = args.n_agents
n_episode = args.n_episodes
max_episode_length = args.max_episode_len
gamma = args.gamma
lr = args.lr
bs = args.bs
freq_train = args.freq_train
hid_size = args.n_hidden
experience_memory = args.experience_memory
n_iter_per_train = args.n_iter_per_train

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

grid_size = (5,5)
params['reward_map'] = {0: -0.2,
                        1: 1}
params['obs_map'] = {0: -0.5,
           1: -0.25,
           2: 0.,
           3: 0.25,
           4: 0.5}

freq_print = 250

action_space = ['left', 'right', 'up', 'down']
action_map = {0: 'left',
              1: 'right',
              2: 'up',
              3: 'down'}

obs_space_size = 2*n_landmarks + 1 + 1
if args.algo=='rpg':
    agents = [RPG(obs_space_size, len(action_space), hid_size, 'softmax', gamma, lr, bs,
              freq_train, experience_memory, n_iter_per_train) for _ in range(n_agents)]
elif args.algo=='random':
    agents = [RandomAgent(len(action_space)) for _ in range(n_agents)]
else:
    raise NotImplementedError

env = MultiAgent(action_space, n_agents, grid_size=grid_size, n_landmarks=n_landmarks)
# Some preprocessing functions for RPG
def preproc_observations(O, t):
    #if t>1:
    #    return [np.zeros(obs_space_size) for _ in O]
    preproc_O = []
    for obs in O:
        preproc_obs = [params['obs_map'][o] for o in obs[:-1]]
        preproc_obs.append(obs[-1])
        preproc_obs.append(1) # flag used to indicate that we are actually giving the 
        # positions and not zeroing out everything
        preproc_obs = np.asarray(preproc_obs)
        # TODO: try 
        for i in range(len(O)):
            preproc_obs[2+i*2:2+(i+1)*2] -= preproc_obs[:2]
        preproc_O.append(preproc_obs[2:])
    return preproc_O

postproc_action = lambda a: action_map[int(a)]
preproc_rewards = lambda R: [params['reward_map'][r] for r in R]
#reward_counts = np.zeros((n_episode, n_agents))
episode_len = np.ones((n_episode, n_agents)) * -1


for i_episode in tqdm(range(n_episode)):
    observations = env.reset()
    observations = preproc_observations(observations, 0)
    for i in range(n_agents):
        agents[i].reset(observations[i])

    r_t = [0 for _ in range(n_agents)]
    r_t = preproc_rewards(r_t)
    done = [False for _ in range(n_agents)]
    for t in range(1, max_episode_length):
        actions, actions_env = [], []
        # actions_env stores action in the env format
        for i, a in enumerate(agents): 
            # we don't care if done is true for one agent
            action = a.sample_action(observations[i], r_t[i])
            actions.append(action)
            actions_env.append(postproc_action(action))
        new_observations, r_t, done, info = env.step(actions_env)
        r_t = preproc_rewards(r_t)
        #print "rew:", r_t_1
        new_observations = preproc_observations(new_observations, t)
        for i, a in enumerate(agents):
            # TODO: if t==0: continue?
            if done[i] and episode_len[i_episode, i] > 0 and episode_len[i_episode, i] <= t - 1:
                continue
            elif done[i] and episode_len[i_episode, i] < 0:
                episode_len[i_episode, i] = t
                #print "done at time", t, "agent", i
            a.update(observations[i], actions[i], new_observations[i], r_t[i], done[i])
            #print "ep:", i_episode, " t:", t, " agent", i, " rew:", r_t[i], " done:", done[i]
        if all([done[i] and 0 <= episode_len[i_episode, i] <= t-1 for i in range(n_agents)]):
            #print "all done at time", t
            break
        observations = new_observations
            
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


save = lambda obj, fname: pickle.dump(obj, open(model_dir + "/" + fname + ".p", "w"))

def save_json(data, json_filename):
    with open(model_dir + "/" + json_filename + ".json", "w") as f:
        f.write(json.dumps(data, indent=4))
        
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
