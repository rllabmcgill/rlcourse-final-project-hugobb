#!/usr/local/env python

import time
import json
import gym
import numpy as np
import pickle
import copy
from gym import wrappers
import argparse
import os
import theano

from rl_lstm import RLLSTM
from rpg import RPG

#theano.config.on_unused_input = 'ignore'

def suffix_dir(params):
    r = ''
    for p, v in params.iteritems():
        r += str(p) + '-' + str(v) + '_'
    return r

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('-m', dest='model_dir', type=str, required=True,
                    help='Directory of the gym model')
parser.add_argument('-n', dest='n_episodes', type=int, required=True,
                    help='# episodes')
parser.add_argument('--alpha', dest='alpha', type=float,default=0.0001,
                    help='learning rate')
parser.add_argument('--gamma', dest='gamma', type=float,default=0.95,
                    help='discounting factor')
parser.add_argument('--algo', dest='algo', type=str, default='rpg',
                    help='either "rpg" or "rllstm"')
parser.add_argument('--K', dest='K', type=float,default=0.2)
parser.add_argument('--lambda', dest='lambda_', type=float,default=0.6)
parser.add_argument('--eps', dest='eps', type=float,default=0.1,
                    help='epsilon-greediness')
parser.add_argument('--n-trials', dest='n_trials', type=int, default=1,
                    help='number of trials')

args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
K = args.K
lambda_ = args.lambda_
eps = args.eps
n_episodes = args.n_episodes
params = copy.deepcopy(vars(args))
del params['model_dir']
del params['n_episodes']
model_dir = args.model_dir + '_' + suffix_dir(params)

agent = None

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

env_name = "CartPole-v0" # https://github.com/openai/gym/wiki/CartPole-v0
# cartpole state space: cart pos, cart veloc, pole angle, pole angle veloc
env = gym.make(env_name)

A = env.action_space.n
if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
    S = env.observation_space.n
elif isinstance(env.observation_space, gym.spaces.box.Box):
    S = env.observation_space.shape[0]
else:
    print env.observation_space
    raise Exception()

features_size = S
feature_func = lambda x: np.asarray([x[0], x[2]])
feature_size = 2
#feature_func = lambda x: x
#feature_size = 4

if args.algo == 'rllstm':
    create_agent = lambda: RLLSTM(feature_size, 14, 2, 'linear', gamma, lambda_, K, alpha, eps)
elif args.algo == 'rpg':
    create_agent = lambda: RPG(feature_size, 2, 14, 'softmax', gamma, lambda_, alpha)
else:
    raise NotImplementedError

#np.seterr(all='raise')

rewards_count = np.zeros((args.n_trials, n_episodes))
episodes_len = np.zeros((args.n_trials, n_episodes))
TD_errors = np.zeros((args.n_trials, n_episodes))
#prediction_TD_errors = np.zeros((args.n_trials, n_episodes))
#var_pred_TD_errors = np.zeros((args.n_trials, n_episodes))
V_st = np.zeros((args.n_trials, n_episodes))
V_st1 = np.zeros((args.n_trials, n_episodes))
A_mean = np.zeros((args.n_trials, n_episodes))
next_A_mean = np.zeros((args.n_trials, n_episodes))

# the Gym env rewards 1 if episode is not over (not done)
# we create an artificial reward of 0
# we map both rewards to other rewards that can make learning easier
#reward_map = dict({1:0, 0:-1})
reward_map = dict({1:1, 0:0})

freq = 500
for trial in range(args.n_trials):
    agent = create_agent()
    for i_episode in range(n_episodes):
        # we keep the observations and errors for the error predictor
        observations = []
        E_td = []

        #agent.eps = 3. / (i_episode+1000)#(np.sqrt(i_episode+1000))
        #decaying_lr = 1. / (30*np.sqrt(i_episode+1000))
        #agent.lr.set_value(decaying_lr)#3*np.log(i_episode+10))
        if i_episode % freq == 0:
            print "# episode, eps, lr:", i_episode #, agent.eps#, decaying_lr

            print "mean over last", freq, ":", episodes_len[trial,i_episode-freq:i_episode].mean()
        o_t = env.reset()
        #observations.append(feature_func(o_t))
        agent.reset(feature_func(o_t))
        r_t = 0
        done = False
        t = 0
        r_buf = 0
        limit = 200
        while not done and t < limit:
            a_t = agent.sample_action(feature_func(o_t), r_t)
            o_t_1, r_t_1, done, info = env.step(a_t)
            #observations.append(feature_func(o_t_1))
            r_t_1 = reward_map[r_t_1] # to make rewards only at the end of episode
            r_buf += r_t_1
            logs = agent.update(feature_func(o_t), a_t, feature_func(o_t_1),
                                r_t, False)
            #if logs != None and instanceof(agent, RLLSTM):
            #    E, _V_t, _V_t_1, last_e = logs
            #    E_td.append(E)
            #print E_td[-1], _V_t, _V_t_1, e
            #V_st[trial, i_episode] += _V_t
            #V_st1[trial, i_episode] += _V_t_1
            t += 1
            o_t = o_t_1
            r_t = r_t_1
        # TODO: create a wrapper for environment so that there is a last fictitious
        # state when env is done 
        if t == limit - 1:
            r_t = reward_map[1]
        else:
            r_t = reward_map[0]
        r_buf += r_t
        a_t = agent.sample_action(feature_func(o_t), r_t)
        # only action matter, as observation and reward are not used
        # so we don't mind passing feature_func(o_t) again
        logs = agent.update(feature_func(o_t), a_t, feature_func(o_t), r_t, True)
        #E, _V_st, _V_st1, e = logs
        #E_td.append(E)
        #if np.abs(E) > 10**4:
        #    print "Error: TD error too big at episode", i_episode
        #    break

        #mean, var = ep.train(observations, E_td, n_iter = 2)
        #prediction_TD_errors[trial, i_episode] = mean
        #var_pred_TD_errors[trial, i_episode] = var
        TD_errors[trial, i_episode] = np.mean(E_td)
        V_st[trial, i_episode] /= t
        V_st1[trial, i_episode] /= t
        rewards_count[trial, i_episode] = r_buf
        episodes_len[trial, i_episode] = t
        if i_episode%freq==0:
            print "end of ep:", 
            print "TD:", TD_errors[trial,i_episode-freq:i_episode].mean()
            #print "pred TD:", prediction_TD_errors[trial,i_episode-freq:i_episode].mean()
            #print "variance pred TD:", var_pred_TD_errors[trial,i_episode-freq:i_episode].mean()
            print "V_st, V_st+1:", V_st[trial,i_episode-freq:i_episode].mean(), V_st1[trial,i_episode-freq:i_episode].mean()


pickle.dump(TD_errors, open(model_dir + "/td_errors.p", "wb"))
pickle.dump(rewards_count, open(model_dir + "/rewards_count.p", "wb"))
pickle.dump(episodes_len, open(model_dir + "/episode_lengths.p", "wb"))
#pickle.dump(agent, open(model_dir + "/model.p", "wb"))
