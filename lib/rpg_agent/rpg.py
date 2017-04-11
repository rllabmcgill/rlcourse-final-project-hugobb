#!/usr/bin/env python

import theano
import theano.tensor as T
from theano import shared
from theano.tensor.nnet import sigmoid
from theano import scan, shared, function
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import lasagne
import pickle
from utils import get_activation_function, init_weights_bias
from collections import OrderedDict
from lstm import LSTM
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from experience_replay import ExperienceReplay

floatX = theano.config.floatX

class RPG():
    def __init__(self, obs_space_size, action_space_size, n_h,
                 output_activation, gamma, lr, batch_size=128,
                 freq_train = 50, maxlen = 500):
        self.gamma = gamma
        self.batch_size = batch_size

        # input of the RNN is the observations concat w/ reward
        n_in = obs_space_size + 1
        self.n_out = action_space_size

        self.lstm = LSTM(n_in, n_h, self.n_out, 'softmax', lr)

        # experience will consist of observations, probas over actions, actions
        # and rewards
        self.experience = ExperienceReplay(maxlen)
        self.current_h = []
        self.count_episode = 0
        self.freq_train = freq_train

    def reset(self, o):
        self.count_episode += 1
        if len(self.current_h) > 0:
            self.experience.append(self.current_h)

        self.current_h = []
        if self.count_episode % self.freq_train == 0 and len(self.experience) > 0:
            self._train()
        self.lstm.reset()

    def _pad(self, d, L):
        """
        vertically stack zeros (first axis) so that it goes up to L
        """
        l = d.shape[0]
        res = d
        if L - l > 0:
            res = np.vstack([res, np.zeros((L-l, d.shape[1]))])
        return res
            
    def _process_history(self, H):
        """
        from a list of trajectory [h_1, h_2, ...]
        where h_i=[o_1, r_1, A_1, a_1, o_2, r_2, ..., A_N, a_N]

        outputs (a, X, Y, mask):
        a: ndarray of shape (bs, L) of type int64
        actions taken
        X: ndarray of shape (bs, L, len(O) + 1)
        observations + rewards concatenated
        R: X.shape
        sum of discounted rewards
        mask: ndarray of size (bs, L) of type int64

        Where L=max(N)
        """
        lens = [len(h) for h in H]
        L = max(lens)
        X, R, a, mask = [], [], [], []

        # let's fill X, Y and mask trajectory after trajectory
        for h in H:
            l = len(h)
            mask.append(np.concatenate([np.ones(l), np.zeros(L-l)]))
            X_i = np.asarray([np.concatenate([s[0], np.asarray([s[1]])]) for s in h])
            X.append(self._pad(X_i, L))
            rewards = np.asarray([s[1] for s in h])
            R_i = np.zeros((L), dtype=floatX)
            R_i[l-1] = rewards[l-1]
            for j in reversed(range(l-1)):
                R_i[j] =  R_i[j+1] * self.gamma + rewards[j]
            R.append(R_i)
            #Y_i = np.asarray([s[2] for s in h])
            #Y.append(self._pad(Y_i, L))
            # one hot
            a_i = np.zeros((L, self.n_out))
            a_i[np.arange(l), [int(s[2]) for s in h]] = 1

            # from: http://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            a.append(self._pad(a_i, L))
        mask = np.asarray(mask).astype('int64')

        X = np.asarray(X).astype(floatX)
        R = np.asarray(R).astype(floatX)

        a = np.asarray(a).astype('int64')
        # before: a has shape: (L, n_out, bs)
        np.swapaxes(a, 1, 2) # (L, bs, n_out)
        np.swapaxes(a, 0, 1) # (L, bs, n_out)
        b = np.mean(R)
        return a, X, R, mask, b
        
    def _train(self, n_iter=1):
        # Maybe for a start only do SGD to avoid using masks
        for i in range(n_iter):
            batch = self.experience.get_batch(self.batch_size, random_order=True)
            a, X, R, mask, b = self._process_history(batch)
            #print "some shapes:" 
            #print "X:", X.shape
            #print "R:", R.shape
            #print "mask:", mask.shape
            #print "a:", a.shape
            self.lstm.train_f(X,R,a, mask, b)
            
    def sample_action(self, o, r):
        # pass obs through RNN
        a = self.lstm.step(o, r)
        return a
    
    def update(self, o_t, a_t, o_t_1, r_t, done):
        # store history in experience
        # TODO: maybe experience should directly be stored as an ndarray?
        # we don't want the preprocessing to occur at train time as we 
        # access more than we store. 
        h = (o_t, r_t, a_t)
        self.current_h.append(h)
        return None

if __name__=="__main__":
    agent = RPG(2, 2, 14, softmax, 0.99, 0.6, 0.001)
