#!/usr/bin/env python

from __future__ import division

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
                 freq_train = 50, maxlen = 500, n_iter_per_train=10):
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_iter_per_train = n_iter_per_train
        self.exp_memory_size = maxlen

        # input of the RNN is the observations concat w/ reward
        n_in = obs_space_size + 1
        self.n_out = action_space_size

        self.lstm = LSTM(n_in, n_h, self.n_out, 'softmax', lr, baseline='const',
                         optimizer='adam')

        # experience will consist of observations, probas over actions, actions
        # and rewards
        self.experience = ExperienceReplay(maxlen)
        self.current_h = []
        self.count_episode = 0
        self.freq_train = freq_train

    def reset(self, o):
        self.count_episode += 1
        if len(self.current_h) > 0:
            #print "last episode added:", [e[1] for e in self.current_h], len(self.current_h)
            self.experience.append(self.current_h)

        self.current_h = []
        if self.count_episode % self.freq_train == 0 and len(self.experience) > 0:
            self._train()
        self.lstm.reset()
        self.current_done = False

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

        r, X, R, a, mask = [], [], [], [], []
        # r stores the reward, it is used to compute the baseline.

        # let's fill X, Y and mask trajectory after trajectory
        for h in H:
            l = len(h)
            if l == L:
                continue
            # compute returns. Have to take into account the last step 
            # where done is true.
            r_i = np.zeros((L), dtype=floatX)
            r_i[:l] = np.asarray([s[1] for s in h])
            R_i = np.zeros((L), dtype=floatX)
            R_i[l-1] = 0#r_i[l-1]
            for j in reversed(range(l-1)):
                R_i[j] = r_i[j+1] + self.gamma * R_i[j+1]
            R.append(R_i)
            r.append(r_i)

            # We actually don't want to consider the last action (when done)
            mask.append(np.concatenate([np.ones(l-1), np.zeros(L-l+1)]))
            a_i = np.zeros((L, self.n_out))
            a_i[np.arange(l), [int(s[2]) for s in h]] = 1
            a.append(self._pad(a_i, L))

            X_i = np.hstack([h[0][0], np.asarray(h[0][1]), np.zeros(self.n_out)])
            if l>1:
                X_i_rest = np.vstack([np.hstack([s[0], np.asarray([s[1]]), a_i[i-1]]) for i, s in enumerate(h[1:])])

                #print "Xirest", X_i_rest.shape
                X_i = np.vstack([X_i, X_i_rest])
            else:
                X_i = X_i.reshape((1,-1))

            X.append(self._pad(X_i, L))
            #Y_i = np.asarray([s[2] for s in h])
            #Y.append(self._pad(Y_i, L))
            # one hot

        mask = np.asarray(mask).astype(floatX)
        X = np.asarray(X).astype(floatX)
        R = np.asarray(R).astype(floatX)
        r = np.asarray(r).astype(floatX)

        a = np.asarray(a).astype(floatX)
        # before: a has shape: (L, n_out, bs)
        np.swapaxes(a, 1, 2) # (L, bs, n_out)
        np.swapaxes(a, 0, 1) # (L, bs, n_out)

        return a, X, R, mask

    def _compute_const_baseline(self):
        """compute baseline with the whole history"""
        return 0
        mask, R = [], []
        H = self.experience.get_batch(min(1024, self.exp_memory_size),
                                      random_order=True)
        L = max([len(h) for h in H])
        for h in H:
            l = len(h)
            if l == L:
                continue
            mask.append(np.concatenate([np.ones(l-1), np.zeros(L-l+1)]))
            r_i = np.zeros((L), dtype=floatX)
            r_i[:l] = np.asarray([s[1] for s in h])
            R_i = np.zeros((L), dtype=floatX)
            R_i[l-1] = 0#r_i[l-1]
            for j in reversed(range(l-1)):
                R_i[j] = r_i[j+1] + self.gamma * R_i[j+1]
            R.append(R_i)
        mask = np.asarray(mask).astype(floatX)
        R = np.asarray(R).astype(floatX)
        b = np.sum(R) / np.sum(mask)
        # alternative: mean return computed on episodes
        #b_2 = (np.sum(R, axis=1) / np.sum(mask, axis=1)).mean()
        #b3 = -1
        #print "2 diff baselines", b, b_2
        return b

    def _train(self):
        # Maybe for a start only do SGD to avoid using masks
        b = self._compute_const_baseline()
        for i in range(self.n_iter_per_train):
            batch = self.experience.get_batch(self.batch_size, random_order=True)
            a, X, R, mask = self._process_history(batch)
            #print "some shapes:" 
            #print "X:", X
            #print "R:", R
            #print "mask:", mask
            #print "a:", a
            grads = self.lstm.train_f(X,R,a, mask, b)
            #for g in grads:
            #    print "min, mean, max", np.min(g), "," ,np.mean(g), ",", np.max(g)
            
    def sample_action(self, o_t, r_t, a_t_1):
        # pass obs through RNN
        oh_a = np.zeros(self.n_out)
        oh_a[a_t_1] = 1
        x = np.concatenate([o_t, np.asarray([r_t]), oh_a])
        return self.lstm.step(x)
    
    def get_debug_info(self):
        batch = self.experience.get_batch(self.batch_size, random_order=True)
        a, X, R, mask = self._process_history(batch)
        b = self._compute_const_baseline()
        return [a, X, R, mask, b]

    def update(self, o_t, a_t, o_t_1, r_t, done):
        # store history in experience
        # TODO: maybe experience should directly be stored as an ndarray?
        # we don't want the preprocessing to occur at train time as we 
        # access more than we store. 
        assert(not self.current_done)
        if done:
            self.current_done = True
        h = (o_t, r_t, a_t)
        self.current_h.append(h)
        return None

if __name__=="__main__":
    agent = RPG(2, 2, 14, softmax, 0.99, 0.6, 0.001)
