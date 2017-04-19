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
from lasagne.nonlinearities import softmax
from experience_replay import ExperienceReplay
from error_predictor import ErrorPredictor

from rpg_ep import RPGRecurrentBaseline
from lstm_multi_out import LSTMMultiOut

floatX = theano.config.floatX

class RPGCommunicate(RPGRecurrentBaseline):
    """
    This agent emits an action but also emits speech
    """
    def __init__(self, obs_space_size, action_space_size, vocab_size, n_h,
                 output_activation, gamma, lr, batch_size=128,
                 freq_train = 50, maxlen = 500, n_iter_per_train=10):
        self.gamma = gamma
        self.batch_size = batch_size

        # input of the RNN is the observations concat w/ reward
        self.n_iter_per_train = n_iter_per_train
        self.obs_space_size = obs_space_size
        n_in = obs_space_size + 1
        self.n_a = action_space_size 
        self.n_c = vocab_size

        self.lstm = LSTMMultiOut(n_in + self.n_a + self.n_c, n_h,
                                 [self.n_a, self.n_c], 'softmax',
                                 lr, baseline='matrix')
        self.baseline = ErrorPredictor(obs_space_size + self.n_a + self.n_c, n_h, 0.001)
        # experience will consist of observations, probas over actions, actions
        # and rewards
        self.experience = ExperienceReplay(maxlen)
        self.current_h = []
        self.count_episode = 0
        self.freq_train = freq_train

    def _process_history(self, H):
        """
        from a list of trajectory [h_1, h_2, ...]
        where h_i=[o_1, r_1, a_1, c_1, o_2, r_2, ..., a_N, c_N]

        outputs (a, X, Y, mask):
        a: ndarray of shape (bs, L) of type int64
        actions taken
        c: ndarray of shape (bs, L) of type int64
        speech emitted
        X: ndarray of shape (bs, L, len(O) + 1)
        observations + rewards concatenated
        R: X.shape
        sum of discounted rewards
        mask: ndarray of size (bs, L) of type int64

        Where L=max(N)
        """
        lens = [len(h) for h in H]
        L = max(lens)
        X, R, a, c, mask = [], [], [], [], []

        # let's fill X, Y and mask trajectory after trajectory
        for h in H:
            l = len(h)
            #if l==L:
            #    continue
            mask.append(np.concatenate([np.ones(l-1), np.zeros(L-l+1)]))

            r_i = np.zeros(L)
            r_i[:l] = np.asarray([s[1] for s in h])
            R_i = np.zeros(L)
            R_i[l-2] = r_i[l-1]
            for j in reversed(range(l-2)):
                R_i[j] = r_i[j+1] + self.gamma * R_i[j+1]
            R.append(R_i)

            # we process actions and speech similarly
            a_i = np.zeros((L, self.n_a))
            a_i[np.arange(l), [int(s[2]) for s in h]] = 1
            a.append(self._pad(a_i, L))

            c_i = np.zeros((L, self.n_c))
            c_i[np.arange(l), [int(s[3]) for s in h]] = 1
            c.append(self._pad(c_i, L))

            X_i = [h[0][0], np.asarray(h[0][1]), np.zeros(self.n_a + self.n_c)]
            X_i = np.hstack(X_i)
            if l>1:
                X_i_rest = [np.hstack([s[0], np.asarray([s[1]]),
                            a_i[i-1], c_i[i-1]]) for i, s in enumerate(h[1:])]
                X_i_rest = np.vstack(X_i_rest)
                X_i = np.vstack([X_i, X_i_rest])
            else:
                X_i = X_i.reshape((1,-1))

            X.append(self._pad(X_i, L))
        mask = np.asarray(mask).astype(floatX)
        X = np.asarray(X).astype(floatX)
        R = np.asarray(R).astype(floatX)
        a = np.asarray(a).astype(floatX)
        c = np.asarray(c).astype(floatX)
        # before: a has shape: (L, n_a, bs)
        np.swapaxes(a, 1, 2) # (L, bs, n_a)
        np.swapaxes(a, 0, 1) # (bs, L, n_a)
        # do the same thing for c
        np.swapaxes(c, 1, 2) # (L, bs, n_c)
        np.swapaxes(c, 0, 1) # (bs, L, n_c)
        return a, c, X, R, mask
       
    def _train(self):
        # Maybe for a start only do SGD to avoid using masks

        for i in range(self.n_iter_per_train):
            batch = self.experience.get_batch(self.batch_size, random_order=True)
            a, c, X, R, mask = self._process_history(batch)
            X_b = self._get_baseline_features(X)
            bs, L, _ = X.shape
            b = self.baseline.predict(X_b, mask).reshape((bs, L))
            #print "a", a.shape
            #print "X", X.shape
            #print "R", R.shape
            #print "mask", mask.shape
            #print "X_b", X_b.shape
            #print "a", a.shape
            grads = self.lstm.train_f(X, R, mask, b, a, c)
            #for g in grads:
            #    print "min, mean, max", np.min(g), "," ,np.mean(g), ",", np.max(g)
        
    def get_debug_info(self):
        batch = self.experience.get_batch(self.batch_size, random_order=True)
        a, c, X, R, mask = self._process_history(batch)
        X_b = self._get_baseline_features(X)
        bs, L, _ = X.shape
        b = self.baseline.predict(X_b, mask).reshape((bs, L))
        X = np.dstack([X, a])
        return [a, c, X, R, mask, X_b, b]

    def _get_baseline_features(self, X):
        """
        X: observations (bs, L, n_in + 1 + n_out)
        """
        # remove reward from X
        return np.dstack([X[:,:,:self.obs_space_size], X[:,:,self.obs_space_size+1:]])

    def _train_baseline(self, n_iter=50):
        # Maybe for a start only do SGD to avoid using masks
        err = []
        for i in range(n_iter):
            batch = self.experience.get_batch(self.batch_size, random_order=True)
            _, _, X, R, mask = self._process_history(batch)
            X_b = self._get_baseline_features(X)
            err.append(self.baseline.train(X_b, mask, R))
        #if np.mean(err) < 10.0:
        #    break
        print "training baseline...", np.mean(err)
        return np.mean(err)
            
    def sample_action(self, o_t, r_t, a_t_1, c_t_1):
        oh_a = np.zeros(self.n_a, dtype=floatX)
        oh_c = np.zeros(self.n_c, dtype=floatX)
        oh_a[a_t_1] = 1
        oh_c[c_t_1] = 1
        x = np.concatenate([o_t, np.asarray([r_t]), oh_a, oh_c]).astype(floatX)
        #print "sample action: oh, ot, x", oh_a.shape, o_t.shape, x.shape
        a, c = self.lstm.step(x)
        return a, c
    
    def update(self, o_t, a_t, c_t, o_t_1, r_t, done):
        h = (o_t, r_t, a_t, c_t)
        self.current_h.append(h)
        return None

if __name__=="__main__":
    agent = RPGRecurrentBaseline(2, 2, 14, softmax, 0.99, 0.6, 0.001)
