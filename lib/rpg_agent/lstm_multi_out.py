from __future__ import division

import theano
import theano.tensor as T
from theano import shared
import numpy as np
from theano.tensor.nnet import sigmoid
from theano import scan, shared, function
from lasagne.updates import adam, sgd, total_norm_constraint
import pickle
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gradient import disconnected_grad
from utils import get_activation_function, init_weights_bias
from collections import OrderedDict 
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.extra_ops import to_one_hot

floatX = theano.config.floatX

class LSTMMultiOut():
    """
    this LSTM can output two independent categorical distributions 
    conditionned on the hidden state
    """
    def __init__(self, n_in, n_h, list_n_out, output_activation, lr, baseline, optimizer='adam'):
        """
        n_in, n_h: dim of in, hidden
        list_n_out: list of output size
        output_activation: 'softmax', 'sigmoid', ...
        lr: learning rate
        baseline: either 'const', 'matrix'
        """
        len_out = len(list_n_out)
        self.n_h = n_h
        h_0_val = np.random.uniform(size=(n_h)).astype(floatX)
        self.h_0 = shared(h_0_val, name='h_0')
        self.c = shared(np.zeros(n_h, dtype=floatX), name='c')
        self.h = shared(np.zeros(n_h, dtype=floatX), name='h')

        # i, f, o gate go together as they are sigmoided
        iW, ib = init_weights_bias((n_in, 3*n_h), 'sigmoid')
        iR, _ = init_weights_bias((n_h, 3*n_h), 'sigmoid')
        W_ifo = shared(iW, name='W_ifo') 
        R_ifo = shared(iR, name='R_ifo') 
        b_ifo = shared(ib, name='b_ifo', broadcastable=(True, False))

        iW, ib = init_weights_bias((n_in, n_h), 'tanh')
        iR, _ = init_weights_bias((n_h, n_h), 'tanh')
        W_z = shared(iW, name='W_z') 
        R_z = shared(iR, name='R_z') 
        b_z = shared(ib, name='b_z', broadcastable=(True, False))

        self.params = [W_ifo, R_ifo, b_ifo, W_z, R_z, b_z, self.h_0]
        self.params_out = []

        W_out, b_out = [], []
        for i, n_out in enumerate(list_n_out):
            print "create output of size", n_out
            iW, ib = init_weights_bias((n_h, n_out), output_activation)
            W_out.append(shared(iW, name='W_out_' + str(i)))
            b_out.append(shared(ib, name='b_out_' + str(i),
                                broadcastable=(True, False)))
            self.params_out.append([W_out[-1], b_out[-1]])
        assert(len(W_out) == len(b_out) == len_out)

        rng = RandomStreams(np.random.RandomState(0).randint(2**30)) 

        def step(x, c, h):
            """
            compute output and updates of hidden and cell states of LSTM
            returns tuple(output, updates)
            """
            ifo = sigmoid(T.dot(x, W_ifo) + T.dot(h, R_ifo) + b_ifo)
            i = ifo[:,:n_h]
            f = ifo[:,n_h:-n_h]
            o = ifo[:,-n_h:]

            z = T.tanh(T.dot(x, W_z) + T.dot(h, R_z) + b_z)
            next_c = i * z + f * c
            next_h = o * T.tanh(next_c)

            f = get_activation_function(output_activation)
            y, a = [], []
            # y contains the distributions parameters
            # a contains sample from these distributions
            for W_, b_ in zip(W_out, b_out):
                y.append(f(T.dot(next_h, W_) + b_))
                a.append(rng.multinomial(pvals=y[-1], dtype=floatX))
            return a + y + [next_c, next_h]

        # we want to unroll step by step when acting
        X_t = T.vector('X_t', dtype=floatX) # observation and reward, one timestep
        step_res = step(X_t, self.c, self.h) 
        next_c_t, next_h_t = step_res[-2:]
        one_hot_t = step_res[:len_out]
        discrete_out_t = [T.argmax(oh) for oh in one_hot_t]

        updates_cell = OrderedDict()
        updates_cell[self.c] = next_c_t.flatten()
        updates_cell[self.h] = next_h_t.flatten()
        self.step_f = function([X_t], discrete_out_t, updates=updates_cell)

        # we also want to be able to train on one whole trajectories
        X = T.tensor3('X', dtype=floatX) # observation and reward concatenated
        # shape is: (batch size, max sequence length, n_in)
        #mask = T.matrix('mask', dtype='int64') # shape=(batch_size, max_length)
        batch_size = X.shape[0] 
        L = X.shape[1]
        outputs_eval = ([None,]*len_out +
                        [None,]*len_out +
                        [T.zeros((batch_size, n_h), dtype=floatX)] +
                        [T.alloc(self.h_0, batch_size, n_h)])
        returned, updates = scan(fn = step,
                            sequences = X.dimshuffle(1,0,2), # (L, bs, n_in)
                            outputs_info=outputs_eval)

        distributions = returned[len_out:len_out*2]

        one_hot_out = []
        discrete_out = []
        for i,_ in enumerate(list_n_out):
            oh = T.tensor3('one_hot_' + str(i), dtype=floatX)
            one_hot_out.append(oh)
            discrete_out.append(T.argmax(oh, axis=2)) # (bs, l)
        
        R = T.matrix('R', dtype=floatX) # matrix of empirical returns, shape=(bs, l)
        mask = T.matrix('mask', dtype=floatX) # mask, shape=(bs, l)
        #b = T.dvector('b', dtype=floatX) # baseline
        if baseline == 'const':
            b = T.scalar('b', dtype=floatX)
        elif baseline =='matrix':
            b = T.matrix('b', dtype=floatX) # shape (bs,l)

        # here we compute REINFORCE estimates
        gradients = OrderedDict() 
        self.params_all = self.params + sum(self.params_out, [])
        # init all gradients to 0
        for p in self.params_all:
            gradients[p.name] = 0

        # A shape should be (L,bs,n_out), C (L,bs)
        for i, tuple_distrib_discrete in enumerate(zip(distributions, discrete_out)):
            A, a = tuple_distrib_discrete
            C = A[T.arange(A.shape[0]).reshape((-1, 1)), T.arange(A.shape[1]), a.T] 
            advantage = (R.T - b.T)*mask.T
            for p in self.params + self.params_out[i]:
                print "output #" + str(i) + ":", p.broadcastable, p.name
                gradients[p.name] += -(T.grad(T.sum(T.log(C)*advantage), p) /
                                       T.sum(mask))

        gradients = gradients.values()
        #gradients = total_norm_constraint(gradients, 5)
        if optimizer == 'adam':
            updates_opt = adam(gradients, self.params_all, lr)
        elif optimizer == 'sgd':
            updates_opt = sgd(gradients, self.params_all, lr)
        else:
            raise NotImplementedError
        #for k in updates_opt.keys():
        #    print "update:", k.name, k
        #print len(updates+updates_opt)
        self.train_f = function([X, R, mask, b] + one_hot_out, gradients, updates=updates+updates_opt)
        self.reset()

    def step(self, x):
        return self.step_f(x)

    def reset(self):
        self.h.set_value(self.h_0.get_value())
        self.c.set_value(np.zeros(self.n_h, dtype=floatX))
