import theano
import theano.tensor as T
from theano import shared
import numpy as np
from theano.tensor.nnet import sigmoid
from theano import scan, shared, function
from lasagne.updates import adam
import pickle
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.shared_randomstreams import RandomStreams
from theano.gradient import disconnected_grad
from utils import get_activation_function, init_weights_bias
from collections import OrderedDict 
from theano.tensor.shared_randomstreams import RandomStreams
#from smorms3 import SMORMS3
from theano.tensor.extra_ops import to_one_hot

floatX = theano.config.floatX

class LSTM():
    def __init__(self, n_in, n_h, n_out, output_activation, lr, baseline, optimizer='adam'):
        """
        n_in, n_h, n_out: dim of in, hidden, output
        output_activation: 'softmax', 'sigmoid', ...
        lr: learning rate
        baseline: either 'const', 'matrix'
        """
        self.h_0 = shared(np.random.uniform(size=(n_h)), name='h_0')
        self.c = shared(np.zeros(n_h), name='c')
        self.h = shared(np.zeros(n_h), name='h')

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

        iW, ib = init_weights_bias((n_h, n_out), output_activation)
        W_out = shared(iW, name='W_out') 
        b_out = shared(ib, name='b_out', broadcastable=(True, False))

        self.params = [W_ifo, R_ifo, b_ifo, W_z, R_z,
                       b_z, W_out, b_out, self.h_0]

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
            y = f(T.dot(next_h, W_out) + b_out)#.flatten()
            # TODO: remove sampling from here...
            # Or at least fix a which doesn't take into account batchsize...
            a = rng.multinomial(pvals=y, dtype=floatX)
            #a = to_one_hot(a, y.shape[1])
            return a, y, next_c, next_h

        # we want to unroll step by step when acting
        X_t = T.vector('X_t', dtype=floatX) # observation and reward, one timestep
        one_hot_a_t, _, next_c_t, next_h_t = step(X_t, self.c, self.h) 
        a_t = T.argmax(one_hot_a_t)

        updates_cell = OrderedDict()
        updates_cell[self.c] = next_c_t.flatten()
        updates_cell[self.h] = next_h_t.flatten()
        self.step_f = function([X_t], a_t, updates=updates_cell)

        # we also want to be able to train on one whole trajectories
        X = T.tensor3('X', dtype=floatX) # observation and reward concatenated
        # shape is: (batch size, max sequence length, n_in)
        #mask = T.matrix('mask', dtype='int64') # shape=(batch_size, max_length)
        batch_size = X.shape[0] 
        L = X.shape[1]
        outputs_eval = [None,
                        None,
                        T.zeros((batch_size, n_h)),
                        T.alloc(self.h_0, batch_size, n_h)]
        returned, updates = scan(fn = step,
                            sequences = X.dimshuffle(1,0,2), # (L, bs, n_in)
                            outputs_info=outputs_eval)

        _, A, _, _ = returned
        one_hot_a = T.tensor3('one_hot_a', dtype=floatX)
        a = T.argmax(one_hot_a, axis=2) # (bs, l)
        #a = T.matrix('a', dtype='int64') #bs, l

        #loss = (mask.reshape((L*batch_size,)) * CE).sum() / mask.sum()
        # TODO: mask
        
        R = T.dmatrix('R') # matrix of empirical returns, shape=(bs, l)
        mask = T.dmatrix('mask') # mask, shape=(bs, l)
        #b = T.dvector('b') # baseline
        if baseline == 'const':
            b = T.dscalar('b')
        elif baseline =='matrix':
            b = T.dmatrix('b') # shape (bs,l)

        gradients = []
        # A shape should be (L,bs,n_out), C (L,bs)
        C = A[T.arange(A.shape[0]).reshape((-1, 1)), T.arange(A.shape[1]), a.T] 
        for p in self.params:
            # if b is scalar: transpose doesn't matter
            log_grad = -T.grad(T.sum(T.log(C)*(R.T - b.T)*mask.T), p) 
            print p.broadcastable, p.name
            gradients.append(log_grad)

        if optimizer == 'adam':
            updates_opt = adam(gradients, self.params, lr)
        else:
            raise NotImplementedError
        #for k in updates_opt.keys():
        #    print "update:", k.name, k
        #print len(updates+updates_opt)
        self.train_f = function([X, R, one_hot_a, mask, b], gradients, updates=updates+updates_opt)

    def step(self, o_t, r_t):
        return self.step_f(np.concatenate([o_t, np.asarray([r_t])]))

    def reset(self):
        self.h.set_value(self.h_0.get_value())
