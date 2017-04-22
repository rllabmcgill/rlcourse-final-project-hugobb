import lasagne
import theano.tensor as T
import theano
from ..updates import DeepMindRmsprop, Adam
import numpy as np
from time import time
from ..memory import RecurrentMemory

class RecurrentDeepQAgent(object):
    def __init__(self, network, state_space, update_frequency=10000, norm=1., batch_size=32, seq_length=1,
                discount=0.99, clip_delta=0.,  memory_size=int(1e5), n_hidden=256,
                optimizer=Adam(.001), double_q_learning=False):

        self.update_frequency = update_frequency
        self.norm = norm
        self.discount = discount
        self.clip_delta = clip_delta
        self.net = network
        self.optimizer = optimizer
        self.update_counter = 0
        self.double_q_learning = double_q_learning
        self.state_space = state_space
        self.replay_memory = RecurrentMemory(state_space, memory_size)
        self.n_hidden = self.net.num_units
        self.batch_size = batch_size
        self.seq_length = seq_length

    def init(self, num_actions):
        self.num_actions = num_actions
        state = T.tensor3('state')
        next_state = T.tensor3('next_state')
        reward = T.matrix('reward')
        action = T.imatrix('action')
        done = T.imatrix('done')
        hidden = T.matrix('hidden')
        mask = T.imatrix('mask')
        next_mask = T.imatrix('next_mask')

        self.network = self.net.build_network(hidden, num_actions, shape=(None, None)+self.state_space)

        if self.update_frequency > 0:
            self.target_network = self.net.build_network(hidden, num_actions,shape=(None, None)+self.state_space)
            self.update_q_hat()

        self.seq_shared = theano.shared(
            np.zeros((self.batch_size, self.seq_length) + self.state_space,
                     dtype=theano.config.floatX))
        self.reward_shared = theano.shared(
            np.zeros((self.batch_size, self.seq_length), dtype=theano.config.floatX),
            broadcastable=(False, False))
        self.action_shared = theano.shared(
            np.zeros((self.batch_size, self.seq_length), dtype='int32'),
            broadcastable=(False, False))
        self.done_shared = theano.shared(
            np.zeros((self.batch_size, self.seq_length), dtype='int32'),
            broadcastable=(False, False))
        self.mask_shared = theano.shared(np.zeros((self.batch_size, self.seq_length+1), dtype='int32'))

        self.state_shared = theano.shared(
            np.zeros(self.state_space,
                     dtype=theano.config.floatX))
        self.hidden_shared = theano.shared(
            np.zeros(self.n_hidden, dtype=theano.config.floatX))

        output = lasagne.layers.get_output([self.network['l_out']]+ self.network['l_hidden'],
                                inputs={self.network['l_in']: state/self.norm, self.network['l_mask']: mask})

        q_vals, next_hidden = output[0], output[1:]

        next_q_vals = lasagne.layers.get_output(self.network['l_out'], inputs={self.network['l_in']: next_state/self.norm,
                                                    self.network['l_mask']: next_mask})
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        double_q_vals = lasagne.layers.get_output(self.target_network['l_out'],
                            inputs={self.target_network['l_in']: next_state/self.norm, self.target_network['l_mask']: next_mask})

        doneX = done.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).dimshuffle(('x','x',0)),
                          action.reshape((self.batch_size, self.seq_length, 1))).astype(theano.config.floatX)

        next_action = T.argmax(next_q_vals, axis=1, keepdims=True).astype('int32')
        next_actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          next_action).astype(theano.config.floatX)

        if self.double_q_learning:
            target = (reward + (T.ones_like(doneX) - doneX) *
                        self.discount * (double_q_vals * next_actionmask).sum(axis=1).reshape((-1,1)))
        else:
            target = (reward +
                      (T.ones_like(doneX) - doneX) *
                      self.discount * T.max(double_q_vals, axis=2))
        output = (q_vals * actionmask).sum(axis=2)
        diff = T.sum((target - output)*mask, axis=1)

        if self.clip_delta > 0:
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = T.sum(loss)
        params = lasagne.layers.helper.get_all_params(self.network['l_out'])
        updates = self.optimizer(loss, params)

        train_givens = {
            state: self.seq_shared[:, :-1],
            next_state: self.seq_shared[:, 1:],
            reward: self.reward_shared,
            action: self.action_shared,
            done: self.done_shared,
            mask: self.mask_shared[:,:-1],
            next_mask: self.mask_shared[:, 1:],
            hidden: np.zeros((1, self.n_hidden), dtype=theano.config.floatX)
        }

        print "Compiling..."
        t = time()
        self._train = theano.function([], [loss], updates=updates, givens=train_givens)

        q_givens = {
            state: self.state_shared.reshape((1, 1)+self.state_space),
            hidden: self.hidden_shared.reshape((1,+self.n_hidden)),
            mask: np.ones((1,1), dtype='int32')
        }
        self._q_vals = theano.function([], [q_vals[0]]+next_hidden, givens=q_givens)
        print '%.2f to compile.'%(time()-t)

    def update_q_hat(self):
        all_params = lasagne.layers.get_all_param_values(self.network['l_out'])
        lasagne.layers.set_all_param_values(self.target_network['l_out'], all_params)

    def save(self, filename):
        all_params = lasagne.layers.get_all_param_values(self.network['l_out'])
        np.save(filename, all_params)

    def load(self, filename):
        all_params = np.load(filename)
        lasagne.layers.set_all_param_values(self.network['l_out'], all_params)
        lasagne.layers.set_all_param_values(self.target_network['l_out'], all_params)

    def train(self):
        state, mask, action, reward, done = self.replay_memory.sample(self.seq_length, self.batch_size)
        self.seq_shared.set_value(state)
        self.action_shared.set_value(action)
        self.reward_shared.set_value(reward)
        self.done_shared.set_value(done)
        self.mask_shared.set_value(mask)
        if (self.update_frequency > 0 and
            self.update_counter % self.update_frequency == 0):
            self.update_q_hat()
        loss = self._train()
        self.update_counter += 1

        return np.sqrt(loss)

    def q_vals(self, state, hidden):
        self.state_shared.set_value(state)
        self.hidden_shared.set_value(hidden.squeeze())
        return self._q_vals()

    def get_action(self, state, eps=0.1):
        q_vals, self.hidden = self.q_vals(state.squeeze(), self.hidden)
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)
        return np.argmax(q_vals)

    def reset(self):
        self.hidden = np.zeros(self.n_hidden, dtype=theano.config.floatX)
