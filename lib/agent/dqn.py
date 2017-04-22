import lasagne
import theano.tensor as T
import theano
from ..updates import DeepMindRmsprop, Adam, SGD
import numpy as np
from time import time
from ..memory import Memory

class DeepQAgent(object):
    def __init__(self, network, state_space, update_frequency=10000, norm=1.,
                discount=0.99, clip_delta=0., memory_size=int(1e5), batch_size=32,
                optimizer=Adam, learning_rate=0.001, double_q_learning=False):

        self.update_frequency = update_frequency
        self.norm = norm
        self.discount = discount
        self.clip_delta = clip_delta
        self.network = network
        self.update_counter = 0
        self.double_q_learning = double_q_learning
        self.state_space = state_space
        self.replay_memory = Memory(state_space, memory_size)
        self.lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))
        self.optimizer = optimizer(learning_rate=self.lr)
        self.batch_size = batch_size

    def init(self, num_actions):
        self.l_out = self.network.build_network(num_actions, shape=(None, 1)+self.state_space)

        if self.update_frequency > 0:
            self.next_l_out = self.network.build_network(num_actions, shape=(None, 1)+self.state_space)
            self.update_q_hat()

        self.num_actions = num_actions
        state = T.tensor3('state')
        next_state = T.tensor3('next_state')
        reward = T.col('reward')
        action = T.icol('action')
        done = T.icol('done')

        self.seq_shared = theano.shared(
            np.zeros((self.batch_size, 1) + self.state_space,
                     dtype=theano.config.floatX))
        self.reward_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.action_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.done_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.state_shared = theano.shared(
            np.zeros((1,)+self.state_space,
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, inputs=state/self.norm)

        next_q_vals = lasagne.layers.get_output(self.l_out, inputs=next_state/self.norm)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        double_q_vals = lasagne.layers.get_output(self.next_l_out, inputs=next_state/self.norm)

        doneX = done.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          action.reshape((-1, 1))).astype(theano.config.floatX)

        next_action = T.argmax(next_q_vals, axis=1, keepdims=True).astype('int32')
        next_actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          next_action).astype(theano.config.floatX)

        if self.double_q_learning:
            target = (reward + (T.ones_like(doneX) - doneX) *
                        self.discount * (double_q_vals * next_actionmask).sum(axis=1).reshape((-1,1)))
        else:
            target = (reward +
                      (T.ones_like(doneX) - doneX) *
                      self.discount * T.max(double_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        if self.clip_delta > 0:
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = T.sum(loss)
        params = lasagne.layers.helper.get_all_params(self.l_out)
        updates = self.optimizer(loss, params)

        train_givens = {
            state: self.seq_shared[:, :-1],
            next_state: self.seq_shared[:, 1:],
            reward: self.reward_shared,
            action: self.action_shared,
            done: self.done_shared
        }

        print "Compiling..."
        t = time()
        self._train = theano.function([], [loss], updates=updates, givens=train_givens)

        q_givens = {
            state: self.state_shared.reshape((1, 1)+self.state_space)
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)
        print '%.2f to compile.'%(time()-t)

    def update_q_hat(self):
        all_params = lasagne.layers.get_all_param_values(self.l_out)
        lasagne.layers.set_all_param_values(self.next_l_out, all_params)

    def save(self, filename):
        all_params = lasagne.layers.get_all_param_values(self.l_out)
        np.save(filename, all_params)

    def load(self, filename):
        all_params = np.load(filename)
        lasagne.layers.set_all_param_values(self.l_out, all_params)
        lasagne.layers.set_all_param_values(self.next_l_out, all_params)

    def train(self):
        state, action, reward, done = self.replay_memory.sample()
        self.seq_shared.set_value(state)
        self.action_shared.set_value(action)
        self.reward_shared.set_value(reward)
        self.done_shared.set_value(done)
        if (self.update_frequency > 0 and
            self.update_counter % self.update_frequency == 0):
            self.update_q_hat()
        loss = self._train()
        self.update_counter += 1

        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def get_action(self, state, eps=0.1):
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset(self):
        return
