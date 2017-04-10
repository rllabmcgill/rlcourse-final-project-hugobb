import lasagne
import theano.tensor as T
import theano
import numpy as np
from time import time
from ..updates import Adam
from base import Agent

class DeepQAgent(Agent):
    def __init__(self, action_space, network, gamma=0.9, optimizer=Adam(), *args, **kwargs):
        super(DeepQAgent, self).__init__(action_space, *args, **kwargs)
        self.network = network

        state = T.matrix('state')
        next_state = T.matrix('next_state')
        reward = T.iscalar('reward')
        action = T.iscalar('action')
        done = T.iscalar('done')

        q_vals = lasagne.layers.get_output(self.network, inputs=state).reshape((-1,))

        next_q_vals = lasagne.layers.get_output(self.network, inputs=next_state).reshape((-1,))
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = reward + (1 - done)*gamma*T.max(next_q_vals)
        output = q_vals[action]
        diff = target - output
        loss = T.sum(0.5 * diff ** 2)

        params = lasagne.layers.helper.get_all_params(self.network)
        updates = optimizer(loss, params)

        print "Compiling..."
        t = time()
        self._train = theano.function([state, next_state, reward, action, done], loss, updates=updates)
        self._q_vals = theano.function([state], q_vals)
        print '%.2f to compile.'%(time()-t)

    def save(self, filename):
        all_params = lasagne.layers.get_all_param_values(self.network)
        np.save(filename, all_params)

    def load(self, filename):
        all_params = np.load(filename)
        lasagne.layers.set_all_param_values(self.network, all_params)

    def update_q(self, observation, action, new_observation, reward, done):
        action = self.action_space.index(action)
        observation = np.reshape(observation, (1,-1)).astype(theano.config.floatX)
        new_observation = np.reshape(observation, (1,-1)).astype(theano.config.floatX)
        loss = self._train(observation, new_observation, action, reward, done)
        return np.sqrt(loss)

    def get_action(self, observation, epsilon):
        if np.random.binomial(1, epsilon):
            return np.random.choice(self.action_space)

        q_vals = self._q_vals(observation)
        action = self.action_space[np.argmax(q_vals)]
        return action
