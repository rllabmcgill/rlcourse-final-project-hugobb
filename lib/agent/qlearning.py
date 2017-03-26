import numpy as np
from base import Agent

class QLearning(Agent):
    def __init__(self, state_space, action_space, gamma=0.9, lr=0.05, epsilon=0.9, *args, **kwargs):
        super(QLearning, self).__init__(state_space, action_space, *args, **kwargs)
        self.q = np.zeros(state_space + (len(action_space),))
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.action_space = ['left', 'right', 'up', 'down']

    def get_action(self, observation):
        if np.random.binomial(1, self.epsilon):
            return np.random.choice(self.action_space)

        action = self.action_space[self.q[observation].argmax()]

        return action

    def update(self, observation, action, new_observation, reward):
        action = self.action_space.index(action)
        self.q[observation+(action,)] += self.lr*(reward + self.gamma*self.q[new_observation].max() - self.q[observation+(action,)])

    def save(self, filename):
        np.save(filename, self.q)

    def load(self, filename):
        self.q = np.load(filename)
