import numpy as np

class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        action = np.random.choice(self.action_space)
        return action
