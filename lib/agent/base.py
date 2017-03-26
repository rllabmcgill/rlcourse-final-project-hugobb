import numpy as np

class Agent(object):
    def __init__(self, state_space, action_space, goal=None):
        self.state_space = state_space
        self.action_space = action_space
        self.state = (0,0)
        self.done = False
        self.goal = goal

    def get_action(self, observation):
        action = np.random.choice(self.action_space)
        return action

    def update(self):
        raise NotImplementedError
