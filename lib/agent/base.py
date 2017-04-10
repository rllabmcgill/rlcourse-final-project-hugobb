import numpy as np

class Agent(object):
    def __init__(self, action_space, goal=None):
        self.action_space = action_space
        self.done = False
        self.goal = goal
        self.state = None

    def get_action(self, observation):
        action = np.random.choice(self.action_space)
        return action

    def update_state(self, state):
        self.state = state

    def reset(self, state):
        self.done = False
        self.state = state
