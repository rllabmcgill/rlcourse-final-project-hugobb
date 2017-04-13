import numpy as np

class RandomAgent():
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def reset(self, o):
        pass

    def sample_action(self, o, r):
        return np.random.choice(self.action_space_size)

    def update(self, o, a, o2, r, done):
        pass


