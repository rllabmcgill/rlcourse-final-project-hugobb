import numpy as np
from collections import deque
from random import shuffle

class ExperienceReplay(object):
    """
    naive implementation of Experience Replay
    trajectories are stored in a list with a maximum length
    so that old trajectories are discarded when new ones are added
    """
    def __init__(self, maxlen):
        """
        maxlen is the size of the experience replay buffer 
        i.e. the # of trajectories to store
        """
        self.maxlen = maxlen
        self.exp = []

    def append(self, h):
        self.exp.append(h)
        if len(self.exp) >= self.maxlen:
            self.exp = self.exp[1:]

    def __len__(self):
        return len(self.exp)

    def get_batch(self, size, random_order=True):
        """
        if random order: return random 
        else: return lastly inserted
        """
        if random_order:
            idx = np.random.choice(len(self.exp), 
                                   size=min(len(self.exp), size), replace=False)
            return [self.exp[i] for i in idx]
        else:
            return self.exp[-size:]

    def __getstate__(self):
        return (self.exp, self.maxlen)

    def __setstate__(self, s):
        self.exp, self.maxlen = s
