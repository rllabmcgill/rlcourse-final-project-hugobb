#!/usr/bin/env python

import pickle
import numpy as np
from lib.rpg_agent.rpg_ep import RPGRecurrentBaseline
import theano.misc.pkl_utils as pkl_utils

def test_agent_save():
    agent1 = RPGRecurrentBaseline(2, 2, 2, 'softmax', 0.99, 0.001, freq_train=10, maxlen=500)
    k = 20
    for _ in range(k):
        for _ in range(k):
            agent1.update(np.asarray([0.,1.]), 1, None, 6, False)
        agent1.reset(3)

    assert(len(agent1.experience) == k)
    with open("tmp.zip", 'wb') as f:
        agent1_dump = pkl_utils.dump(agent1, f)

    with open('tmp.zip', 'rb') as f:
        agent2 = pkl_utils.load(f)

    assert(len(agent2.experience) == k)
    print "reload and retrain"
    # try to force the retraining
    for _ in range(k):
        agent2.reset(3)



