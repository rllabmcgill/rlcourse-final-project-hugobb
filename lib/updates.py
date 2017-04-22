from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import rmsprop, adam, get_or_compute_grads, sgd

class DeepMindRmsprop(object):
    def __init__(self, learning_rate, rho, epsilon):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def __call__(self, loss_or_grads, params):
        grads = get_or_compute_grads(loss_or_grads, params)
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)

            acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            acc_grad_new = self.rho * acc_grad + (1 - self.rho) * grad

            acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            acc_rms_new = self.rho * acc_rms + (1 - self.rho) * grad ** 2


            updates[acc_grad] = acc_grad_new
            updates[acc_rms] = acc_rms_new

            updates[param] = (param - self.learning_rate *
                              (grad /
                               T.sqrt(acc_rms_new - acc_grad_new **2 + self.epsilon)))

        return updates


class Rmsprop(object):
    def __init__(self, learning_rate=1.0, rho=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def __call__(self, loss_or_grads, params):
        return rmsprop(loss_or_grads, params, self.learning_rate, self.rho, self.epsilon)

class Adam(object):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, loss_or_grads, params):
        return adam(loss_or_grads, params, self.learning_rate, self.beta1,
                    self.beta2, self.epsilon)

class SGD(object):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def __call__(self, loss_or_grads, params):
        return sgd(loss_or_grads, params, learning_rate=self.learning_rate)
