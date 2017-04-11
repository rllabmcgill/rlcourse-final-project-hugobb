import numpy as np
import theano
import theano.tensor as T

def softmax(x):
    exp_x = T.exp(x)
    return exp_x / (T.sum(exp_x) + 10**-6)

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def get_activation_function(func_name):
    """
    returns theano implementation of function
    func_name is either 'relu', 'sigmoid', 'tanh', 'softmax', 'linear'
    """
    return {
        'linear': lambda x: x,
        'relu': lambda x: x * (x > 0),
        'elu': lambda x: x * (x >= 0) + (T.exp(x) - 1) * (x < 0),
        'softmax': T.nnet.softmax,
        'tanh': T.tanh,
        'log_softmax': log_softmax,
        'sigmoid': T.nnet.sigmoid
    }[func_name]

def one_hot_encode(y, out_size):
    """ y is of shape (n) """
    n = len(y)
    oh = np.zeros((n, out_size))
    oh[range(n), y] = 1
    return oh    

def init_weights_bias(shape, activation):
    n_in, n_out = shape
    if activation in ['sigmoid','tanh','softmax','log_softmax','linear']:
        if activation in ['sigmoid','softmax','log_softmax','linear']:
            glorot_coefficient = 4.
        elif activation == 'tanh':
            glorot_coefficient = 1.
        bound = glorot_coefficient * np.sqrt(6. / (n_in + n_out))
        init_weights = np.random.uniform(-bound, bound, (n_in, n_out))
        init_bias = np.random.uniform(-0.05, 0.05, (1, n_out))
    elif activation in ['relu', 'elu']:
        init_weights = np.random.normal(0, 1, (n_in, n_out))
        init_weights *= np.sqrt(2.0/float(n_out))
        init_bias = np.zeros((1,n_out))
    return (init_weights.astype(theano.config.floatX),
            init_bias.astype(theano.config.floatX))


