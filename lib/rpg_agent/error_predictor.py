import theano
import theano.tensor as T
import lasagne
from lasagne.updates import adam

class ErrorPredictor(object):
    """
    Predicts a baseline that depends on an history (POMDP)
    """
    def __init__(self, feature_size, hid_size, learning_rate):
        """
        feature_size: size of the inputs to be passed at each timestep
        learning_rate: learning rate
        hid_size: cell/hidden state size
        """
        l_in, l_out = self._build_lstm(feature_size, hid_size)

        X = l_in.input_var
        Y_pred = lasagne.layers.get_output(l_out)
        Y_true = T.matrix('targets', dtype=theano.config.floatX)

        loss = lasagne.objectives.squared_error(Y_pred, Y_true)
        mean_loss = loss.mean()
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        updates = adam(mean_loss, params, learning_rate=learning_rate)

        self.predict_f = theano.function([X], Y_pred)
        self.train_f = theano.function([X, Y_true],
                                       mean_loss, updates=updates)

    def _build_lstm(self, n_in, n_h):
        """
        build a lstm with 1 output unit with ReLU
        returns first and last lasagne layer of lstm
        """
        l_in = lasagne.layers.InputLayer(shape=(None, None, n_in))
        l_lstm = lasagne.layers.LSTMLayer(l_in, n_h, learn_init = True, peepholes = False)
        l_shp = lasagne.layers.ReshapeLayer(l_lstm, (-1, n_h))
        l_out_pre = lasagne.layers.DenseLayer(l_shp, 1, nonlinearity=lasagne.nonlinearities.linear)
        batchsize, seqlen, _ = l_in.input_var.shape
        l_out = lasagne.layers.ReshapeLayer(l_out_pre, (batchsize, seqlen))
        return l_in, l_out

    def predict(self, observations):
        return self.predict_f(observations)

    def train(self, observations, targets):
        return self.train_f(observations, targets)


