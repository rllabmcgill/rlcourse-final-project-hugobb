import lasagne

class MLP(object):
    def __init__(self, num_units=64, n_layers=1):
        self.num_units = num_units
        self.n_layers = n_layers

    def build_network(self, output_dim, shape):
        l_hidden = lasagne.layers.InputLayer(shape=shape)
        for i in range(self.n_layers):
            l_hidden = lasagne.layers.DenseLayer(
                l_hidden,
                num_units=self.num_units,
                nonlinearity=lasagne.nonlinearities.rectify,
            )

        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=None
        )

        return l_out

class RNN(object):
    def __init__(self, num_units=64, n_layers=1):
        self.num_units = num_units
        self.n_layers = n_layers

    def build_network(self, hidden, output_dim, shape):
        network = {}
        network['l_in'] = lasagne.layers.InputLayer(shape=shape)
        network['l_mask'] = lasagne.layers.InputLayer(shape=shape)
        network['l_hidden'] = []
        for i in range(self.n_layers):
            if i == 0:
                input = network['l_in']
            else:
                input = network['l_hidden'][-1]

            network['l_hidden'].append(lasagne.layers.LSTMLayer(
                input,
                num_units=self.num_units,
                hid_init=hidden,
                mask_input=network['l_mask'],
                grad_clipping=10.,
                forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.))
            ))

        network['l_out'] = lasagne.layers.DenseLayer(
            network['l_hidden'][-1],
            num_units=output_dim,
            nonlinearity=None,
            num_leading_axes=2
        )

        return network
