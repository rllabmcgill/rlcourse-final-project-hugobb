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
