from pure_cnn_py.util.constant import ACTIVATION_LINEAR


class Layer(object):
    def __init__(self, name, units):
        self.name = name
        self.units = units

        self.activation = ACTIVATION_LINEAR

        self.next_layer = None
        self.pre_layer = None

        self.input_shape = None

        self.layer_index = None
        self.layer_type = None

    def is_last_layer(self):
        return self.next_layer is None

    def set_input_layer(self, input_layer):
        self.pre_layer = input_layer
        self.input_shape = input_layer.output_shape

    def set_output_layer(self, output_layer):
        self.next_layer = output_layer

    def batch_update(self):
        pass
