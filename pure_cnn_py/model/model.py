from pure_cnn_py.util.constant import LAYER_INPUT, \
    LAYER_CONV, \
    LAYER_MAXPOOL, \
    LAYER_FULLY_CONNECTED

from pure_cnn_py.model.input_layer import InputLayer
from pure_cnn_py.model.maxpool_layer import MaxpoolLayer
from pure_cnn_py.model.conv_layer import ConvLayer
from pure_cnn_py.model.fc_layer import FCLayer


class PureCnn(object):
    def __init__(self, mini_batch_size):
        self.layers = []
        self.next_layer = []

        self.next_layer_index = 0

        self.learning_rate = 0.01
        self.momentum = 0.9
        self.l2 = 0

        self.mini_batch_size = mini_batch_size
        self.training_error = 0

        self.batch_learning_rate = None
        self.label_list_one_hot = []

    def add_layer(self, layer_info):
        layer_type = layer_info['type']
        if layer_type == LAYER_INPUT:
            new_layer = InputLayer(layer_info['name'],
                                   layer_info['width'],
                                   layer_info['height'],
                                   layer_info['depth'],
                                   self)
        elif layer_type == LAYER_CONV:
            new_layer = ConvLayer(layer_info['name'],
                                  layer_info['units'],
                                  layer_info['kernel_width'],
                                  layer_info['kernel_height'],
                                  layer_info['stride_x'],
                                  layer_info['stride_y'],
                                  layer_info['padding'],
                                  self)
        elif layer_type == LAYER_MAXPOOL:
            new_layer = MaxpoolLayer(layer_info['name'],
                                     layer_info['pool_width'],
                                     layer_info['pool_height'],
                                     layer_info['stride_x'],
                                     layer_info['stride_y'],
                                     self)
        elif layer_type == LAYER_FULLY_CONNECTED:
            new_layer = FCLayer(layer_info['name'],
                                layer_info['units'],
                                layer_info['activation'],
                                self)
        else:
            raise Exception("No such layer info {0} supported.".format(layer_info))

        if self.next_layer_index == 0:
            if new_layer.layer_type != LAYER_INPUT:
                raise Exception("First Layer should be input layer.")
        else:
            pre_layer = self.layers[self.next_layer_index - 1]
            pre_layer.set_output_layer(new_layer)
            new_layer.set_input_layer(pre_layer)

        new_layer.layer_index = self.next_layer_index
        self.layers.append(new_layer)
        self.next_layer_index += 1

    def train(self, image_data_list, image_label_list):
        self.batch_learning_rate = self.learning_rate / self.mini_batch_size

        self.training_error = 0

        self._one_hot(image_label_list)
        self._forward(image_data_list)
        self._backward()
        self._mini_batch()
        self.training_error /= self.mini_batch_size

    def predict(self, image_data_list):
        self.mini_batch_size = len(image_data_list)
        self.batch_learning_rate = self.learning_rate
        self.layers[0].forward(image_data_list)
        for i in range(1, len(self.layers)):
            self.layers[i].forward()
        output_layer = self.layers[-1]
        return output_layer.output

    def _one_hot(self, image_label_list):
        self.label_list_one_hot = [None] * self.mini_batch_size
        for i in range(0, self.mini_batch_size):
            self.label_list_one_hot[i] = [0] * self.layers[-1].units
            for j in range(0, self.layers[-1].units):
                self.label_list_one_hot[i][j] = 1 if j == image_label_list[i] else 0

    def _forward(self, image_data_list):
        self.layers[0].forward(image_data_list)
        for i in range(1, len(self.layers)):
            self.layers[i].forward()

    def _backward(self):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].backward()

    def _mini_batch(self):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].batch_update()
