import math
from pure_cnn_py.model.layer import Layer
from pure_cnn_py.util.mat import Mat
from pure_cnn_py.util.mat_shape import MatShape
from pure_cnn_py.util.constant import INIT_ZEROS, \
    INIT_RANDN, \
    LAYER_FULLY_CONNECTED, \
    ACTIVATION_TANH


class FCLayer(Layer):
    def __init__(self, name, units, activation, network):
        super(FCLayer, self).__init__(name, units)
        self.layer_type = LAYER_FULLY_CONNECTED
        self.output_shape = MatShape(1, 1, units)
        self.activation = activation

        self.weights = [0] * units
        self.biases = [0] * units

        self.weight_grad = [0] * units
        self.biases_grad = [0] * units

        self.network = network
        self.input = None
        self.output = None

        self.back_error = None

    def set_input_layer(self, input_layer):
        super(FCLayer, self).set_input_layer(input_layer)
        for i in range(0, self.units):
            self.weights[i] = Mat(self.input_shape, INIT_RANDN)
            self.weight_grad[i] = Mat(self.input_shape, INIT_ZEROS)

    def set_output_layer(self, output_layer):
        super(FCLayer, self).set_output_layer(output_layer)

    def set_params(self, weight, bias):
        for i in range(0, self.units):
            self.weights[i].set_value(weight[i])

        self.biases = bias

    def forward(self):
        self.input = self.pre_layer.output
        self.output = [0] * self.network.mini_batch_size

        size = self.input_shape.get_size()

        for i in range(0, self.network.mini_batch_size):
            self.output[i] = Mat(self.output_shape, INIT_ZEROS)

            for j in range(0, self.units):
                for k in range(0, size):
                    self.output[i].value[j] += self.weights[j].value[k] * self.input[i].value[k]

                self.output[i].value[j] += self.biases[j]

            self.output[i].activate(self.activation)

    def backward(self):
        self.back_error = [None] * self.network.mini_batch_size

        for i in range(0, self.network.mini_batch_size):
            self.back_error[i] = Mat(self.input_shape, INIT_ZEROS)

            for j in range(0, self.units):
                if self.is_last_layer():
                    error = self.output[i].value[j] - self.network.label_list_one_hot[i][j]
                else:
                    error = self.next_layer.back_error[i].get_value_by_coordinate(0, 0, j)

                if self.activation == ACTIVATION_TANH:
                    error *= (1 - math.pow(self.output[i].value[j], 2))

                self.network.training_error += abs(error)

                grad = -1 * self.network.batch_learning_rate * error
                self.weight_grad[j].operation_add_scaled_mat(grad, self.input[i])
                self.biases_grad[j] += grad

                self.back_error[i].operation_add_scaled_mat(error, self.weights[j])

    def batch_update(self):
        l2_regularization = 1 - self.network.batch_learning_rate * self.network.l2

        for j in range(0, self.units):
            self.weights[j].operation_scale_and_add_mat(l2_regularization, self.weight_grad[j])
            self.biases[j] += self.biases_grad[j]

            self.weight_grad[j].operation_scale_mat(self.network.momentum)
            self.biases_grad[j] *= self.network.momentum
