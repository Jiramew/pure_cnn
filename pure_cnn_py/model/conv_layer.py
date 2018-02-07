from pure_cnn_py.model.layer import Layer
from pure_cnn_py.util.mat import Mat
from pure_cnn_py.util.mat_shape import MatShape
from pure_cnn_py.util.constant import INIT_ZEROS, \
    INIT_RANDN, \
    LAYER_CONV, \
    ACTIVATION_RELU


class ConvLayer(Layer):
    def __init__(self, name, units, kernel_width, kernel_height, kernel_stride_x, kernel_stride_y, padding, network):
        super(ConvLayer, self).__init__(name, units)

        self.layer_type = LAYER_CONV
        self.activation = ACTIVATION_RELU
        self.input_layer = None

        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_depth = None

        self.kernel_stride_x = 1
        self.kernel_stride_y = 1
        self.pad_x = 0
        self.pad_y = 0

        self.kernel = [0] * self.units
        self.biases = [0] * self.units
        self.kernel_grad = [0] * units
        self.biases_grad = [0] * units

        self.back_error = None

        self.output_shape = None

        self.network = network
        self.input = None
        self.output = None

    def set_input_layer(self, input_layer):
        super(ConvLayer, self).set_input_layer(input_layer)

        output_width = (self.input_shape.width + self.pad_x * 2 - self.kernel_width) // self.kernel_stride_x + 1
        output_height = (self.input_shape.height + self.pad_y * 2 - self.kernel_height) // self.kernel_stride_y + 1

        self.output_shape = MatShape(output_width, output_height, self.units)
        self.kernel_depth = self.input_shape.depth

        kernel_shape = MatShape(self.kernel_width, self.kernel_height, self.kernel_depth)

        for j in range(0, self.units):
            self.kernel[j] = Mat(kernel_shape, INIT_RANDN)
            self.kernel_grad[j] = Mat(kernel_shape, INIT_ZEROS)

    def set_params(self, weight, bias):
        for i in range(0, self.units):
            self.kernel[i].set_value(weight[i])

        self.biases = bias

    def forward(self):
        self.input = self.pre_layer.output
        self.output = [0] * self.network.mini_batch_size

        for i in range(0, self.network.mini_batch_size):
            self.output[i] = Mat(self.output_shape, INIT_ZEROS)

            for j in range(0, self.units):
                for out_y in range(0, self.output_shape.height):
                    for out_x in range(0, self.output_shape.width):
                        for ker_y in range(0, self.kernel_height):
                            input_y = out_y + ker_y
                            for ker_x in range(0, self.kernel_width):
                                input_x = out_x + ker_x
                                for ker_d in range(0, self.kernel_depth):
                                    self.output[i].add_value_by_coordinate(out_x,
                                                                           out_y,
                                                                           j,
                                                                           self.kernel[j].get_value_by_coordinate(ker_x,
                                                                                                                  ker_y,
                                                                                                                  ker_d) *
                                                                           self.input[i].get_value_by_coordinate(
                                                                               input_x, input_y, ker_d))
                        self.output[i].add_value_by_coordinate(out_y, out_x, j, self.biases[j])
            self.output[i].activate(self.activation)

    def backward(self):
        next_layer = self.next_layer

        self.back_error = [0] * self.network.mini_batch_size

        for i in range(0, self.network.mini_batch_size):
            self.back_error[i] = Mat(self.input_shape, INIT_ZEROS)

        for i in range(0, self.network.mini_batch_size):
            for j in range(0, self.units):
                for out_y in range(0, self.output_shape.height):
                    for out_x in range(0, self.output_shape.width):
                        error_delta = next_layer.back_error[i].get_value_by_coordinate(out_x, out_y, j) * (
                            1 if self.output[i].get_value_by_coordinate(out_x, out_y, j) > 0 else 0)
                        error_delta_with_learning_rate = -1 * self.network.batch_learning_rate * error_delta
                        self.biases_grad[j] += error_delta_with_learning_rate

                        for ker_d in range(self.kernel_depth):
                            for ker_y in range(self.kernel_height):
                                input_y = ker_y + out_y
                                for ker_x in range(self.kernel_width):
                                    input_x = ker_x + out_x

                                    self.kernel_grad[j].add_value_by_coordinate(ker_x, ker_y, ker_d,
                                                                                error_delta_with_learning_rate *
                                                                                self.input[i].get_value_by_coordinate(
                                                                                    input_x, input_y, ker_d))

                                    self.back_error[i].add_value_by_coordinate(input_x, input_y, ker_d,
                                                                               error_delta * self.kernel[
                                                                                   j].get_value_by_coordinate(ker_x,
                                                                                                              ker_y,
                                                                                                              ker_d))

    def batch_update(self):
        l2_regularization = 1 - self.network.batch_learning_rate * self.network.l2

        for j in range(0, self.units):
            self.kernel[j].operation_scale_and_add_mat(l2_regularization, self.kernel_grad[j])
            self.biases[j] += self.biases_grad[j]

            self.kernel_grad[j].operation_scale_mat(self.network.momentum)
            self.biases_grad[j] *= self.network.momentum
