from pure_cnn_py.model.layer import Layer
from pure_cnn_py.util.mat import Mat
from pure_cnn_py.util.mat_shape import MatShape
from pure_cnn_py.util.constant import INIT_ZEROS, LAYER_MAXPOOL


class MaxpoolLayer(Layer):
    def __init__(self, name, pool_width, pool_height, pool_stride_x, pool_stride_y, network):
        super(MaxpoolLayer, self).__init__(name, 0)
        self.layer_type = LAYER_MAXPOOL

        self.pool_width = pool_width
        self.pool_height = pool_height
        self.pool_stride_x = pool_stride_x
        self.pool_stride_y = pool_stride_y

        self.network = network
        self.input = None
        self.output = None

        self.output_width = None
        self.output_height = None
        self.output_depth = None
        self.output_shape = None

        self.max_info = None

        self.back_error = None

        self.pool_max_activation_index = None

    def set_input_layer(self, input_layer):
        super(MaxpoolLayer, self).set_input_layer(input_layer)
        self.output_width = (self.input_shape.width - self.pool_width) // self.pool_stride_x + 1
        self.output_height = (self.input_shape.height - self.pool_height) // self.pool_stride_y + 1
        self.output_depth = self.input_shape.depth

        self.output_shape = MatShape(self.output_width, self.output_height, self.output_depth)

    def forward(self):
        self.input = self.pre_layer.output
        self.output = [0] * self.network.mini_batch_size
        self.max_info = [0] * self.network.mini_batch_size

        for i in range(0, self.network.mini_batch_size):
            self.output[i] = Mat(self.output_shape, INIT_ZEROS)
            self.max_info[i] = Mat(self.output_shape, INIT_ZEROS)

            for out_d in range(0, self.output_depth):
                for out_y in range(0, self.output_height):
                    for out_x in range(0, self.output_width):
                        max_value = 0
                        max_index = [0, 0, 0]
                        for input_y in range(0, self.pool_height):
                            stride_y = out_y * self.pool_stride_y + input_y
                            for input_x in range(0, self.pool_width):
                                stride_x = out_x * self.pool_stride_x + input_x
                                compare_input = self.input[i].get_value_by_coordinate(stride_x, stride_y, out_d)
                                if compare_input >= max_value:
                                    max_value = compare_input
                                    max_index = [stride_x, stride_y, out_d]

                        self.max_info[i].set_value_by_coordinate(out_x, out_y, out_d,
                                                                 "-".join([str(i) for i in max_index]))
                        self.output[i].set_value_by_coordinate(out_x, out_y, out_d, max_value)

    def backward(self):
        next_layer = self.next_layer
        self.back_error = [0] * self.network.mini_batch_size

        for i in range(0, self.network.mini_batch_size):
            self.back_error[i] = Mat(self.input_shape, INIT_ZEROS)
            for mi in self.max_info[i].get_value():
                input_x, input_y, input_d = [int(a) for a in mi.split("-")]
                self.back_error[i].set_value_by_coordinate(input_x,
                                                           input_y,
                                                           input_d,
                                                           next_layer.back_error[i].get_value_by_coordinate(
                                                               input_x // self.pool_width,
                                                               input_y // self.pool_height,
                                                               input_d))
