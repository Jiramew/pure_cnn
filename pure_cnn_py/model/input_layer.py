from pure_cnn_py.model.layer import Layer
from pure_cnn_py.util.mat import Mat
from pure_cnn_py.util.mat_shape import MatShape
from pure_cnn_py.util.constant import INIT_ZEROS, LAYER_INPUT


class InputLayer(Layer):
    def __init__(self, layer_name, image_width, image_height, image_depth, network):
        super(InputLayer, self).__init__(layer_name, 1)
        self.network = network
        self.layer_type = LAYER_INPUT
        self.output_shape = MatShape(image_width, image_height, image_depth)
        self.output = []

    def forward(self, image_data_list):
        for i in range(0, self.network.mini_batch_size):
            self.output.append(Mat(self.output_shape, INIT_ZEROS))
            self.output[i].set_value_by_image(image_data_list[i], self.output_shape.depth)

    def backward(self):
        pass
