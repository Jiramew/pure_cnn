import math

from pure_cnn_py.util.randn import Randn
from pure_cnn_py.util.constant import \
    INIT_RANDN, \
    INIT_ZEROS, \
    ACTIVATION_RELU, \
    ACTIVATION_TANH, \
    ACTIVATION_SOFTMAX


class Mat(object):
    def __init__(self, shape, init_type=INIT_ZEROS):
        self.shape = shape
        self.size = shape.get_size()
        self.value = [0] * self.size

        if init_type == INIT_RANDN:
            sd = 1 / math.sqrt(self.size)
            randn = Randn()
            for i in range(0, self.size):
                self.value[i] = sd * randn.get_randn()

    def __getitem__(self, n):
        return self.value[n]

    def operation_scale_and_add_mat(self, scale, add_mat):
        for i in range(0, self.size):
            self.value[i] *= scale
            self.value[i] += add_mat[i]

    def operation_add_scaled_mat(self, scaled, add_mat):
        for i in range(0, self.size):
            self.value[i] += scaled * add_mat[i]

    def operation_scale_mat(self, scale):
        for i in range(0, self.size):
            self.value[i] *= scale

    def activate(self, act_type):
        if act_type == ACTIVATION_RELU:
            for i in range(0, self.size):
                self.value[i] = max(0, self.value[i])
        elif act_type == ACTIVATION_TANH:
            for i in range(0, self.size):
                self.value[i] = math.tanh(self.value[i])
        elif act_type == ACTIVATION_SOFTMAX:
            max_value = 0
            sum_value = 0
            for i in range(0, self.size):
                max_value = max(max_value, self.value[i])

            for i in range(0, self.size):
                self.value[i] = math.exp(self.value[i] - max_value)
                sum_value += self.value[i]

            for i in range(0, self.size):
                self.value[i] /= sum_value
        else:
            raise Exception("No such activation type {0}".format(act_type))

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def set_value_by_image(self, image_data, depth):
        scale = 1 / 255

        area = image_data.width * image_data.height

        for d in range(0, depth):
            for h in range(0, image_data.height):
                for w in range(0, image_data.width):
                    img_index = 4 * (image_data.width * h + w)
                    mat_index = area * d + image_data.width * h + w

                    self.value[mat_index] = float(image_data.data[img_index + d] * scale)

    def set_value_by_coordinate(self, x, y, z, v):
        mat_index = (z * self.shape.height + y) * self.shape.width + x
        self.value[mat_index] = v

    def add_value_by_coordinate(self, x, y, z, v):
        mat_index = (z * self.shape.height + y) * self.shape.width + x
        self.value[mat_index] += v

    def get_value_by_coordinate(self, x, y, z):
        mat_index = (z * self.shape.height + y) * self.shape.width + x
        return self.value[mat_index]
