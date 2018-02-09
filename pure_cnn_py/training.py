import cv2
import json
import random
from pure_cnn_py.util.constant import \
    LAYER_CONV, \
    LAYER_MAXPOOL, \
    LAYER_INPUT, \
    LAYER_FULLY_CONNECTED, \
    ACTIVATION_SOFTMAX
from pure_cnn_py.model.model import PureCnn
from pure_cnn_py.util.image_data import ImageData

from pure_cnn_py.util.image_manipulation import write_image_from_image_data, \
    show_image_from_image_data


class Training(object):
    def __init__(self):
        self.train_num = 50000
        self.test_num = 10000
        self.validate_num = 10000

        self.test_offset = 60000
        self.validate_offset = 50000

        self.image_per_file_num = 10000

        self.mini_batch_size = 20
        self.validate_size = 100

        self.iter = 0
        self.epoch = 0
        self.example_seen = 0

        self.train_images = []
        self.test_images = None
        self.validate_images = None
        self.labels = None

        self.model = None

        self.load_data_from_file()
        self.initialize_network()

    def load_data_from_file(self):
        train_file_template = "./resource/mnist_training_{0}.png"
        for i in range(0, 5):
            self.train_images.append(cv2.cvtColor(cv2.imread(train_file_template.format(i)), cv2.COLOR_BGR2RGBA))

        test_file_template = "./resource/mnist_test.png"
        self.test_images = cv2.cvtColor(cv2.imread(test_file_template), cv2.COLOR_BGR2RGBA)

        validate_file_template = "./resource/mnist_validation.png"
        self.validate_images = cv2.cvtColor(cv2.imread(validate_file_template), cv2.COLOR_BGR2RGBA)

        label_file_template = "./resource/mnist_label.txt"
        with open(label_file_template, 'r') as f:
            self.labels = [int(s) for s in f.read().strip().split(",")]

    def initialize_network(self):
        self.model = PureCnn(self.mini_batch_size)
        self.model.add_layer({
            "name": "image",
            "type": LAYER_INPUT,
            "width": 24,
            "height": 24,
            "depth": 1})
        self.model.add_layer({
            "name": "conv1",
            "type": LAYER_CONV,
            "units": 10,
            "kernel_width": 5,
            "kernel_height": 5,
            "stride_x": 1,
            "stride_y": 1,
            "padding": False
        })
        self.model.add_layer({
            "name": "pool1",
            "type": LAYER_MAXPOOL,
            "pool_width": 2,
            "pool_height": 2,
            "stride_x": 2,
            "stride_y": 2
        })
        self.model.add_layer({
            "name": "conv2",
            "type": LAYER_CONV,
            "units": 20,
            "kernel_width": 5,
            "kernel_height": 5,
            "stride_x": 1,
            "stride_y": 1,
            "padding": False
        })
        self.model.add_layer({
            "name": "pool2",
            "type": LAYER_MAXPOOL,
            "pool_width": 2,
            "pool_height": 2,
            "stride_x": 2,
            "stride_y": 2
        })
        self.model.add_layer({
            "name": "out",
            "type": LAYER_FULLY_CONNECTED,
            "units": 10,
            "activation": ACTIVATION_SOFTMAX
        })

    def train(self):
        if self.model is None:
            return

        while True:
            if self.iter < self.train_num:
                train_image_batch = []
                train_label_batch = []
                i = 0
                while True:
                    if i < self.mini_batch_size and self.iter < self.train_num:
                        train_image_batch.append(self.get_train_image_data(self.iter))
                        train_label_batch.append(self.labels[self.iter])
                        i += 1
                        self.iter += 1
                    else:
                        break
                self.example_seen += self.mini_batch_size
                self.model.train(train_image_batch, train_label_batch)
                accuracy = self.validate_accuracy()

                self.epoch += 1
                print(self.example_seen, self.iter, self.model.training_error, accuracy)
                self.save_model("./model.json")
            else:
                break

    def get_train_image_data(self, ite):
        image_file_index = ite // self.image_per_file_num
        image_index = ite % self.image_per_file_num

        start_x = 28 * (image_index % 100) + random.randint(0, 4)
        start_y = 28 * (image_index // 100) + random.randint(0, 4)

        return ImageData(self.train_images[image_file_index][start_y:(start_y + 24), start_x:(start_x + 24)])

    def get_validate_image_data(self, ite):
        start_x = 28 * (ite % 100) + random.randint(0, 4)
        start_y = 28 * (ite // 100) + random.randint(0, 4)

        return ImageData(self.validate_images[start_y:(start_y + 24), start_x:(start_x + 24)])

    def validate_accuracy(self):
        correct = 0
        image_data_list = []
        image_label_list = []

        for i in range(0, 10, 10):
            for j in range(0, 10):
                validate_image_index = int(random.random() * self.validate_num)
                validate_image_label = self.labels[validate_image_index + self.validate_offset]

                for rand in range(0, 1):
                    image_data_list.append(self.get_validate_image_data(validate_image_index))
                    image_label_list.append(validate_image_label)

            result = self.model.predict(image_data_list)

            for m in range(0, 10):
                guess = 0
                max_value = 0
                for c in range(0, result[m].shape.depth):
                    c_sum = 0
                    for rand in range(0, 1):
                        c_sum += result[m + rand].get_value_by_coordinate(0, 0, c)

                    if c_sum > max_value:
                        max_value = c_sum
                        guess = c
                if guess == image_label_list[m]:
                    correct += 1
        return correct / 10

    def save_model(self, filename):
        if self.model is None:
            raise Exception("No model found.")

        model_dict = {
            "layers": [],
            "example_seen": self.example_seen,
            "mini_batch_size": self.mini_batch_size,
            "momentum": self.model.momentum,
            "learning_rate": self.model.learning_rate,
            "l2": self.model.l2,
        }

        for layer in self.model.layers:
            layer_dict = {
                "name": layer.name,
                "type": layer.layer_type,
                "index": layer.layer_index
            }

            if layer.layer_type == LAYER_INPUT:
                layer_dict["width"] = layer.output_shape.width
                layer_dict["height"] = layer.output_shape.height
                layer_dict["depth"] = layer.output_shape.depth
            elif layer.layer_type == LAYER_CONV:
                layer_dict["units"] = layer.units
                layer_dict["weight"] = []
                layer_dict["kernel_width"] = layer.kernel_width
                layer_dict["kernel_height"] = layer.kernel_height
                layer_dict["kernel_stride_x"] = layer.kernel_stride_x
                layer_dict["kernel_stride_y"] = layer.kernel_stride_y
                layer_dict["pad_x"] = layer.pad_x
                layer_dict["pad_y"] = layer.pad_y

                for j in range(0, layer.units):
                    layer_dict["weight"].append(layer.kernel[j].get_value())

                layer_dict["biases"] = layer.biases

            elif layer.layer_type == LAYER_MAXPOOL:
                layer_dict["pool_width"] = layer.pool_width
                layer_dict["pool_height"] = layer.pool_height
                layer_dict["pool_stride_x"] = layer.pool_stride_x
                layer_dict["pool_stride_y"] = layer.pool_stride_y

            elif layer.layer_type == LAYER_FULLY_CONNECTED:
                layer_dict["units"] = layer.units
                layer_dict["weight"] = []
                layer_dict["activation"] = layer.activation

                for j in range(0, layer.units):
                    layer_dict["weight"].append(layer.weights[j].get_value())

                layer_dict["biases"] = layer.biases

            else:
                raise Exception("No such layer type {0} supported.".format(layer.layer_type))

            model_dict["layers"].append(layer_dict)

        with open(filename, "w") as f:
            f.writelines(json.dumps(model_dict))


if __name__ == '__main__':
    tr = Training()
    tr.load_data_from_file()
    tr.train()
