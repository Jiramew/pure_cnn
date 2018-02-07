import cv2
import random
from pure_cnn_py.util.constant import \
    LAYER_CONV, \
    LAYER_MAXPOOL, \
    LAYER_INPUT, \
    LAYER_FULLY_CONNECTED, \
    ACTIVATION_SOFTMAX
from pure_cnn_py.model.model import PureCnn
from pure_cnn_py.util.image_data import ImageData


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


if __name__ == '__main__':
    tr = Training()
    tr.load_data_from_file()
    tr.train()
