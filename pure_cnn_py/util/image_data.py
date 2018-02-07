from pure_cnn_py.util.flatten import flatten


class ImageData(object):
    def __init__(self, cv2_image):
        self.width = cv2_image.shape[0]
        self.height = cv2_image.shape[1]
        self.data = self._get_data(cv2_image)

    @staticmethod
    def _get_data(cv2_image):
        return flatten(cv2_image.tolist())


if __name__ == '__main__':
    import cv2

    im = ImageData(cv2.cvtColor(cv2.imread("../resource/mnist_test.png"), cv2.COLOR_BGR2RGBA))
