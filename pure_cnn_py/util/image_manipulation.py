import cv2
import numpy as np
from matplotlib import pyplot as plt


def write_image_from_image_data(image_data):
    image = np.array(image_data.data).reshape([24, 24, 4])[:, :, 0:3]
    cv2.imwrite("image_demo.png", image)


def show_image_from_image_data(image_data):
    image = np.array(image_data.data).reshape([24, 24, 4])[:, :, 0:3]
    plt.imshow("image_demo.png", image)
