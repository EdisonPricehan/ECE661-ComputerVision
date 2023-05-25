import numpy as np
from scipy.signal import convolve2d
import cv2


class Haar:
    def __init__(self, img: np.array, kernel_size: int = 5):
        """
        Extract Haar gradients of input image and vectorize them as image feature vector
        :param img:
        :param kernel_size:
        """
        assert len(img.shape) == 2, "Image for Haar filter should be 2D gray image!"
        h, w = img.shape
        # print(f"Image shape: {h=} {w=}")
        assert kernel_size < h and kernel_size < w, f"Kernel size {kernel_size} should be less than image size {h, w}!"

        self.img = img
        self.haar_width = kernel_size

    def get_haar_kernel(self, axis=0):
        """
        Haar kernel matrix
        :param axis: 0 along image height, 1 along image width
        :return: haar matrix with pre-calculated width
        """
        kernel = np.ones((self.haar_width, self.haar_width))
        if axis == 0:
            kernel[int(self.haar_width / 2):, :] = -1
        else:
            kernel[:, :int(self.haar_width / 2)] = -1
        return kernel

    def get_gradient_image_vec(self):
        """
        Kernel filtering input image to get gradient vector
        :return: gradient image
        """
        haar_kernel_x = self.get_haar_kernel(axis=0)
        haar_kernel_y = self.get_haar_kernel(axis=1)
        # print(f"{haar_kernel_x=}")
        # print(f"{haar_kernel_y=}")
        dx_img = convolve2d(self.img, haar_kernel_x, mode='valid')  # no padding
        dy_img = convolve2d(self.img, haar_kernel_y, mode='valid')  # np padding
        # print(f"{dx_img.shape=}")

        grad_img_vec = np.stack((dx_img, dy_img), axis=-1).flatten()
        # print(f"{grad_img_vec=}")
        return grad_img_vec


if __name__ == '__main__':
    test_img_path = 'CarDetection/train/positive/000001.png'
    test_img = cv2.imread(test_img_path, cv2.COLOR_BGR2GRAY)[..., 0]

    haar = Haar(test_img, kernel_size=9)
    print(f"{len(haar.get_gradient_image_vec())=}")

