import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt
from typing import Literal, Optional, List


class Otsu:
    def __init__(self, img_path : str, flip: bool = False):
        self.img_rgb = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        self.flip = flip

    def get_thresholded_img(self, channel: Literal['rgb', 'texture', 'h'], iterations: int = 1) \
            -> Optional[np.ndarray]:
        print(f"Thresholded by {channel}")
        if channel == 'h':
            img_hsv = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2HSV)
            # print(f"{img_hsv.shape=}")
            img_h = img_hsv[..., 0]
            # print(f"{img_h.shape=}")
            thr = self._get_best_threshold(img_h)
            return self.binarize(img_h, thr, flip=self.flip)
        elif channel == 'rgb':
            img_chw = np.transpose(self.img_rgb, (2, 0, 1))
        elif channel == 'texture':
            img_chw = self.create_convolved_nd_image(self.img_gray, [3, 5, 7])
        else:
            print(f"Unrecognized intensity scale {channel}.")
            return None

        and_img = np.ones_like(self.img_gray)
        for iter in range(iterations):
            print(f"Current iteration: {iter}")
            c, h, w = img_chw.shape
            and_img_chw = np.zeros_like(img_chw)
            for i in range(c):
                thr = self._get_best_threshold(img_chw[i])
                and_img_chw[i] = self.binarize(img_chw[i], thr, flip=self.flip)

                self._plot(and_img_chw[i])  # plot single channel mask
                contour_img = self.contour_extraction((and_img_chw[i] / 255).astype(int))
                self._plot(contour_img)  # plot single channel contour

            and_img = np.all(and_img_chw == 255, axis=0)  # AND 2d array along channel dimension

            img_chw = np.where(and_img, img_chw, 0)  # keep only foreground, make background 0

            and_img = and_img.astype(int)

        return and_img

    def create_convolved_nd_image(self, img: np.ndarray, window_size_list: List[int]) -> np.ndarray:
        """
        2d image convoluted to 3d image
        :param img: h x w image
        :param window_size_list: n window sizes
        :return: n x h x w image
        """
        assert len(img.shape) == 2, "accept grayscale image only!"
        h, w = img.shape
        c = len(window_size_list)
        img_chw = np.zeros((c, h, w))

        # self._plot(img)

        for i, k in enumerate(window_size_list):
            mean_op = np.ones((k, k)) / k ** 2
            # carry out convolution to compute mean of square, and square of mean
            mean_of_sq = scipy.signal.convolve2d(img ** 2, mean_op, mode='same', boundary='symm')
            sq_of_mean = scipy.signal.convolve2d(img, mean_op, mode='same', boundary='symm') ** 2
            # win_var = mean_of_sq - sq_of_mean
            win_var = sq_of_mean - mean_of_sq

            # self._plot(win_var)

            img_chw[i] = win_var

        # scale variance
        img_chw = (img_chw * 255 / img_chw.max()).astype(int)

        return img_chw

    def _plot(self, img):
        plt.Figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

    def _get_best_threshold(self, arr: np.ndarray) -> int:
        """
        Otsu algorithm that finds the best grayness which maximizes between class scatter
        :param arr: 2d image
        :return: threshold or grayness
        """
        arr_1d = arr[arr != 0]
        mink, maxk = arr_1d.min(), arr_1d.max()
        hist, bin_edges = np.histogram(arr_1d, bins=np.unique(arr_1d), density=True)
        print(f"{hist.shape=} {mink=} {maxk=}")
        # print(f"{bin_edges=}")

        best_k = -1
        best_bc_var = -1  # largest between-class variance
        for i in range(len(hist)):
            w0 = np.sum(hist[:i + 1])
            w1 = 1 - w0
            if w0 == 0 or w1 == 0:
                continue
            mu0 = np.dot(bin_edges[:i + 1], hist[:i + 1]) / w0
            mu1 = np.dot(bin_edges[i + 1: -1], hist[i + 1:]) / w1
            bc_var = w0 * w1 * (mu0 - mu1) ** 2
            if bc_var > best_bc_var:
                best_bc_var = bc_var
                best_k = i
                # print(f"{mu0=} {mu1=}")
        print(f"{best_k=}")

        # plt.Figure()
        # plt.plot(bin_edges[:-1], hist)
        # plt.axvline(x=best_k, color='red')
        # plt.show()

        return best_k

    @staticmethod
    def binarize(arr: np.ndarray, thr: int, flip: bool = False) -> np.ndarray:
        return np.where(arr > thr if not flip else arr <= thr, 255, 0)

    def contour_extraction(self, img: np.ndarray) -> np.ndarray:
        """
        Extract contours by identifying white pixel whose neighbours have at least one black pixel
        :param img:
        :return:
        """
        countour_img = np.zeros_like(img)
        h, w = img.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if img[i, j] == 1:
                    if np.sum(img[i - 1: i + 2, j - 1: j + 2]) < 9:
                        countour_img[i, j] = 1

        return countour_img


if __name__ == '__main__':
    # img_path = 'HW6-Images/cat.jpg'
    # img_path = 'HW6-Images/car.jpg'
    img_path = 'HW6-Images/sq.jpg'
    # img_path = 'HW6-Images/pig.jpg'

    otsu = Otsu(img_path, flip=True)
    # img_thrd = otsu.get_thresholded_img('h')
    # img_thrd = otsu.get_thresholded_img('rgb', iterations=1)
    img_thrd = otsu.get_thresholded_img('texture', iterations=1)

    otsu._plot(img_thrd)

    countour_img = otsu.contour_extraction(img_thrd)
    otsu._plot(countour_img)

