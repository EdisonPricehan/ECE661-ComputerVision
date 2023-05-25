
import numpy as np
import math as m
import cv2
from typing import List


class LBP:
    def __init__(self, P: int = 8, R: float = 1.):
        """
        Local Binary Pattern class
        :param P: number of rays from each center pixel
        :param R: radius of circle
        """
        self.P = P
        self.R = R

    def get_hist(self, img_path: str) -> dict:
        """
        Get histogram of occurrences of encoded integers
        :param img_path: image path
        :return: dictionary from integer to number of occurrences
        """
        # read in the image as grayscale, then resize to a small size
        width, height = 64, 64
        img_gray = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_AREA)
        img_gray = img_gray[..., 0]  # get 2d array

        hist = {b: 0 for b in range(self.P + 2)}
        # calculate pattern for each pixel (excluding 4 boundary pixels)
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                pattern = []

                # positive x downward, positive y rightward
                for p in range(self.P):
                    delta_k = self.R * m.cos(2 * m.pi * p / self.P)  # x component
                    delta_l = self.R * m.sin(2 * m.pi * p / self.P)  # y component

                    # bilinear interpolation
                    # TODO consider R > 1
                    A = img_gray[i, j]  # A is current pixel grayness
                    if delta_k > 0:
                        D = img_gray[i + 1, j + 1] if delta_l > 0 else img_gray[i + 1, j - 1]  # D is diagonal
                        B = img_gray[i, j + 1] if delta_l > 0 else img_gray[i, j - 1]  # B is along l
                        C = img_gray[i + 1, j]  # C is along k
                    else:
                        D = img_gray[i - 1, j + 1] if delta_l > 0 else img_gray[i - 1, j - 1]
                        B = img_gray[i, j + 1] if delta_l > 0 else img_gray[i, j - 1]
                        C = img_gray[i - 1, j]

                    # only need absolute values of delta_k and delta_l
                    delta_k, delta_l = abs(delta_k), abs(delta_l)

                    # calculate weighted average of point at direction p and radius R
                    value = (1 - delta_k) * (1 - delta_l) * A + (1 - delta_k) * delta_l * B + \
                        delta_k * (1 - delta_l) * C + delta_k * delta_l * D
                    pattern.append(int(value > A))

                # print(f"Pattern for {i, j} is {pattern}.")

                # rearrange pattern and update histogram by integer encoding
                enc = self.get_int_encoding(pattern)
                hist[enc] += 1
        assert sum(list(hist.values())) == (width - 2) * (height - 2)
        return hist

    def _bin_list_to_integer(self, pattern: List[int]) -> int:
        """
        Encode binary list to integer
        :param pattern: list of binary numbers
        :return: integer
        """
        assert len(pattern) == self.P, "Pattern length not match!"
        result = 0
        for i in range(self.P):
            result += m.pow(2, self.P - i - 1) * pattern[i]
        return result

    def _bin_list_shift(self, pattern: List[int], step: int):
        """
        Rotate binary list by step, 0 step returns original pattern
        :param pattern: list of binary numbers
        :param step: number of times to rotate the list
        :return:
        """
        return np.roll(pattern, step).tolist()

    def _get_min_int_bin_list(self, pattern: List[int]) -> List[int]:
        """
        Get the binary list that has the minimum integer encoding
        :param pattern: list of binary numbers
        :return: list of binary numbers
        """
        assert len(pattern) == self.P, "Pattern length not match!"
        min_bin_list = pattern
        min_int = self._bin_list_to_integer(min_bin_list)
        for s in range(1, self.P):
            bin_list = self._bin_list_shift(pattern, s)
            val = self._bin_list_to_integer(bin_list)
            if val < min_int:
                min_int = val
                min_bin_list = bin_list
        return min_bin_list

    def _get_runs_num(self, pattern: List[int]) -> int:
        """
        Get number of continuous 1s and continuous 0s in a binary list
        :param pattern: list of binary numbers
        :return: number of runs
        """
        assert len(pattern) == self.P, "Pattern length not match!"
        num_runs = 1
        last_elem = pattern[0]
        for i in range(1, self.P):
            cur_elem = pattern[i]
            if cur_elem != last_elem:
                num_runs += 1
                last_elem = cur_elem
        return num_runs

    def get_int_encoding(self, pattern: List[int]) -> int:
        """
        Get the integer encoding based on the rule that continuous 1s and continuous 0s contain more feature.
        All 0s give 0, all 1s give P, 2 runs give the number of 1s, more than 2 runs give P+1
        :param pattern: list of binary numbers
        :return: encoded integer in [0, P+1], in total P+2 choices
        """
        # get minimum integer binary list
        min_int_pattern = self._get_min_int_bin_list(pattern)

        # judge on encoding integer according to run of 1s
        runs = self._get_runs_num(min_int_pattern)
        assert runs >= 1, 'Number of runs must be at least 1!'

        if runs > 2:
            return self.P + 1
        elif runs == 2:
            return np.count_nonzero(min_int_pattern)
        elif runs == 1:
            return 0 if min_int_pattern[0] == 0 else self.P


def plot_histogram(img_path: str):
    """
    Plot image and LBP histogram side by side
    :param img_path:
    :return:
    """
    import matplotlib.pyplot as plt
    img = cv2.imread(img_path)
    lbp = LBP()
    hist = lbp.get_hist(img_path)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].bar(range(len(hist)), list(hist.values()))

    plt.show()
    # plt.axis('off')
    # plt.savefig('hist', bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    # img_path = 'HW7-Auxilliary/data/training/cloudy1.jpg'
    # img_path = 'HW7-Auxilliary/data/training/rain1.jpg'
    # img_path = 'HW7-Auxilliary/data/training/shine1.jpg'
    img_path = 'HW7-Auxilliary/data/training/sunrise1.jpg'

    # plot_histogram(img_path)
    # exit()

    training_path = 'HW7-Auxilliary/data/training'
    testing_path = 'HW7-Auxilliary/data/testing'

    training_csv = 'train_lbp.csv'
    testing_csv = 'test_lbp.csv'

    # extract texture feature vector, store it in csv file along with its image label
    import glob
    import csv
    import tqdm
    import os
    lbp = LBP()

    # print(glob.glob(testing_path + '/*.jpg'))
    for img in tqdm.tqdm(glob.glob(testing_path + '/*.jpg')):
        print(f"Processing {img} ...")
        hist = lbp.get_hist(img)
        feature = list(hist.values())
        base_name = os.path.basename(img)
        if 'cloudy' in base_name:
            label = 0
        elif 'rain' in base_name:
            label = 1
        elif 'shine' in base_name:
            label = 2
        elif 'sunrise' in base_name:
            label = 3
        else:
            print(f"Cannot find label from image name {img}!")
            continue
        feature.append(label)
        print(f"{feature}")
        with open(testing_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(feature)

    print(f"All features and labels have been written to csv file!")
