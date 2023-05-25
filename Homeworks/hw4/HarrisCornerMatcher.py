import cv2
import matplotlib.pyplot as plt

from HarrisCornerDetector import HarrisCornerDetector
import numpy as np


class HarrisCornerMatcher:
    def __init__(self, img1_path, img2_path, window_size=21, sigma=1.2, metric="ssd"):
        self.hcd1 = HarrisCornerDetector(img1_path, sigma=sigma, scale_factor=1)
        self.hcd2 = HarrisCornerDetector(img2_path, sigma=sigma, scale_factor=1)
        self.ws = window_size
        self.metric = metric
        self.corners1 = None
        self.corners2 = None
        self.left_right_flip = False
        self.corrs = []

    def get_correspondences(self):
        self.corners1 = self.hcd1.get_corners()
        self.corners2 = self.hcd2.get_corners()
        cor1_len, cor2_len = len(self.corners1[0]), len(self.corners2[0])

        if cor1_len < cor2_len:
            corners_less = self.corners1
            corners_more = self.corners2
        else:
            corners_less = self.corners2
            corners_more = self.corners1
            self.left_right_flip = True

        border = int(self.ws / 2)
        print(f"{border=}")
        padded_img1 = cv2.copyMakeBorder(self.hcd1.img, border, border, border, border,
                                         borderType=cv2.BORDER_CONSTANT, value=0)
        padded_img2 = cv2.copyMakeBorder(self.hcd2.img, border, border, border, border,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

        for c1x, c1y in zip(*corners_less):
            best_dist = np.inf
            best_corner = None
            mat1 = padded_img1[c1x: c1x + 2 * border + 1, c1y: c1y + 2 * border + 1]
            for c2x, c2y in zip(*corners_more):
                mat2 = padded_img2[c2x: c2x + 2 * border + 1, c2y: c2y + 2 * border + 1]
                cur_dist = self.ssd(mat1, mat2) if self.metric == "ssd" else self.ncc(mat1, mat2)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_corner = (c2x, c2y)
            self.corrs.append([(c1x, c1y), best_corner, best_dist])
        print(f"Found {len(self.corrs)} correspondences.")
        return self.corrs

    def display(self):
        img1_rgb = self.hcd1.img_rgb.copy()
        img1_rgb[self.corners1] = [255, 0, 0]
        img2_rgb = self.hcd2.img_rgb.copy()
        img2_rgb[self.corners2] = [255, 0, 0]
        if self.left_right_flip:
            img_cat = np.concatenate((img2_rgb, img1_rgb), axis=1)
            ofs_y = img2_rgb.shape[1]
        else:
            img_cat = np.concatenate((img1_rgb, img2_rgb), axis=1)
            ofs_y = img1_rgb.shape[1]

        plt.imshow(img_cat)
        corners_left = np.array([c[0] for c in self.corrs])
        corners_right = np.array([c[1] for c in self.corrs])
        plt.scatter(corners_left[:, 1], corners_left[:, 0], s=30, facecolors='none', edgecolors='r')
        plt.scatter(corners_right[:, 1] + ofs_y, corners_right[:, 0], s=30, facecolors='none', edgecolors='r')
        for c1, c2, _ in self.corrs:
            plt.plot([c1[1], c2[1] + ofs_y], [c1[0], c2[0]])
        plt.show()

    @staticmethod
    def ssd(mat1, mat2):
        assert mat1.shape == mat2.shape, f"{mat1.shape=} {mat2.shape=}"
        return np.sum((mat1 - mat2) ** 2)

    @staticmethod
    def ncc(mat1, mat2):
        assert mat1.shape == mat2.shape
        mean1, mean2 = np.mean(mat1), np.mean(mat2)
        deviation1, deviation2 = mat1 - mean1, mat2 - mean2
        return np.sum(deviation1 * deviation2) / np.sqrt(np.sum(deviation1 ** 2) * np.sum(deviation2 ** 2))


if __name__ == "__main__":
    # books pair path
    # img1_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_1.jpeg"
    # img2_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_2.jpeg"

    # fountain pair path
    # img1_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_1.jpg"
    # img2_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_2.jpg"

    # building pair path
    img1_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/building_1.jpg"
    img2_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/building_2.jpg"

    # building pair path
    # img1_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/garden_1.jpg"
    # img2_path = "/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/garden_2.jpg"

    hcm = HarrisCornerMatcher(img1_path, img2_path, sigma=1.2, metric="ncc")

    hcm.get_correspondences()
    hcm.display()



