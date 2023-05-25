from skimage.transform import warp
from skimage.io import imshow, imread
import numpy as np
import matplotlib.pyplot as plt

from Points import nighthawks_line_pairs
from Task1_2_twostep import warp_nighthawks_2steps
from Task1_2_onestep import get_affine_removing_homography


if __name__ == "__main__":
    # H, nighthawks = warp_nighthawks_2steps()
    # nighthawks_warped = warp(nighthawks, np.linalg.pinv(H))

    nighthawks = imread("hw3images/nighthawks.jpg")
    H = get_affine_removing_homography(nighthawks_line_pairs)
    nighthawks_warped = warp(nighthawks, np.linalg.pinv(H))

    imshow(nighthawks_warped)
    plt.show()

