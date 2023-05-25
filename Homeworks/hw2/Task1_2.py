
import numpy as np
from skimage.io import imread

from Task1_1 import get_homography, warp, disp_with_lines
from Points import *


if __name__ == "__main__":
    # read images into numpy array
    card_a = imread("hw2images/card1.jpeg")
    card_c = imread("hw2images/card3.jpeg")

    # read own images
    box_a = imread("hw2images/box1.jpg")
    box_c = imread("hw2images/box3.jpg")

    # calculate 2 chained homographies
    # homog_a2b = get_homography(card1_points, card2_points)
    # homog_b2c = get_homography(card2_points, card3_points)

    homog_a2b = get_homography(box1_points, box2_points)
    homog_b2c = get_homography(box2_points, box3_points)

    # get homography from card a to card c
    homog_a2c = np.matmul(homog_b2c, homog_a2b)

    # warp card a to card c
    # warp(card_a, card_c, np.linalg.pinv(homog_a2c), card3_points)
    warp(box_a, box_c, np.linalg.pinv(homog_a2c), box3_points)

    # display
    # disp_with_lines(card_a, card_c, card1_points, card3_points)
    disp_with_lines(box_a, box_c, box1_points, box3_points)
