
import numpy as np
from skimage.io import imread

from Task1_1 import get_affine, warp, disp_with_lines
from Points import *


def warp_affine_with_points(src_img, dst_img, src_points, dst_points):
    """
    Affine transform (warp) the ROI defined by src_points in src_img to ROI defined by dst_points in dst_img
    Homography is calculated based on correspondences from src_points to dst_points
    :param src_img: source image in numpy.array
    :param dst_img: destination image in numpy.array
    :param src_points: list of ROI vertex points in src_img [[x, y], ...]
    :param dst_points: list of ROI vertex points in dst_img [[x, y], ...]
    :return:
    """
    # get homography from car to card a
    Aff = get_affine(src_points, dst_points)
    print(Aff)

    # warp car to card a
    warp(src_img, dst_img, np.linalg.pinv(Aff), dst_points)

    # display
    disp_with_lines(src_img, dst_img, src_points, dst_points)


if __name__ == "__main__":
    # read all images
    # car = imread("hw2images/car.jpg")
    # h, w, _ = car.shape
    # print(f"Car width: {w}, height: {h}")
    # card_a = imread("hw2images/card1.jpeg")
    # card_b = imread("hw2images/card2.jpeg")
    # card_c = imread("hw2images/card3.jpeg")

    # read own images
    car = imread("hw2images/dog.jpg")
    h, w, _ = car.shape
    print(f"Car width: {w}, height: {h}")
    card_a = imread("hw2images/box1.jpg")
    card_b = imread("hw2images/box2.jpg")
    card_c = imread("hw2images/box3.jpg")

    # use whole car image
    car_points = [[0, 0], [h - 1, 0], [h - 1, w - 1], [0, w - 1]]

    # warp from car to card1 with only affine transformation
    # warp_affine_with_points(car, card_a, car_points, card1_points)
    warp_affine_with_points(car, card_a, car_points, box1_points)

    # warp from car to card1 with only affine transformation
    # warp_affine_with_points(car, card_b, car_points, card2_points)
    warp_affine_with_points(car, card_b, car_points, box2_points)

    # warp from car to card1 with only affine transformation
    warp_affine_with_points(car, card_c, car_points, box3_points)



