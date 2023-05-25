
import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt

from Homography import Homography
from Polygon import Polygon
from Points import *


def get_disp_coord(x, y):
    """
    Transform numpy array coordinate to matplotlib (image) coordinate
    :param x: positive x coordinate along height axis (downward)
    :param y: positive y coordinate along width axis (to the right)
    :return: (x', y') coordinate in image coordinate (positive x axis to the right, positive y axis downward)
    """
    return y, x


def get_homography(src_points, dst_points):
    """
    Get homography matrix given corresponding point pairs
    :param src_points: [[x1, y1], ...]
    :param dst_points: [[x1, y1], ...], same size as src_points
    :return: 3x3 matrix in numpy.array
    """
    homog = Homography()
    for src, dst in zip(src_points, dst_points):
        homog.add_point_pair(*src, *dst)
    H = homog.get_homog()
    return H


def get_affine(src_points, dst_points):
    """
        Get affine matrix given corresponding point pairs
        :param src_points: [[x1, y1], ...]
        :param dst_points: [[x1, y1], ...], same size as src_points
        :return: 3x3 matrix in numpy.array
    """
    homog = Homography()
    for src, dst in zip(src_points, dst_points):
        homog.add_point_pair(*src, *dst)
    homog.get_homog()
    Aff = homog.get_affine()
    return Aff


def warp(src_img, dst_img, H_inv, dst_points):
    """
    Warp ROI pixels in dst_img based on inverse homography and src_img
    :param src_img: source image in numpy.array
    :param dst_img: destination image in numpy.array
    :param H_inv: inverse of the homography which transforms from src to dst
    :param dst_points: list of points in dst_img that defines ROI, [[x, y], ...]
    :return:
    """
    # construct polygon for destination image ROI for inverse homography
    # only care about pixels within the polygon (including edges)
    poly_dst = Polygon()
    for dst in dst_points:
        poly_dst.add_point(*dst)
    assert poly_dst.valid()
    xmin, xmax, ymin, ymax = poly_dst.minx, poly_dst.maxx, poly_dst.miny, poly_dst.maxy

    # transform interested pixels location from src to dst by inverse homography to find the desired rgb value
    # then set that back to src pixel location
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            if poly_dst.contain(i, j):
                p_dst = [i, j, 1]
                p_src = np.matmul(H_inv, np.asarray(p_dst))
                p_src = (p_src / p_src[-1]).astype(int)
                # avoid index out of range error
                if 0 <= p_src[0] < src_img.shape[0] and 0 <= p_src[1] < src_img.shape[1]:
                    dst_img[i][j] = src_img[p_src[0]][p_src[1]]


def disp_with_lines(src_img, dst_img, src_points, dst_points):
    """
    Draw dst_img and src_img in a row, with ROI lines in both images
    :param src_img: source image in numpy.array
    :param dst_img: destination image in numpy.array
    :param src_points: list of ROI vertex points in src_img [[x, y], ...]
    :param dst_points: list of ROI vertex points in dst_img [[x, y], ...]
    :return:
    """
    # get display coordinates for ROI points of both src and dst images
    src_points_disp = []
    for p in src_points:
        src_points_disp.append(get_disp_coord(*p))
    src_points_disp.append(get_disp_coord(*src_points[0]))
    dst_points_disp = []
    for p in dst_points:
        dst_points_disp.append(get_disp_coord(*p))
    dst_points_disp.append(get_disp_coord(*dst_points[0]))

    # display
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dst_img)
    ax[0].plot(*(list(zip(*dst_points_disp))), marker='o', linewidth=2, color='g')
    ax[1].imshow(src_img)
    ax[1].plot(*(list(zip(*src_points_disp))), marker='o', linewidth=2, color='g')
    plt.show()


def warp_with_points(src_img, dst_img, src_points, dst_points):
    """
    Transform (warp) the ROI defined by src_points in src_img to ROI defined by dst_points in dst_img
    Homography is calculated based on correspondences from src_points to dst_points
    :param src_img: source image in numpy.array
    :param dst_img: destination image in numpy.array
    :param src_points: list of ROI vertex points in src_img [[x, y], ...]
    :param dst_points: list of ROI vertex points in dst_img [[x, y], ...]
    :return:
    """
    # get homography from src to dst (points start from top-left corner, counter-clockwise, i.e. PQRS)
    H = get_homography(src_points, dst_points)
    print(H)

    # warp src to dst by inverse homography
    warp(src_img, dst_img, np.linalg.pinv(H), dst_points)

    # display images with corresponding ROI boundaries
    disp_with_lines(src_img, dst_img, src_points, dst_points)


if __name__ == "__main__":
    # read given images
    car = imread("hw2images/car.jpg")
    h_src, w_src, _ = car.shape
    print(f"Car width: {w_src}, height: {h_src}")
    card1 = imread("hw2images/card1.jpeg")
    h_dst, w_dst, _ = card1.shape
    print(f"Card1 width: {w_dst}, height: {h_dst}")

    # read own images
    dog = imread("hw2images/dog.jpg")
    h_src, w_src, _ = dog.shape
    print(f"Dog width: {w_src}, height: {h_src}")
    box3 = imread("hw2images/box3.jpg")
    h_dst, w_dst, _ = box3.shape
    print(f"Box1 width: {w_dst}, height: {h_dst}")

    warp_with_points(dog, box3, dog_points, box3_points)
