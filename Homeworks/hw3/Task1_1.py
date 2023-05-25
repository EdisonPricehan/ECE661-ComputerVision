
import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt

from Homography import Homography
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


def disp_with_lines(src_img, dst_img, src_points, src_points2=None, dst_points=None):
    """
    Draw dst_img and src_img in a row, with ROI lines in both images
    :param src_points2:
    :param src_img: source image in numpy.array
    :param dst_img: destination image in numpy.array
    :param src_points: list of ROI vertex points in src_img [[x, y], ...]
    :param dst_points: list of ROI vertex points in dst_img [[x, y], ...], or None
    :return:
    """
    # get display coordinates for ROI points of both src and dst images
    src_points_disp = []
    src_points_disp2 = []
    disp_line_pairs = False
    if len(np.asarray(src_points).shape) > 2:
        disp_line_pairs = True
        for line_vertical, line_horizontal in src_points:
            for point in line_vertical:
                src_points_disp.append(get_disp_coord(*point))
            for point in line_horizontal:
                src_points_disp.append(get_disp_coord(*point))
    else:
        for p in src_points:
            src_points_disp.append(get_disp_coord(*p))
        src_points_disp.append(get_disp_coord(*src_points[0]))

        if src_points2:
            for p in src_points2:
                src_points_disp2.append(get_disp_coord(*p))
            src_points_disp2.append(get_disp_coord(*src_points2[0]))

    dst_points_disp = []
    if dst_points:
        for p in dst_points:
            dst_points_disp.append(get_disp_coord(*p))
        dst_points_disp.append(get_disp_coord(*dst_points[0]))

    # display
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dst_img)
    ax[0].plot(*(list(zip(*dst_points_disp))), marker='o', linewidth=1, color='g')
    ax[1].imshow(src_img)
    if disp_line_pairs:
        assert len(src_points_disp) % 2 == 0
        cmap = plt.get_cmap('jet_r')
        n_lines = int(len(src_points_disp) / 2)
        for i in range(n_lines):
            color = cmap(float(int(i / 2)) / n_lines * 2)  # each line pair has a unique color
            ax[1].plot(*list(zip(*src_points_disp[i * 2: i * 2 + 2])), marker='*', linewidth=1, color=color)
    else:
        ax[1].plot(*(list(zip(*src_points_disp))), marker='o', linewidth=1, color='g')
        if src_points2:
            ax[1].plot(*(list(zip(*src_points_disp2))), marker='o', linewidth=1, color='r')

    plt.show()


def warp_with_homog(src_img, H):
    """
    Warp src_img by homography matrix H
    :param src_img: source distorted image in numpy.array
    :param H: homography matrix which transforms from src to dst
    :return: rectified undistorted (no projective, no affine) image in numpy.array
    """
    src_h, src_w, _ = src_img.shape
    print(f"Src image height {src_h}, width {src_w}")
    dst_bottom_right = H @ np.array([src_h - 1, src_w - 1, 1])
    dst_bottom_right = dst_bottom_right / dst_bottom_right[-1]
    print(f"{dst_bottom_right=}")

    dst_top_right = H @ np.array([0, src_w - 1, 1])
    dst_top_right = dst_top_right / dst_top_right[-1]
    print(f"{dst_top_right=}")

    dst_bottom_left = H @ np.array([src_h - 1, 0, 1])
    dst_bottom_left = dst_bottom_left / dst_bottom_left[-1]
    print(f"{dst_bottom_left=}")


    dst_h, dst_w = dst_bottom_right[:2].astype(int)
    dst_img = np.zeros((dst_h + 1, dst_w + 1, 3)).astype(int)
    dst_h, dst_w, _ = dst_img.shape
    print(f"Dst image height {dst_h}, width {dst_w}")
    H_inv = np.linalg.pinv(H)

    for i in range(dst_h):
        for j in range(dst_w):
            p_src = H_inv @ np.array([i, j, 1])
            p_src = (p_src / p_src[-1]).astype(int)
            # avoid index out of range error
            if 0 <= p_src[0] < src_h and 0 <= p_src[1] < src_w:
                dst_img[i][j] = src_img[p_src[0]][p_src[1]]

    return dst_img


def warp_with_points(src_img, src_points, dst_points):
    """
    Warp source image with point correspondences
    Homography is calculated based on correspondences from src_points to dst_points
    :param src_img: source image in numpy.array
    :param src_points: list of ROI vertex points in src_img [[x, y], ...]
    :param dst_points: list of ROI vertex points in dst_img [[x, y], ...]
    :return:
    """
    # get homography from src to dst (points start from top-left corner, counter-clockwise)
    H = get_homography(src_points, dst_points)
    print(f"{H=}")

    # remove pure projective and pure affine transforms by homography
    undistorted = warp_with_homog(src_img, H)

    # display images with corresponding ROI boundaries
    disp_with_lines(src_img, undistorted, src_points, dst_points=dst_points)


if __name__ == "__main__":
    # read given building image and rectify it by removing projective and affine parts
    # building = imread("hw3images/building.jpg")
    # warp_with_points(building, building_points, building_rectified_points)

    # read given nighthawks image and rectify it by removing projective and affine parts
    # nighthawks = imread("hw3images/nighthawks.jpg")
    # warp_with_points(nighthawks, nighthawks_points, nighthawks_rectified_points)

    # card image
    # card = imread("hw3images/card.jpeg")
    # warp_with_points(card, card_points, card_rectified_points)

    # facade image
    facade = imread("hw3images/facade.jpg")
    warp_with_points(facade, facade_points, facade_rectified_points)





