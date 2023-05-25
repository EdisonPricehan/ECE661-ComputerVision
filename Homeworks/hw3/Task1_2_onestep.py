
import numpy as np
from skimage.io import imread, imsave

from Points import *
from Task1_1 import warp_with_homog, disp_with_lines
from Task1_2_twostep import get_line


def get_affine_removing_homography_by_squares(square1, square2):
    A = []
    b = []

    for i in range(2):
        point1 = square1[i]
        point2 = square1[(i + 1) % 4]
        point3 = square1[(i + 2) % 4]
        l = get_line(*point1, *point2)
        m = get_line(*point2, *point3)

        A.append([l[0] * m[0],
                  l[1] * m[0] + l[0] * m[1],
                  l[1] * m[1],
                  l[2] * m[0] + l[0] * m[2],
                  l[2] * m[1] + l[1] * m[2]])
        b.append(-l[2] * m[2])

    for i in range(3):
        point1 = square2[i]
        point2 = square2[(i + 1) % 4]
        point3 = square2[(i + 2) % 4]
        l = get_line(*point1, *point2)
        m = get_line(*point2, *point3)

        A.append([l[0] * m[0],
                  l[1] * m[0] + l[0] * m[1],
                  l[1] * m[1],
                  l[2] * m[0] + l[0] * m[2],
                  l[2] * m[1] + l[1] * m[2]])
        b.append(-l[2] * m[2])

    print(f"{A=}")
    print(f"{b=}")

    a, b, c, d, e = np.linalg.pinv(np.asarray(A)) @ np.asarray(b)
    C_dual = np.array([[a, b, d],
                       [b, c, e],
                       [d, e, 1]])
    print(f"{C_dual=}")

    # find 2x2 principal submatrix A in homography matrix H given C_dual[:2, :2] = AA^T
    eigval, eigvec = np.linalg.eig(C_dual[:2, :2])

    for i in range(len(eigval)):
        if eigval[i] < 0:
            eigval[i] = -eigval[i]

    print(f"{eigval=}")
    print(f"{eigvec=}")
    A = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
    print(f"{A=}")

    # get vector v given Av=[d, e]^T
    v = np.linalg.pinv(A) @ np.array([d, e])
    print(f"{v=}")

    # construct homography matrix H that transforms degenerate dual conic in physical plane to dual conic in image plane
    H = np.eye(3)
    H[:2, :2] = A
    H[2, :2] = v

    # scale the left 2 columns to control image size
    H[:, :2] = H[:, :2] / 1000

    print(f"{H=}")

    return H


def warp_building_1step():
    building = imread("hw3images/building.jpg")
    # H = get_affine_removing_homography_by_squares(building_points, building_points1)
    # dst = warp_with_homog(building, np.linalg.pinv(H))
    # imsave("building_affine_removed_1step.jpg", dst)

    building_2step_removed = imread("hw3images/results/1step/building_affine_removed_1step.jpg")

    disp_with_lines(building, building_2step_removed, building_points, building_points1)


def warp_nighthawks_1step():
    nighthawks = imread("hw3images/nighthawks.jpg")
    # H = get_affine_removing_homography_by_squares(nighthawks_points, nighthawks_points1)
    # dst = warp_with_homog(nighthawks, np.linalg.pinv(H))
    # imsave("nighthawks_affine_removed_1step.jpg", dst)

    nighthawks_2step_removed = imread("hw3images/results/1step/nighthawks_affine_removed_1step.jpg")

    disp_with_lines(nighthawks, nighthawks_2step_removed, nighthawks_points, nighthawks_points1)


def warp_card_1step():
    card = imread("hw3images/card.jpeg")
    H = get_affine_removing_homography_by_squares(card_points, card_points1)
    dst = warp_with_homog(card, np.linalg.pinv(H))
    imsave("card_affine_removed_1step.jpg", dst)
    disp_with_lines(card, dst, card_points, card_points1)


def warp_facade_1step():
    facade = imread("hw3images/facade.jpg")
    H = get_affine_removing_homography_by_squares(facade_points, facade_points1)
    dst = warp_with_homog(facade, np.linalg.pinv(H))
    imsave("facade_affine_removed_1step.jpg", dst)
    disp_with_lines(facade, dst, facade_points, facade_points1)


if __name__ == "__main__":
    # read given building image and rectify it by removing projective and affine parts in 1 step
    # warp_building_1step()

    # read given nighthawks image and rectify it by removing projective and affine parts in 1 step
    # warp_nighthawks_1step()

    # card
    # warp_card_1step()

    # facade
    warp_facade_1step()








