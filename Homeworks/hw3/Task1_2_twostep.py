
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale

from Points import *
from Task1_1 import warp_with_homog, disp_with_lines


def get_line(x1, y1, x2, y2):
    """

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    l = np.cross([x1, y1, 1], [x2, y2, 1])
    l = l / l[-1]
    return l


def get_point(l11, l12, l13, l21, l22, l23):
    """

    :param l11:
    :param l12:
    :param l13:
    :param l21:
    :param l22:
    :param l23:
    :return:
    """
    p = np.cross([l11, l12, l13], [l21, l22, l23])
    p = p / p[-1]
    return p


def get_vanishing_line(square_points):
    """
    Counter-clockwise 4 points in image plane that form a square in physical world plane
    :param square_points: [[x, y], ...]
    :return: numpy.array([l1, l2, l3])
    """
    p1, p2, p3, p4 = square_points[0], square_points[1], square_points[2], square_points[3]
    l14 = get_line(*p1, *p4)
    l23 = get_line(*p2, *p3)
    vp1 = get_point(*l14, *l23)  # vanishing point 1
    print(f"{vp1=}")
    l12 = get_line(*p1, *p2)
    l34 = get_line(*p3, *p4)
    vp2 = get_point(*l12, *l34)  # vanishing point 2
    print(f"{vp2=}")
    return get_line(*vp1[:2], *vp2[:2])


def get_projective_removing_homography(square_points):
    """
    Construct homography that maps vanishing line to infinity, thus removes the projective part
    :param square_points:
    :return: 3x3 matrix in numpy.array
    """
    # get vanishing line
    vl = get_vanishing_line(square_points)
    print(f"{vl=}")

    # construct homography matrix
    H = np.eye(3)
    H[2] = vl
    # print(np.linalg.inv(H).T @ vl)
    return H


def get_affine_removing_homography(square_points):
    """
    Counter-clockwise 4 points in image plane that form a square in physical world plane
    :param square_points: [[x, y], ...]
    :return: 3x3 matrix in numpy.array
    """
    p1, p2, p3, p4 = square_points

    # use 4 adjacent lines where each adjacent pair form 90 degrees angle in physical plane
    l12 = get_line(*p1, *p2)
    l23 = get_line(*p2, *p3)
    l34 = get_line(*p3, *p4)
    l14 = get_line(*p1, *p4)

    # get params for over-determined linear system (4 pairs of orthogonal lines of a square)
    param1 = [l12[0] * l23[0], l12[0] * l23[1] + l12[1] * l23[0], l12[1] * l23[1]]
    param2 = [l23[0] * l34[0], l23[0] * l34[1] + l23[1] * l34[0], l23[1] * l34[1]]
    param3 = [l34[0] * l14[0], l34[0] * l14[1] + l34[1] * l14[0], l34[1] * l14[1]]
    param4 = [l12[0] * l14[0], l12[0] * l14[1] + l12[1] * l14[0], l12[1] * l14[1]]

    # construct Ax=b problem to solve x, which has 2 entities in symmetric matrix S
    A = [
         param1[:2],
         param2[:2],
         param3[:2],
         param4[:2]
         ]
    b = [
         -param1[2],
         -param2[2],
         -param3[2],
         -param4[2]
         ]

    s11, s12 = np.linalg.pinv(np.asarray(A)) @ np.asarray(b)
    S = np.array([[s11, s12],
                  [s12, 1]])
    print(f"{S=}")

    # find A matrix given S = AA^T
    eigval, eigvec = np.linalg.eig(S)
    print(f"{eigval=}")
    print(f"{eigvec=}")

    # get A matrix by A = PDP^T
    A = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
    print(f"{A=}")

    # construct pure affine homography matrix
    H = np.eye(3)
    H[:2, :2] = A
    return H


def warp_building_2steps():
    # Step 1: get projective removed image and save
    # building = imread("hw3images/building.jpg")
    # H_remove_projective = get_projective_removing_homography(building_points)
    # print(f"{H_remove_projective=}")
    # dst = warp_with_homog(building, H_remove_projective)
    # imsave("building_projective_removed.jpg", dst)

    building_projective_removed = imread("hw3images/results/2steps/building_projective_removed.jpg")
    # disp_with_lines(building, building_projective_removed, building_points)

    # Step 2: get affine removed image and save
    # building_projective_removed = imread("building_projective_removed.jpg")
    # H_remove_affine = get_affine_removing_homography(building_projective_removed_square1)
    # dst = warp_with_homog(building_projective_removed, np.linalg.pinv(H_remove_affine))
    # imsave("building_affine_removed.jpg", dst)

    building_affine_removed = imread("hw3images/results/2steps/building_affine_removed.jpg")
    disp_with_lines(building_projective_removed, building_affine_removed, building_projective_removed_square1)


def warp_nighthawks_2steps():
    # Step 1: get projective removed image and save
    # nighthawks = imread("hw3images/nighthawks.jpg")
    # H_remove_projective = get_projective_removing_homography(nighthawks_points)
    # print(f"{H_remove_projective=}")
    # dst = warp_with_homog(nighthawks, H_remove_projective)
    # imsave("nighthawks_projective_removed.jpg", dst)

    nh_proj_rem = imread("hw3images/results/2steps/nighthawks_projective_removed.jpg")
    # disp_with_lines(nighthawks, nh_proj_rem, nighthawks_points)

    # Step 2: get affine removed image and save
    # nighthawks_projective_removed = imread("nighthawks_projective_removed.jpg")
    # H_remove_affine = get_affine_removing_homography(nighthawks_projective_removed_square2)
    # dst = warp_with_homog(nighthawks_projective_removed, np.linalg.pinv(H_remove_affine))
    # imsave("nighthawks_affine_removed.jpg", dst)

    nh_aff_rem = imread("hw3images/results/2steps/nighthawks_affine_removed.jpg")
    disp_with_lines(nh_proj_rem, nh_aff_rem, nighthawks_projective_removed_square1)


def warp_card_2steps():
    # Step 1: get projective removed image and save
    card = imread("hw3images/card.jpeg")
    # H_remove_projective = get_projective_removing_homography(card_points)
    # print(f"{H_remove_projective=}")
    # dst = warp_with_homog(card, H_remove_projective)
    # imsave("card_projective_removed.jpg", dst)

    card_proj_rem = imread("hw3images/results/2steps/card_projective_removed.jpg")
    disp_with_lines(card, card_proj_rem, card_points)

    # Step 2: get affine removed image and save
    # card_affine_removed = imread("card_projective_removed.jpg")
    # H_remove_affine = get_affine_removing_homography(card_projective_removed_square2)
    # H_remove_affine[:2, :2] = H_remove_affine[:2, :2] * 10
    # print(f"{H_remove_affine=}")
    # dst = warp_with_homog(card_affine_removed, np.linalg.pinv(H_remove_affine))
    # imsave("card_affine_removed.jpg", dst)

    card_aff_rem = imread("hw3images/results/2steps/card_affine_removed.jpg")
    disp_with_lines(card_proj_rem, card_aff_rem, card_projective_removed_square2)


def warp_facade_2step():
    # Step 1: get projective removed image and save
    # facade = imread("hw3images/facade.jpg")
    # H_remove_projective = get_projective_removing_homography(facade_points)
    # print(f"{H_remove_projective=}")
    # dst = warp_with_homog(facade, H_remove_projective)
    # imsave("facade_projective_removed.jpg", dst)

    facade_proj_rem = imread("hw3images/results/2steps/facade_projective_removed.jpg")
    # disp_with_lines(facade, facade_proj_rem, facade_points)

    # Step 2: get affine removed image and save
    # facade_affine_removed = imread("facade_projective_removed.jpg")
    # H_remove_affine = get_affine_removing_homography(facade_projective_removed_square2)
    # print(f"{H_remove_affine=}")
    # dst = warp_with_homog(facade_affine_removed, np.linalg.pinv(H_remove_affine))
    # imsave("facade_affine_removed.jpg", dst)

    facade_aff_rem = imread("hw3images/results/2steps/facade_affine_removed.jpg")
    disp_with_lines(facade_proj_rem, facade_aff_rem, facade_projective_removed_square2)


if __name__ == "__main__":
    # read given building image and rectify it by removing projective and affine parts in 2 steps
    # warp_building_2steps()

    # read given nighthawks image and rectify it by removing projective and affine parts in 2 steps
    # warp_nighthawks_2steps()

    # card
    # warp_card_2steps()

    # facade
    warp_facade_2step()


