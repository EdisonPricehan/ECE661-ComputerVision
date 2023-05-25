import numpy as np


class Homography:
    def __init__(self, inhomogeneous=True):
        """
        Class that outputs 3x3 homography matrix by solving overdetermined linear least square homogeneous equations
        """
        self.inhomogeneous = inhomogeneous  # solve linear least square problem in homogeneous or inhomogeneous manner
        self.H = np.eye(3)  # homography matrix, identity matrix by default
        self.A = []  # matrix to store A in Ax=0

    def add_point_pair(self, x1: int, y1: int, x2: int, y2: int):
        """
        Homography is from (x1, y1) to (x2, y2)
        :param x1: x coordinate of pixel 1
        :param y1: y coordinate of pixel 1
        :param x2: x coordinate of pixel 2
        :param y2: y coordinate of pixel 2
        :return:
        """
        # append A by 2 independent homogeneous equations given 1 correspondence
        self.A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
        self.A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, -y2])

    def get_homography(self, points1: np.array, points2: np.array) -> np.array:
        assert points1.shape[1] == 2, f'Points1 have wrong cols {points1.shape[1]}!'
        assert points2.shape[1] == 2, f'Points2 have wrong cols {points2.shape[1]}!'
        assert points1.shape[0] == points2.shape[0], 'Points1 and points2 shapes do not match!'

        # init matrix A in Ax=0 linear least square form
        num_corr = points1.shape[0]  # number of correspondences
        self.A = np.zeros((2 * num_corr, 9)).astype(float)  # 2N x 9 matrix

        # make homogeneous representation of corresponding points
        points1_homog = np.vstack((points1.T, [1] * num_corr))  # 3 x N

        # fill matrix A
        self.A[::2, :3] = points1_homog.T
        self.A[1::2, 3:6] = points1_homog.T
        self.A[::2, 6:] = (points1_homog * (-points2[:, 0])).T
        self.A[1::2, 6:] = (points1_homog * (-points2[:, 1])).T

        return self.get_H()

    def get_H(self) -> np.array:
        """
        Solve homography matrix
        :return: homography matrix
        """
        A = np.asarray(self.A)
        assert A.shape[0] >= 8, "Need at least 4 correspondences to solve homography matrix!"
        assert A.shape[1] == 9, "Wrong columns in matrix A!"

        if self.inhomogeneous:
            # Ax=b inhomogeneous form
            b = -A[:, -1]
            A = A[:, :-1]
            h = np.linalg.pinv(A) @ b  # first 8 elements of matrix H
            h = np.append(h, 1)
        else:
            # Ax=0 homogeneous form
            U, S, VT = np.linalg.svd(A.T @ A)
            h = VT[-1, :]  # last row is the eigen vector to the smallest eigenvalue
        self.H = h.reshape((3, 3))

        # scale H so that last element is 1, but seems affecting very little
        if self.H[-1, -1] != 0:
            self.H = self.H / self.H[-1, -1]

        # print(f"{self.H=}")
        return self.H

    @staticmethod
    def transform(homog, points, round_int: bool = True):
        """
        Transform points by some homography matrix
        :param homog: 3x3 homography matrix
        :param points: Nx2 domain points
        :param round_int: round the transformed points to integers
        :return: Nx2 range points
        """
        homog = np.asarray(homog)
        assert homog.shape == (3, 3), "Homography matrix should be 3x3"

        points = np.asarray(points)
        assert points.shape[1] == 2, "Points only need x and y values"

        # represent points in homogeneous coordinate, 3xN
        points = np.vstack((points.T, [1] * points.shape[0]))

        # transform all homogeneous points, 3xN
        points_range = homog @ points

        # convert homogeneous coordinate to physical coordinate
        points_range = points_range / points_range[-1, :]

        # convert representation to align with input domain points, Nx2
        points_range = points_range[:-1, :].T

        # round to integer if needed
        if round_int:
            points_range = points_range.astype(int)

        return points_range

