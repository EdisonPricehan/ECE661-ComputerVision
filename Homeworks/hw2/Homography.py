import numpy as np


class Homography:
    def __init__(self):
        self.homog = np.eye(3)  # homography matrix, identity matrix by default
        self.A = []  # matrix to store A in Ax=b
        self.b = []  # vector to store b in Ax=b

    def add_point_pair(self, x1: int, y1: int, x2: int, y2: int):
        """
        Homography is from (x1, y1) to (x2, y2)
        :param x1: x coordinate of pixel 1
        :param y1: y coordinate of pixel 1
        :param x2: x coordinate of pixel 2
        :param y2: y coordinate of pixel 2
        :return:
        """
        # append A and b by linear equation 1
        self.A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        self.b.append(x2)

        # append A and b by linear equation 2
        self.A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        self.b.append(y2)

    def get_homog(self) -> np.array:
        """
        Solve homography matrix in Ax=b form
        :return: homography matrix
        """
        homog = np.matmul(np.linalg.pinv(self.A), np.asarray(self.b))
        homog = np.append(homog, 1)
        self.homog = homog.reshape((3, 3))
        # print(f"{self.homog=}")
        return self.homog

    def get_affine(self) -> np.array:
        """
        Get affine matrix from projective matrix
        :return: affine matrix
        """
        h1, h2, h3, h4, h5, h6, h7, h8 = self.homog.flatten()[:-1]
        affine = np.array([[h1 - h3 * h7, h2 - h3 * h8, h3],
                           [h4 - h6 * h7, h5 - h6 * h8, h6],
                           [0, 0, 1]])
        return affine
