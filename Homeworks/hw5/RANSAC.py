import numpy as np
import math

from Homography import Homography


class RANSAC:
    def __init__(self, n=9, delta=6, p=0.99, eps=0.5, max_N=10000):
        self.n = int(n)  # number of randomly chosen correspondences to estimate homography
        self.delta = delta  # pixel distance threshold to separate inliers and outliers
        self.p = p  # desired probability that there exists at least one trial free of outliers, affects N
        self.eps = eps  # approximate ratio of false correspondences (outliers), affects number of total trials N
        self.max_N = int(max_N)  # allowed maximum number of trials

        # params validation
        # assert 4 < self.n < 10, "n should be in (4, 10)"
        assert self.delta > 0, "delta should be a positive number"
        assert 0 < self.p < 1, "p should be in (0, 1)"
        assert 0 < self.eps < 1, "eps should be in (0, 1)"
        assert self.max_N > 0, "max_N should be a positive number"

    def set_n_total(self, n_total):
        """
        Set number of total correspondences and update relative parameters
        :param n_total:
        :return:
        """
        self.n_total = n_total  # total number of correspondences
        assert self.n_total >= self.n, "sampling number should not exceed total correspondence number"

        # suppress max_N by non-replacement sampling number
        self.max_N = min(self.max_N,
                         math.factorial(int(self.n_total)) / math.factorial(
                             int(self.n_total - self.n)) / math.factorial(int(self.n)))
        print(f"Max trials is set to {self.max_N}")

        # update total number of trials needed
        self.N = int(min(self.max_N, np.log(1 - self.p) / np.log(1 - np.power(1 - self.eps, self.n))))
        print(f"Total trials is set to {self.N}")

        # set minimum size of inliers set for it to be acceptable
        self.M = int((1 - self.p) * self.n_total)

    def get_inlier_ids(self, mkpts1, mkpts2):
        """
        Get indices of inliers in the best inliers set
        :param mkpts1: matched keypoints in domain image
        :param mkpts2: matched keypoints in range image
        :return: list of indices of inliers in the best inliers set
        """
        assert len(mkpts1) == len(mkpts2), "Matching keypoints size should be the same!"
        self.set_n_total(len(mkpts1))

        max_inliers_num = 0
        best_inliers_set = []
        for i in range(self.N):
            try:
                rnd_indices = np.random.choice(len(mkpts1), self.n, replace=False)
            except ValueError:
                print("Random choice options depleted, return.")
                break
            rnd_kpts1, rnd_kpts2 = mkpts1[rnd_indices], mkpts2[rnd_indices]
            homog = Homography()
            for p1, p2 in zip(rnd_kpts1, rnd_kpts2):
                homog.add_point_pair(*p1, *p2)
            H = homog.get_H()
            range_points = Homography.transform(H, mkpts1, round_int=True)

            tmp_inliers_set = []
            for index, (pn, po) in enumerate(zip(range_points, mkpts2)):
                dist = self.get_dist(*pn, *po)  # get distance between new reprojected point with original point
                # print(f"{dist=}")
                if dist <= self.delta:
                    tmp_inliers_set.append(index)
            tmp_inliers_num = len(tmp_inliers_set)

            # update current best inliers set if more inliers are within threshold
            if tmp_inliers_num >= self.M and tmp_inliers_num > max_inliers_num:
                max_inliers_num = tmp_inliers_num
                best_inliers_set = tmp_inliers_set

        print(f"Total correspondences {len(mkpts1)}, inliers {max_inliers_num}")
        return best_inliers_set

    def get_dist(self, x1: int, y1: int, x2: int, y2: int, metric: str = 'l2') -> float:
        """
        Get distance between 2 pixels in designated metric
        :param x1: x coordinate of pixel 1
        :param y1: y coordinate of pixel 1
        :param x2: x coordinate of pixel 2
        :param y2: y coordinate of pixel 2
        :param metric: l1 or l2 or linf
        :return: float distance
        """
        if metric == 'l1':  # manhattan distance
            return abs(x2 - x1) + abs(y2 - y1)
        elif metric == 'l2':  # euclidean distance
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        elif metric == 'linf':  # chessboard distance
            return max(abs(x2 - x1), abs(y2 - y1))
        else:
            print(f"Unrecognized metric!")
            return 0
