import numpy as np

from RANSAC import RANSAC


class Matcher:
    def __init__(self, metric='ncc'):
        self.metric = metric
        self.filter = RANSAC()
        self.inliers = None
        self.ssd_thr = 0.5  # hard-coded threshold for ssd distance between 2 unity normalized vectors
        self.ncc_thr = 0.5  # hard-coded threshold for ncc distance between 2 unity normalized vectors

    def get_matched_points(self, kpts1, kpts2, desc1, desc2, do_ransac=True):
        """
        Brute-force approach to calculate the matching keypoints by their descriptors
        :param kpts1:
        :param kpts2:
        :param desc1:
        :param desc2:
        :return:
        """
        # transpose descriptor matrix to align with shape of keypoints matrix
        desc1 = desc1.T
        desc2 = desc2.T
        print(f"{kpts1.shape=} {desc1.shape=} {kpts2.shape=} {desc2.shape=}")
        assert kpts1.shape[0] == desc1.shape[0], "Keypoint and descriptor number do not match for image 1!"
        assert kpts2.shape[0] == desc2.shape[0], "Keypoint and descriptor number do not match for image 2!"
        assert kpts1.shape[0] > 0, "Should contain at least one keypoint for image 1!"
        assert kpts2.shape[0] > 0, "Should contain at least one keypoint for image 2!"

        mkpts1, mkpts2 = [], []
        for i in range(len(kpts1)):
            kpt1 = kpts1[i]
            descriptor1 = desc1[i]
            min_dist = np.inf
            nearest_kpt2 = None
            for j in range(len(kpts2)):
                kpt2 = kpts2[j]
                descriptor2 = desc2[j]
                dist = self.ssd(descriptor1, descriptor2) if self.metric == 'ssd' \
                    else self.ncc(descriptor1, descriptor2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_kpt2 = kpt2
            assert nearest_kpt2 is not None, "Nearest keypoint2 is None!"

            # print(f"Best ssd dist {min_dist}")
            do_keep = min_dist < self.ssd_thr if self.metric == 'ssd' else min_dist < self.ncc_thr
            if do_keep:
                mkpts1.append(kpt1)
                mkpts2.append(nearest_kpt2)

        mkpts1, mkpts2 = np.asarray(mkpts1), np.asarray(mkpts2)

        if do_ransac:
            inlier_indices = self.filter.get_inlier_ids(mkpts1, mkpts2)
            mkpts1, mkpts2 = mkpts1[inlier_indices], mkpts2[inlier_indices]
            self.inliers = (mkpts1, mkpts2)

        return mkpts1, mkpts2

    @staticmethod
    def ssd(vec1, vec2):
        """
        Sum of Squared Difference
        Get the squared norm of the difference vector
        :param vec1:
        :param vec2:
        :return: scalar distance
        """
        assert vec1.shape == vec2.shape, f"{vec1.shape=} {vec2.shape=}"
        return np.sum((vec1 - vec2) ** 2)

    @staticmethod
    def ncc(vec1, vec2):
        """
        Normalized Cross Correlation
        Get cos(theta) of two center shifted vectors, in range [-1, 1], thus larger ncc means nearer distance
        so use 1 minus the ncc to get the distance aligned with ssd, smaller is nearer
        :param vec1:
        :param vec2:
        :return: scalar distance
        """
        assert vec1.shape == vec2.shape
        mean1, mean2 = np.mean(vec1), np.mean(vec2)
        deviation1, deviation2 = vec1 - mean1, vec2 - mean2
        return 1 - np.sum(deviation1 * deviation2) / np.sqrt(np.sum(deviation1 ** 2) * np.sum(deviation2 ** 2))
