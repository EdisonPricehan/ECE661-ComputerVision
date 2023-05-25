import cv2
import numpy as np


class StereoMatching:
    def __init__(self, img_left_path: str, img_right_path: str, gt_left_path: str, gt_right_path: str,
                 M: tuple = (7, 7), dmax: int = 10, delta: int = 2):
        """
        Stereo matching by local context around each pixel
        :param img_left_path:
        :param img_right_path:
        :param gt_left_path:
        :param gt_right_path:
        :param M: window size (height, width)
        :param dmax: maximum disparity, i.e. max column difference
        :param delta: allowable disparity difference with groundtruth mask to be considered as correct
        """
        self.img_left = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
        self.img_right = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)
        print(f"Left image dim: {self.img_left.shape}")
        print(f"Right image dim: {self.img_right.shape}")
        assert self.img_left.shape == self.img_right.shape, "Image shape not match!"
        self.h, self.w = self.img_left.shape

        self.disp_left = np.zeros_like(self.img_left)
        self.gt_left = cv2.imread(gt_left_path, cv2.IMREAD_GRAYSCALE)
        self.gt_right = cv2.imread(gt_right_path, cv2.IMREAD_GRAYSCALE)
        self.correct_mask_left = np.zeros_like(self.gt_left)
        self.delta = delta

        self.M = M
        self.offset_x, self.offset_y = self.M[0] // 2, self.M[1] // 2
        self.dmax = dmax
        print(f"Window size: {self.M}, offset_x: {self.offset_x}, offset_y: {self.offset_y}, dmax: {self.dmax}")
        self.context_left, self.context_right = None, None
        self.disp_left = np.zeros_like(self.img_left)  # disparity map for left image
        self._create_context()

    def _create_context(self):
        print(f"Start creating context ...")
        # create 3d array to store window context for all pixels
        window_len = self.M[0] * self.M[1]
        img_left_3d = np.repeat(self.img_left[..., np.newaxis], window_len, axis=2)
        img_right_3d = np.repeat(self.img_right[..., np.newaxis], window_len, axis=2)
        ctx_left = np.zeros_like(img_left_3d)
        ctx_right = np.zeros_like(img_right_3d)
        print(f"{ctx_left.shape=}")

        # create padded 2d array to slide window
        pad_left = np.pad(self.img_left, ((self.offset_x, ), (self.offset_y, )))
        pad_right = np.pad(self.img_right, ((self.offset_x, ), (self.offset_y, )))
        print(f"{pad_left.shape=}")

        # slide window on 2d array and fill context into 3d array
        for i in range(self.h):
            for j in range(self.w):
                ctx_left[i, j, :] = pad_left[i: i + self.M[0], j: j + self.M[1]].flatten()
                ctx_right[i, j, :] = pad_right[i: i + self.M[0], j: j + self.M[1]].flatten()

        # convert 3d context array to 3d bitmatrix
        self.context_left = (ctx_left > img_left_3d).astype(int)
        self.context_right = (ctx_right > img_right_3d).astype(int)
        print(f"Context created!")

    def match(self):
        print(f"Start matching to find disparity ...")

        # local context distance metric
        def bitvec_dist(vec1, vec2):
            return np.sum(np.abs(vec1 - vec2))

        assert self.context_left is not None and self.context_right is not None

        # find best matches for every pixel in the left image from the right image along the same row within dmax
        for i in range(self.h):
            for j in range(self.w):
                min_disparity = np.inf
                best_d = 0
                for d in range(dmax + 1):
                    col_right = j - d
                    if col_right >= 0:
                        vec_left = self.context_left[i, j]
                        vec_right = self.context_right[i, col_right]
                        cur_dist = bitvec_dist(vec_left, vec_right)
                        if cur_dist < min_disparity:
                            min_disparity = cur_dist
                            best_d = d
                assert min_disparity != np.inf
                self.disp_left[i, j] = best_d

        disp_left = self.disp_left
        left_disparity_path = 'Task3Images/Task3Images/results/left_disparity_' + str(self.M[0]) + 'x' + str(self.M[1]) + '.jpg'
        cv2.imwrite(left_disparity_path, disp_left)
        # cv2.imshow('Left disparity', disp_left)
        # cv2.waitKey(100000)

        # evaluate the disparity mask
        gt_left = (self.gt_left.astype(float) / 4).astype(np.uint8)
        print(f"GT dmax is {np.max(gt_left)}")
        # cv2.imshow('GT left', gt_left)
        # cv2.waitKey(10000)

        self.correct_mask_left[np.where(np.abs(self.disp_left - gt_left) <= self.delta)] = 255
        accuracy = int(np.sum(self.correct_mask_left == 255) / \
                   (self.correct_mask_left.shape[0] * self.correct_mask_left.shape[1]) * 100)
        print(f"Percent {accuracy=}")
        left_mask_path = 'Task3Images/Task3Images/results/left_mask_' + str(self.M[0]) + 'x' + str(
            self.M[1]) + '_acc' + str(accuracy) + '.jpg'
        cv2.imwrite(left_mask_path, self.correct_mask_left)
        # cv2.imshow('Left correct mask', self.correct_mask_left)
        # cv2.waitKey(100000)


if __name__ == '__main__':
    img_left_path = 'Task3Images/Task3Images/im2.png'
    img_right_path = 'Task3Images/Task3Images/im6.png'
    disparity_left_path = 'Task3Images/Task3Images/disp2.png'
    disparity_right_path = 'Task3Images/Task3Images/disp6.png'

    window = (5, 5)  # (7, 7)
    dmax = 60
    stereo_matcher = StereoMatching(img_left_path, img_right_path, disparity_left_path, disparity_right_path,
                                    M=window, dmax=dmax, delta=2)
    stereo_matcher.match()




