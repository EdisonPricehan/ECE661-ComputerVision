import numpy as np
import cv2
import matplotlib.pyplot as plt


class RectMatcher:
    def __init__(self, img_left_rect_path: str, img_right_rect_path: str):
        self.img_left_rect_rgb = cv2.cvtColor(cv2.imread(img_left_rect_path), cv2.COLOR_BGR2RGB)
        self.img_right_rect_rgb = cv2.cvtColor(cv2.imread(img_right_rect_path), cv2.COLOR_BGR2RGB)
        self.img_left_rect = cv2.cvtColor(self.img_left_rect_rgb, cv2.COLOR_RGB2GRAY)
        self.img_right_rect = cv2.cvtColor(self.img_right_rect_rgb, cv2.COLOR_RGB2GRAY)

        self.edges_left = self.extract_edges(self.img_left_rect)
        self.edges_right = self.extract_edges(self.img_right_rect)

        self.kpts_left, self.kpts_right = [], []

    @staticmethod
    def extract_edges(img):
        kernel_size = 5
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        low_threshold = 100  # Hysteresis minVal gradient magnitude below which is sure non-edge
        high_threshold = 300  # Hysteresis maxVal gradient magnitude above which is sure edge
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(5_000)
        return edges

    @staticmethod
    def descriptor(bin_img, window_size, point):
        padded_img = np.pad(bin_img, (window_size // 2, ))
        desp = padded_img[point[0]: point[0] + window_size, point[1]: point[1] + window_size].flatten()
        assert len(desp) == window_size * window_size, "Wrong length of descriptor vector!"
        return desp

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
        assert vec1.shape == vec2.shape, f"{vec1.shape=} {vec2.shape=}"
        mean1, mean2 = np.mean(vec1), np.mean(vec2)
        deviation1, deviation2 = vec1 - mean1, vec2 - mean2
        return 1 - np.sum(deviation1 * deviation2) / np.sqrt(np.sum(deviation1 ** 2) * np.sum(deviation2 ** 2))

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

    def match(self, window_size: int = 13, search_dist: int = 45, ignore_margin: int = 1, ncc_thr: float = 0.1):
        print(f"Start matching ...")

        hl, wl = self.edges_left.shape
        hr, wr = self.edges_right.shape
        # assert hl == hr, "Rectified images are not of the same height!"
        for row in range(hl):
            if row >= hr:  # when left image is higher than right image, no need to proceed after left height
                break
            interested_cols_left = np.where(self.edges_left[row] > 0)[0]
            # print(f"{interested_cols_left=}")
            if len(interested_cols_left) <= 2 * ignore_margin:  # ignore warped borders
                continue
            cur_col_right = None  # current rightmost column right
            for col_left in interested_cols_left[ignore_margin:-ignore_margin:1]:  # traver from left to right
                interested_cols_right = range(max(0, col_left - search_dist), min(col_left + 1, wr))
                valid_cols_right = [col_right for col_right in interested_cols_right if self.edges_right[row, col_right] > 0]
                if len(valid_cols_right) <= 2 * ignore_margin:  # ignore warped borders
                    continue
                # print(f"{valid_cols_right=}")

                desp_left = self.descriptor(self.img_left_rect, window_size, (row, col_left))
                min_ncc = np.inf
                best_col_right = None
                for valid_col_right in valid_cols_right[-ignore_margin:ignore_margin-1:-1]:  # traverse from right to left
                    if cur_col_right is not None and cur_col_right >= valid_col_right:
                        break
                    desp_right = self.descriptor(self.img_right_rect, window_size, (row, valid_col_right))
                    ncc_dist = self.ncc(desp_left, desp_right)
                    # ncc_dist = self.ssd(desp_left, desp_right)
                    if ncc_dist < min_ncc and ncc_dist < ncc_thr:
                        min_ncc = ncc_dist
                        best_col_right = valid_col_right
                if best_col_right is not None:
                    cur_col_right = best_col_right
                    self.kpts_left.append([col_left, row])
                    self.kpts_right.append([best_col_right, row])

        print(f"Built {len(self.kpts_left)} matches!")
        self.kpts_left = np.asarray(self.kpts_left)
        self.kpts_right = np.asarray(self.kpts_right)
        # print(f"{self.kpts_left=}")
        # print(f"{self.kpts_right=}")

        self.plot_matches(self.img_left_rect_rgb, self.img_right_rect_rgb,
                          self.kpts_left, self.kpts_right, 'rect_matches.jpg')

    @staticmethod
    def plot_matches(img0, img1, kpts0, kpts1, plt_name):
        assert len(kpts0) == len(kpts1), "Keypoints number do not match!"
        ms = 1  # marker size
        lw = 0.2  # line width
        h0, w0, _ = img0.shape
        h1, w1, _ = img1.shape
        img = np.zeros((np.max([h0, h1]), w0 + w1, 3), dtype=np.uint8)
        img[:h0, :w0] = img0
        img[:h1, w0:] = img1

        plt.figure()
        plt.imshow(img)

        # plot all keypoints in red, lines in blue
        for [x0, y0], [x1, y1] in zip(kpts0, kpts1):
            plt.plot(x0, y0, 'r.', markersize=ms)
            plt.plot(x1 + w0, y1, 'g.', markersize=ms)
            plt.plot((x0, x1 + w0), (y0, y1), '--bx', linewidth=lw, markersize=lw)

        plt.axis('off')
        plt.savefig(plt_name, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Matches figure saved!")


if __name__ == '__main__':
    left_rect_img_path = 'LoopAndZhang/LoopAndZhang/rectified/waffle_left_rect.png'
    right_rect_img_path = 'LoopAndZhang/LoopAndZhang/rectified/waffle_right_rect.png'

    matcher = RectMatcher(left_rect_img_path, right_rect_img_path)
    matcher.match(window_size=21, search_dist=45)

