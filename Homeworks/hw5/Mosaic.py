
from superglue_ece661.superglue_ece661 import read_rgb_img, plot_keypoints, SuperGlue
from Homography import Homography
from Matcher import Matcher
from LM import LM

import os
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt


class Mosaic:
    def __init__(self, img_dir: str, out_dir: str, do_ransac: bool = True, do_lm: bool = True):
        # read all images in order from a directory
        self.img_dir = img_dir
        self.image_path_list = []
        self.image_list = []
        list_dir = [int(file.split(".")[0]) for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
        list_dir.sort()
        for img_name in list_dir:
            img_path = img_dir + '/' + str(img_name) + '.jpg'
            self.image_path_list.append(img_path)
            self.image_list.append(read_rgb_img(img_path))
        self.image_num = len(self.image_list)
        print(f"Read {self.image_num} images from {img_dir}.")
        print(f"{self.image_path_list}")
        assert self.image_num > 1, "Cannot mosaic only one image!"

        # image output directory
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # options
        self.do_ransac = do_ransac
        self.do_lm = do_lm

        # keypoints detected
        self.keypoints_list = []
        self.T_list = []  # translation matrix list for center shift

        # keypoints matcher
        self.matcher = Matcher(metric='ncc')  # can also be 'ssd'
        self.matches_list = []

        # image homographies
        self.homographies = []

        # by default anchor image is the middle image
        self.anchor_img_id = self.image_num // 2

    def update_keypoints(self, center_shift: bool = True):
        """
        Store keypoints info (location, score, descriptor) for all images using SuperPoint detector
        :param center_shift: whether center shift locations of detected keypoints based on image size
        :return:
        """
        # load existing keypoints list if exists
        kp_list_path = os.path.join(self.out_dir, 'keypoints_list')
        T_list_path = os.path.join(self.out_dir, 'T_list')

        if os.path.exists(kp_list_path) and os.path.exists(T_list_path):
            with open(kp_list_path, 'rb') as f:
                self.keypoints_list = pickle.load(f)
                print(f"keypoints_list is loaded from file!")
            with open(T_list_path, 'rb') as f:
                self.T_list = pickle.load(f)
                print(f"T_list is loaded from file!")
            return

        print(f"Start calculating keypoints for all images...")
        detector = SuperGlue.create(superglue_wts='outdoor', resize=[-1])  # do not resize any image

        for i in range(self.image_num):
            # detect and compute points using superpoint
            img_path = self.image_path_list[i]
            kp, score, descriptor = detector.detectAndCompute(img_path)
            # convert keypoints location from image convention (x right y down) to numpy convention (x down y right)
            kp[:, [1, 0]] = kp[:, [0, 1]]

            # center shift all keypoint locations, and store the translation matrix T for future homography estimation
            T = np.eye(3)
            if center_shift:
                img = self.image_list[i]
                h, w, _ = img.shape
                half_h, half_w = h // 2, w // 2
                T[0, -1] = -half_h
                T[1, -1] = -half_w
                kp = Homography.transform(T, kp, round_int=True)
            self.T_list.append(T)

            # store detected keypoints of each image
            self.keypoints_list.append((kp, score, descriptor))
            print(f"{len(kp)} keypoints detected in image {i}")

        # save keypoints list and translations list
        with open(kp_list_path, 'wb') as f:
            pickle.dump(self.keypoints_list, f)
            print(f"keypoints_list is saved!")
        with open(T_list_path, 'wb') as f:
            pickle.dump(self.T_list, f)
            print(f"T_list is saved!")

        print(f"Keypoints are extracted!")

    def update_matches(self, plot_matches: bool = False):
        """
        Sequentially calculate matches between adjacent images and store the matches
        :param plot_matches: plot keypoints and matches of current image pair
        :return:
        """
        assert len(self.keypoints_list) == self.image_num, "Keypoints number mismatch!"
        if self.image_num < 2:
            print("Cannot update matches with less than 2 sets of keypoints!")
            return

        mt_list_path = os.path.join(self.out_dir, 'matches_list')
        if os.path.exists(mt_list_path):
            f = open(mt_list_path, 'rb')
            self.matches_list = pickle.load(f)
            print(f"matches_list is loaded from file!")
            return

        print(f"Start updating matches for sequential image pairs...")

        for i in range(self.image_num - 1):
            kp0, _, descriptor0 = self.keypoints_list[i]
            kp1, _, descriptor1 = self.keypoints_list[i + 1]

            # compute matches using superpoint + superglue
            # mkpts0, mkpts1, matching_confidence = detector.match(self.image_path_list[i], self.image_path_list[i + 1])
            # from RANSAC import RANSAC
            # ransac = RANSAC()
            # inlier_indices = ransac.get_inlier_ids(mkpts0, mkpts1)
            # mkpts0, mkpts1 = mkpts0[inlier_indices], mkpts1[inlier_indices]

            # compute matches using brute force method, and filter with RANSAC
            mkpts0, mkpts1 = self.matcher.get_matched_points(kp0, kp1, descriptor0, descriptor1,
                                                             do_ransac=self.do_ransac)

            # np.save(str(i) + '_' + str(i + 1) + '_matches.npy', np.asarray([mkpts0, mkpts1]))

            self.matches_list.append([mkpts0, mkpts1])

            # plot keypoints and matches
            if plot_matches:
                # get image names
                img0, img1 = self.image_path_list[i], self.image_path_list[i + 1]
                img0_base = os.path.basename(img0).split('.')[0]
                img1_base = os.path.basename(img1).split('.')[0]
                # get target image path
                plt_name = os.path.join(self.out_dir,
                                        img0_base + '_and_' + img1_base +
                                        '_ncc_matches_ransac.png')
                # convert keypoints and matchings from center shifted coordinate to original image coordinate
                # i.e. x right y down
                T0_inv, T1_inv = np.linalg.pinv(self.T_list[i]), np.linalg.pinv(self.T_list[i + 1])
                kp0 = Homography.transform(T0_inv, kp0, round_int=True)
                kp1 = Homography.transform(T1_inv, kp1, round_int=True)
                mkpts0 = Homography.transform(T0_inv, mkpts0, round_int=True)
                mkpts1 = Homography.transform(T1_inv, mkpts1, round_int=True)
                kp0[:, [1, 0]] = kp0[:, [0, 1]]
                kp1[:, [1, 0]] = kp1[:, [0, 1]]
                mkpts0[:, [1, 0]] = mkpts0[:, [0, 1]]
                mkpts1[:, [1, 0]] = mkpts1[:, [0, 1]]
                plot_keypoints(img0, img1, kp0, kp1, mkpts0, mkpts1, plt_name)

        # save matches list
        f = open(mt_list_path, 'wb')
        pickle.dump(self.matches_list, f)
        print(f"matches_list is saved!")

        print("Matches are built!")

    def update_homographies(self):
        """
        Calculate homography matrices based on each set of filtered matching keypoints and store those matrices
        Transform homography matrices based on coordinate shift
        Chainly transform homography matrices so that they all map from non-anchor images towards the anchor image
        :return:
        """
        assert len(self.matches_list) > 0, "Update matches first before updating homograohies!"
        assert len(self.matches_list) == self.image_num - 1, "Wrong number of matches!"
        assert len(self.T_list) == self.image_num, "Wrong number of translation matrices!"

        print(f"Start updating homographies for all images w.r.t anchor image...")

        all_costs = []  # for plot cost of all matchings in one image

        for i, (mkpts0, mkpts1) in enumerate(self.matches_list):
            mkpts0_len, mkpts1_len = len(mkpts0), len(mkpts1)
            assert mkpts0_len == mkpts1_len, "Matched keypoints numbers do not equal!"
            print(f"{mkpts0_len} matching keypoints for image pair {i}-{i + 1}")

            homog_solver = Homography(inhomogeneous=False)
            for point0, point1 in zip(mkpts0, mkpts1):
                homog_solver.add_point_pair(*point0, *point1)
            H_tilde = homog_solver.get_H()

            # use LM algorithm to update this homography (keypoints in image center coordinate)
            if self.do_lm:
                levmar = LM(H_tilde, mkpts0, mkpts1)
                H_tilde, cost_list = levmar.refine(get_cost=True)
                if cost_list:
                    all_costs.append(cost_list)

            # H = T'^{-1} x H_tilde x T
            # H matrix maps from domain image coord to range image coord
            H = np.linalg.pinv(self.T_list[i + 1]) @ H_tilde @ self.T_list[i]
            if H[-1, -1] != 0:
                H = H / H[-1, -1]

            self.homographies.append(H)

        # plot costs for all matches, and save as csv file
        if all_costs:
            match_num = len(all_costs)
            for i in range(match_num):
                plt.figure()
                plt.plot(all_costs[i])
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.legend(['match_' + str(i) + '_' + str(i + 1)])
                plt.savefig(self.out_dir + '/lm_cost' + str(i) + '.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

            with open(self.out_dir + '/costs.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(all_costs)

        # update sequential homographies to those that transform from original images to the same anchor image
        assert len(self.homographies) == self.image_num - 1, "Wrong number of homographies!"
        self.homographies.insert(self.anchor_img_id, np.eye(3))
        if self.anchor_img_id > 0:  # no need to update homographies if anchor image is the first image
            # update all homographies to the left of the anchor image in right to left order
            for id in range(self.anchor_img_id - 1, -1, -1):
                self.homographies[id] = self.homographies[id + 1] @ self.homographies[id]
        if self.anchor_img_id < self.image_num - 1:
            # update all homographies to the right of the anchor image in left to right order
            for id in range(self.anchor_img_id + 1, self.image_num):
                self.homographies[id] = self.homographies[id - 1] @ np.linalg.pinv(self.homographies[id])

        print(f"{len(self.homographies)} homographies are built!")

    def stitch(self):
        """
        Stitch images in input order
        :return:
        """
        print("Start stitching images!")

        # offset_x, offset_y, h, w = self._compute_stitched_offset()
        new_img = self.get_empty_stitched_image()
        new_h, new_w, _ = new_img.shape
        offset_y = self.get_offset_y()
        print(f"Stitched image height {new_h}, width {new_w}, offset_y {offset_y}")

        # get anchor points in stitched image coordinate
        xns, yns = np.meshgrid(range(new_h), range(new_w))  # all new x and all new y
        xns, yns = xns.ravel(), yns.ravel()  # spread to 1D
        yns -= offset_y  # convert from new image coordinate to anchor image coordinate

        # convert anchor points like keypoints format (Nx2)
        anchor_points = np.vstack((xns, yns)).T

        for i in range(self.image_num):
            homog = self.homographies[i]  # transform from individual image to anchor image
            homog_inv = np.linalg.pinv(homog)  # transform from anchor image to individual image

            # height and width of individual image
            img = self.image_list[i]
            h, w, _ = img.shape

            ind_points = Homography.transform(homog_inv, anchor_points, round_int=True).T  # 2xN

            # keep those points that are inside the individual image dimension, update anchor points accordingly
            valid_col_indices = (ind_points[0] < h) & (ind_points[0] >= 0) & (ind_points[1] < w) & (ind_points[1] >= 0)
            remaining_ind_points = ind_points[:, valid_col_indices]  # 2xN
            remaining_anchor_points = anchor_points.T[:, valid_col_indices]  # 2xN

            # set valid individual image pixels to corresponding valid stitched image pixels
            # remember y offset between stitched and anchor images
            new_img[remaining_anchor_points[0], remaining_anchor_points[1] + offset_y] = \
                img[remaining_ind_points[0], remaining_ind_points[1]]

        # crop black border of the stitched image
        new_img = self.crop_border(new_img)

        # save stitched image
        plt_name = os.path.join(self.out_dir, 'stitched.jpg')
        plt.figure()
        plt.imshow(new_img)
        plt.axis('off')
        plt.savefig(plt_name, bbox_inches='tight', pad_inches=0, dpi=300)

        print(f"Image stitching finished and saved!")

    def _compute_stitched_offset(self):
        """
        Calculated stitched image size and its offsets to the anchor image based on homographies of all images w.r.t the
        anchor image
        :return: offset_x, offset_y, stitched image height, stitched image width
        """
        target_height, target_width, _ = self.image_list[self.anchor_img_id].shape
        xmin, ymin = 0, 0
        xmax, ymax = target_height, target_width

        for i in range(self.image_num):
            homog = self.homographies[i]

            h, w, _ = self.image_list[i].shape
            left_top_point = np.matmul(homog, np.array([0, 0, 1]))
            left_top_point = left_top_point / left_top_point[-1]
            left_bottom_point = np.matmul(homog, np.array([h, 0, 1]))
            left_bottom_point = left_bottom_point / left_bottom_point[-1]
            right_top_point = np.matmul(homog, np.array([0, w, 1]))
            right_top_point = right_top_point / right_top_point[-1]
            right_bottom_point = np.matmul(homog, np.array([h, w, 1]))
            right_bottom_point = right_bottom_point / right_bottom_point[-1]

            xmin = min([xmin, left_top_point[0], left_bottom_point[0], right_top_point[0], right_bottom_point[0]])
            xmax = max([xmax, left_top_point[0], left_bottom_point[0], right_top_point[0], right_bottom_point[0]])
            ymin = min([ymin, left_top_point[1], left_bottom_point[1], right_top_point[1], right_bottom_point[1]])
            ymax = max([ymax, left_top_point[1], left_bottom_point[1], right_top_point[1], right_bottom_point[1]])

        print(f"{xmin=} {xmax=} {ymin=} {ymax=}")
        new_h, new_w = int(xmax - xmin) + 100, int(ymax - ymin) + 100
        return int(-xmin), int(-ymin), new_h, new_w

    def get_empty_stitched_image(self):
        """
        For aesthetic reason, can force the stitched image height as the max height of all images, width as the sum
        of all image widths, since the panorama image is stitched horizontally in this homework
        :return: black stitched image
        """
        max_height = 0
        total_width = 0
        for img in self.image_list:
            max_height = max(max_height, img.shape[0])
            total_width += img.shape[1]

        return np.zeros((max_height, total_width, 3)).astype(np.uint8)

    def get_offset_y(self):
        """
        Assume all stitched images stick to the top of the overall image, thus only has y axis offset
        :return: offset pixels number
        """
        ofs_y = 0
        for i in range(self.anchor_img_id):
            ofs_y += self.image_list[i].shape[1]
        return ofs_y

    @staticmethod
    def crop_border(img, thr=0):
        """
        Crop black border of the stitched image
        :param img: 2D or 3D image data
        :param thr: pixel value threshold, under which the pixel will be deemed croppable
        :return:
        """
        mask = img > thr
        if img.ndim == 3:
            mask = mask.all(2)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
        return img[row_start: row_end, col_start: col_end]


if __name__ == "__main__":
    # specify input images directory and output panorama image direction
    input_dir = 'HW5-Images/given'
    output_dir = 'HW5-Images/given/Results'

    # input_dir = 'HW5-Images/own'
    # output_dir = 'HW5-Images/own/Results'

    # pipeline of image stitching, the function calling order should be respected
    mosaic = Mosaic(input_dir, output_dir, do_ransac=True, do_lm=True)
    mosaic.update_keypoints(center_shift=True)
    mosaic.update_matches(plot_matches=False)
    mosaic.update_homographies()
    mosaic.stitch()


