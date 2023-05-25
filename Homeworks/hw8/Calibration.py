import numpy as np
import os
from glob import glob
import re
import cv2
import csv
from scipy.optimize import least_squares

from CornerDetector import CornerDetector
from Homography import Homography


class Calibration:
    """
    Zhang's calibration algorithm
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.images = {}
        self.img_corners = {}
        self.homographies = {}  # homographies from checkerboard to all images
        self.world_corners = None
        self.omega = None  # repr of absolute conic in image
        self.K = None  # camera intrinsic matrix
        self.R = {}  # rotation matrix
        self.t = {}  # translation vector
        self.fixed_id = None  # fixed image id

        self.gen_world_corners(black_square_width=0.24, black_sqaure_height=0.245)
        self.get_corners()

    def get_corners(self):
        images_path = glob(self.dataset_path + '/*.jpg')
        print(f"{len(images_path)} images in the dataset.")
        for img_path in images_path:
            print(f"Processing {img_path} ...")
            cd = CornerDetector(img_path, (5, 4))
            corners = cd.get_corners()
            if corners is not None:
                img_name = os.path.basename(img_path)
                index = int(re.findall(r'\d+', img_name)[0])
                self.images[index] = cd.img
                self.img_corners[index] = corners
        print(f"{len(self.img_corners)} images are valid with corner detection!")

    def gen_world_corners(self, black_square_width: float, black_sqaure_height: float, rows: int = 5, cols: int = 4):
        corners = []
        for r in range(rows):
            for c in range(cols):
                x = 2 * c * black_square_width
                y = 2 * r * black_sqaure_height
                corners.append([x, y])
                x += black_square_width
                corners.append([x, y])
            for c in range(cols):
                x = 2 * c * black_square_width
                y = (2 * r + 1) * black_sqaure_height
                corners.append([x, y])
                x += black_square_width
                corners.append([x, y])
        assert len(corners) == rows * cols * 4, 'Wrong number of world corners!'
        self.world_corners = np.asarray(corners)
        print(f"{self.world_corners=}")

    def calibrate(self, optimize: bool = False, with_radial_distortion: bool = True):
        print(f"Start camera calibration!")
        self.solve_absolute_conic()
        self.solve_intrinsics()
        self.solve_extrinsics()

        assert self.img_corners.keys() == self.homographies.keys() == self.R.keys() == self.t.keys(), \
            'Extrinsics length mismatch!'

        self.get_fixed_img_id()
        self.get_reproj_error(self.R, self.t, self.K, (0, 0))

        if optimize:
            self.optimize(with_radial_distortion)
        else:
            self.get_reproj_error(self.R, self.t, self.K, (0, 0), csv_file_name='reproj_error.csv',
                                  img_folder_name='reproj')
            self.combine_Rt(self.K, self.R, self.t, (0, 0))
        print(f"Camera calibration finished!")

    def optimize(self, with_radial_distortion: bool):
        # get initial param
        p0 = []
        p0 += [self.K[0, 0], self.K[0, 1], self.K[0, 2], self.K[1, 1], self.K[1, 2]]

        if with_radial_distortion:
            p0 += [0, 0]  # initial values for k1 and k2

        abs_id_to_img_id = {}
        for index, i in enumerate(self.R.keys()):
            abs_id_to_img_id[index] = i
            R, t = self.R[i], self.t[i]
            w = self._rotation2rodrigues(R)
            p0 += w.tolist()
            p0 += t.tolist()

        p0 = np.asarray(p0)
        # print(f"{p0=}")

        non_Rt_param_num = 7 if with_radial_distortion else 5
        assert len(p0) == non_Rt_param_num + 6 * len(self.R), 'Parameter size wrong!'

        def param2K(param):
            alpha_x, s, x0, alpha_y, y0 = param[:5]
            return np.array([[alpha_x, s, x0],
                           [0, alpha_y, y0],
                           [0, 0, 1]])

        def param2distortion(param):
            return (param[5], param[6]) if with_radial_distortion else (0, 0)

        def param2Rdict(param):
            assert (len(param) - non_Rt_param_num) % 6 == 0, f'Wrong size of param vector: {len(param)}!'
            num = int((len(param) - non_Rt_param_num) / 6)
            p = param[non_Rt_param_num:]  # only iterate over Rt vector part
            R_dict = {}
            for i in range(num):
                w = p[i * 6: i * 6 + 3]
                R = self._rodrigues2rotation(w)
                R_dict[abs_id_to_img_id[i]] = R
            return R_dict

        def param2tdict(param):
            assert (len(param) - non_Rt_param_num) % 6 == 0, f'Wrong size of param vector: {len(param)}!'
            num = int((len(param) - non_Rt_param_num) / 6)
            p = param[non_Rt_param_num:]  # only iterate over Rt vector part
            t_dict = {}
            for i in range(num):
                t = p[i * 6 + 3: i * 6 + 6]
                t_dict[abs_id_to_img_id[i]] = t
            return t_dict

        # define cost function
        def cost_func(param):
            R_dict, t_dict, K, rd = param2Rdict(param), param2tdict(param), param2K(param), param2distortion(param)
            assert len(R_dict) == len(t_dict), 'R dict and t dict length not match!'
            assert R_dict.keys() == t_dict.keys(), 'Cost function wrong!'
            return self.get_reproj_error(R_dict, t_dict, K, rd)

        # start optimize, method can be one of {‘trf’, ‘dogbox’, ‘lm’}
        # result = least_squares(cost_func, p0, loss='soft_l1', verbose=2)

        result = least_squares(cost_func, p0, method='lm', verbose=2)

        if not result.success:
            print(f"Least square failed!")
        else:
            print(f"Least square succeeded!")
        p_opt = result.x  # optimized param vector
        R_dict_opt, t_dict_opt, K_opt, rd = param2Rdict(p_opt), param2tdict(p_opt), param2K(p_opt), param2distortion(p_opt)
        error_filename = 'reproj_lm_rd_error.csv' if with_radial_distortion else 'reproj_lm_error.csv'
        img_folder_name = 'reproj_lm_rd' if with_radial_distortion else 'reproj_lm'
        self.get_reproj_error(R_dict_opt, t_dict_opt, K_opt, rd, csv_file_name=error_filename,
                              img_folder_name=img_folder_name)

        self.combine_Rt(K_opt, R_dict_opt, t_dict_opt, rd)

    def get_homography(self, img_id: int) -> np.array:
        homog = Homography()
        H = homog.get_homography(self.world_corners, self.img_corners[img_id])
        return H

    def solve_absolute_conic(self):
        V = np.zeros((len(self.img_corners) * 2, 6)).astype(float)

        for index, i in enumerate(self.img_corners.keys()):
            H = self.get_homography(i)
            self.homographies[i] = H
            h1 = H[:, 0]
            h2 = H[:, 1]
            V[2 * index, :] = self._construct_V_row(h1, h2)
            V[2 * index + 1, :] = self._construct_V_row(h1, h1) - self._construct_V_row(h2, h2)

        # Vb=0 homogeneous form
        U, S, VT = np.linalg.svd(V.T @ V)
        b = VT[-1, :]  # last row is the eigen vector to the smallest eigenvalue
        self.omega = np.array([[b[0], b[1], b[3]],
                               [b[1], b[2], b[4]],
                               [b[3], b[4], b[5]]])
        print(f"{self.omega=}")

    def solve_intrinsics(self):
        assert self.omega is not None, 'Solve absolute conic first!'

        [w11, w12, w13], [w21, w22, w23], [w31, w32, w33] = self.omega

        x0 = (w12 * w13 - w11 * w23) / (w11 * w22 - w12 ** 2)
        lam = w33 - (w13 ** 2 + x0 * (w12 * w13 - w11 * w23)) / w11
        alpha_x = np.sqrt(lam / w11)
        alpha_y = np.sqrt(lam * w11 / (w11 * w22 - w12 ** 2))
        s = -w12 * alpha_x ** 2 * alpha_y / lam
        y0 = s * x0 / alpha_y - w13 * alpha_x ** 2 / lam

        self.K = np.array([[alpha_x, s, x0],
                           [0, alpha_y, y0],
                           [0, 0, 1]])
        print(f"{self.K=}")

    def solve_extrinsics(self):
        assert self.K is not None, 'Solve camera intrinsics first!'
        assert len(self.homographies) != 0, 'Empty homographies!'

        for i, H in self.homographies.items():
            h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
            K_inv = np.linalg.inv(self.K)
            scaling_factor = 1 / np.linalg.norm(K_inv @ h1)

            r1 = scaling_factor * K_inv @ h1
            r2 = scaling_factor * K_inv @ h2
            r3 = np.cross(r1, r2)
            t = scaling_factor * K_inv @ h3
            R = np.vstack((r1, r2, r3)).T

            # condition the rotation matrix
            U, S, VT = np.linalg.svd(R)
            R = U @ VT

            self.R[i] = R
            self.t[i] = t

    @staticmethod
    def _rotation2rodrigues(R: np.array) -> np.array:
        assert R.shape == (3, 3), 'Rotation matrix should be 3 x 3'

        phi = np.arccos((np.trace(R) - 1) / 2)
        rdg = phi / 2 / np.sin(phi) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return rdg

    @staticmethod
    def _rodrigues2rotation(rdg: np.array) -> np.array:
        assert len(rdg) == 3, 'Wrong dimension of Rodrigues vector!'

        W = np.array([[0, -rdg[2], rdg[1]],
                      [rdg[2], 0, -rdg[0]],
                      [-rdg[1], rdg[0], 0]])
        phi = np.linalg.norm(rdg)
        R = np.eye(3) + np.sin(phi) / phi * W + (1 - np.cos(phi)) / phi ** 2 * W @ W
        return R

    def _construct_V_row(self, h1: np.array, h2: np.array) -> np.array:
        assert len(h1) == len(h2) == 3, 'Invalid homography column input!'
        m = np.outer(h1, h2)  # m is outer producted matrix of 2 homography columns
        return np.array([m[0, 0], m[0, 1] + m[1, 0], m[1, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2]])

    def _radially_distort(self, points: np.array, k1: float = 0, k2: float = 0) -> np.array:
        assert points.shape[1] == 2, 'Points shape wrong!'
        x0, y0 = self.K[0, 2], self.K[1, 2]  # Principal point
        points_rd = []
        for x, y in points:
            r2 = (x - x0) ** 2 + (y - y0) ** 2  # squared distance from (x, y) to Principal point (x0, y0)
            r4 = r2 ** 2
            x_rd = x + (x - x0) * (k1 * r2 + k2 * r4)
            y_rd = y + (y - y0) * (k1 * r2 + k2 * r4)
            points_rd.append([x_rd, y_rd])
        return np.asarray(points_rd)

    def get_fixed_img_id(self):
        """
        Fixed image should have the minimum rotation magnitude
        :return:
        """
        assert len(self.R) > 0, 'No rotation matrix to get fixed image!'
        min_norm = np.inf
        for i, R in self.R.items():
            w = self._rotation2rodrigues(R)
            norm = np.linalg.norm(w)
            if norm < min_norm:
                min_norm = norm
                self.fixed_id = i
        print(f"Fixed image id is {self.fixed_id}.")

    def get_reproj_error(self, R_dict: {}, t_dict: {}, K: np.array, rd: tuple,
                         csv_file_name: str = '', img_folder_name: str = '') -> np.array:
        assert self.fixed_id is not None, 'No fixed image determined!'
        assert len(rd) == 2, 'Radial distortion only has 2 parameters!'

        errors = []
        all_errors = []
        for i, R in R_dict.items():
            t = t_dict[i]
            H = K @ np.vstack((R[:, 0], R[:, 1], t)).T

            # reproject to each image itself
            # reproj_fixed_points = Homography.transform(H, self.world_corners, round_int=False)
            # points_diff = reproj_fixed_points - self.img_corners[i]

            # reproject to the same fixed image
            reproj_world_points = Homography.transform(np.linalg.inv(H), self.img_corners[i], round_int=False)
            reproj_fixed_points = Homography.transform(self.homographies[self.fixed_id], reproj_world_points,
                                                       round_int=False)
            reproj_fixed_points = self._radially_distort(reproj_fixed_points, *rd)

            points_diff = reproj_fixed_points - self.img_corners[self.fixed_id]  # corners_num x 2

            if img_folder_name != '':
                img_reproj_fixed = np.copy(self.images[self.fixed_id])

                # display original image corners
                for corner_index, (x, y) in enumerate(self.img_corners[self.fixed_id]):
                    cv2.circle(img_reproj_fixed, (x, y), color=(0, 0, 255), radius=2, thickness=-1)
                    cv2.putText(img_reproj_fixed, str(corner_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

                # display reprojected image corners
                reproj_fixed_points = reproj_fixed_points.astype(int)
                for corner_index, (x, y) in enumerate(reproj_fixed_points):
                    cv2.circle(img_reproj_fixed, (x, y), color=(0, 255, 0), radius=2, thickness=-1)
                    cv2.putText(img_reproj_fixed, str(corner_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)

                fixed_reproj_folder = os.path.join(self.dataset_path, img_folder_name)
                if not os.path.exists(fixed_reproj_folder):
                    os.mkdir(fixed_reproj_folder)
                img_name = str(i) + '-on-' + str(self.fixed_id) + '.jpg'
                cv2.imwrite(os.path.join(fixed_reproj_folder, img_name), img_reproj_fixed)

            # append current image's error vector to the overall error vector
            all_errors += points_diff.flatten().tolist()

            error_norm = np.linalg.norm(points_diff, axis=1)  # corners_num x 1
            errors.append(error_norm)

        errors = np.asarray(errors)  # images_num x corners_num

        # save to csv file, row denotes geometric re-projection error for all corners, columns denotes images
        if csv_file_name != '':
            filename = os.path.join(self.dataset_path, csv_file_name)
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(errors)
            print(f"Reprojection errors saved!")

        error_mean = np.mean(errors, axis=1)  # images_num x 1
        error_sum = np.sum(error_mean)  # scalar
        error_var = np.var(error_mean)  # scalar
        error_min = np.min(error_mean)  # scalar
        error_max = np.max(error_mean)  # scalar
        error_mean = np.mean(error_mean)  # scalar
        print(f"Reprojection error, {error_mean=} {error_sum=} {error_var=} {error_min=} {error_max=}")

        return np.asarray(all_errors)

    def combine_Rt(self, K, R_dict: {}, t_dict: {}, rd):
        print(f"Camera intrinsic matrix K is: ")
        print(K)
        for i, R in R_dict.items():
            t = t_dict[i]
            Rt = np.vstack((R.T, t)).T
            print(f"Extrinsic matrix for image {i} is: ")
            print(Rt)
        print(f"Camera radial distortion params are: ")
        print(rd)


if __name__ == '__main__':
    dataset_folder = 'HW8-Files/Dataset1'

    # dataset_folder = 'HW8-Files/Dataset2'

    calib = Calibration(dataset_folder)
    calib.calibrate(optimize=True, with_radial_distortion=True)


