import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def triangulate(kpts_left, kpts_right, P, Pp):
    assert len(kpts_left[0]) == 2
    assert len(kpts_right[0]) == 2
    assert P.shape == (3, 4)
    assert Pp.shape == (3, 4)

    p1, p2, p3 = P  # rows of P matrix
    p1p, p2p, p3p = Pp  # rows of P^prime matrix
    world_points = []
    for (x, y), (xp, yp) in zip(kpts_left, kpts_right):
        # solve x in Ax=0 homogeneous form
        A = np.zeros((4, 4))
        A[0] = x * p3 - p1
        A[1] = y * p3 - p2
        A[2] = xp * p3p - p1p
        A[3] = yp * p3p - p2p
        U, s, VT = np.linalg.svd(A.T @ A)
        X = VT[-1]  # last row is the eigen vector to the smallest eigenvalue
        X /= X[-1]
        world_points.append(X)
    return np.asarray(world_points)


class Rectifier:
    def __init__(self, img_left_path: str, img_right_path: str):
        # variables relating to images and their paths
        self.dir = os.path.dirname(img_left_path)
        self.left_name = os.path.basename(img_left_path).split('.')[0]
        self.right_name = os.path.basename(img_right_path).split('.')[0]
        self.img_left_rgb = cv2.cvtColor(cv2.imread(img_left_path), cv2.COLOR_BGR2RGB)
        self.img_right_rgb = cv2.cvtColor(cv2.imread(img_right_path), cv2.COLOR_BGR2RGB)
        self.img_left = cv2.cvtColor(self.img_left_rgb, cv2.COLOR_RGB2GRAY)
        self.img_right = cv2.cvtColor(self.img_right_rgb, cv2.COLOR_RGB2GRAY)
        self.img_left_rectified, self.img_right_rectified = None, None  # rectified images for easier matching
        self.left_rect_path, self.right_rect_path = None, None  # path of rectified images
        print(f"{self.img_left.shape=}")
        print(f"{self.img_right.shape=}")

        # variables relating to keypoints
        self.kpts_left, self.kpts_right = None, None  # keypoints of left and right images
        self.kpts_left_norm, self.kpts_right_norm = None, None  # normalized keypoints
        self.T, self.Tp = None, None  # normalization matrices of keypoints of 2 images

        # variables ralating to epipolar geometry
        self.F = None  # fundamental matrix
        self.E = None  # essential matrix
        self.e, self.ep = None, None  # epipoles of left and right images
        self.P = np.vstack((np.eye(3), np.zeros(3))).T  # canonical projection matrix of left image
        assert self.P.shape == (3, 4), "Wrong dimension of projection matrix!"
        self.Pp = np.zeros_like(self.P)  # canonical projection matrix of right image

        # variables relating to image warping
        self.H, self.Hp = None, None  # 3x3 homography matrices that warp two images to match epipolar lines
        self.offset_left, self.offset_right = np.zeros((2, ), dtype=int), np.zeros((2, ), dtype=int)  # offset (x, y) after warping

    def rectify(self):
        """
        Main pipeline
        :return:
        """
        self.build_matches(display=False)
        self.normalize_points()
        self.build_fundamental_matrix()
        self.build_projection_matrix()
        self.refine()
        self.build_homographies()
        self.warp_images_to_rectify()

    def build_matches(self, display: bool = False):
        print(f"Start building putative matches ...")
        # initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img_left, None)
        print(f"Found {len(kp1)} keypoints in image left, {kp1[0]}.")
        kp2, des2 = sift.detectAndCompute(self.img_right, None)
        print(f"Found {len(kp2)} keypoints in image right, {kp2[0]}.")

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        print(f"{matches[0]=}")
        print(f"Established {len(matches)} point correspondences.")

        # need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.4 * n.distance:
                matchesMask[i] = [1, 0]
        matches_len = len([1 for match in matchesMask if match[0] == 1])
        print(f"Enabled {matches_len} matches.")

        # display
        if display:
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                matchesMask=matchesMask,
                flags=cv2.DrawMatchesFlags_DEFAULT)

            img3 = cv2.drawMatchesKnn(self.img_left_rgb, kp1, self.img_right_rgb, kp2, matches, None, **draw_params)
            plt.axis('off')
            plt.imshow(img3, ), plt.show()

        # save matches as 2 matrices
        self.kpts_left = [kp1[matches[i][0].queryIdx].pt for i, match_mask in enumerate(matchesMask) if match_mask[0] == 1]
        self.kpts_right = [kp2[matches[i][0].trainIdx].pt for i, match_mask in enumerate(matchesMask) if match_mask[0] == 1]

        # keep coords as int (pos x right, pos y down)
        self.kpts_left = [[int(x), int(y)] for x, y in self.kpts_left]
        self.kpts_right = [[int(x), int(y)] for x, y in self.kpts_right]

        print(f"Putative matches building finished!")

    def normalize_points(self):
        assert self.kpts_left is not None and self.kpts_right is not None, "Build keypoints first!"

        # define function that builds normalization matrix T given keypoints
        def build_T(points):
            xy_mean = np.mean(points, axis=0)  # average point
            d = np.mean(np.linalg.norm(points - xy_mean, axis=1))  # mean distance to average point
            s = np.sqrt(2) / d
            T = np.array([[s, 0, -s*xy_mean[0]],
                          [0, s, -s*xy_mean[1]],
                          [0, 0, 1]])
            points_norm = s * (points - xy_mean)
            return points_norm, T

        # build normed keypoints and normalization matrices for keypoints of both images
        self.kpts_left_norm, self.T = build_T(self.kpts_left)
        self.kpts_right_norm, self.Tp = build_T(self.kpts_right)

    def build_fundamental_matrix(self):
        assert self.kpts_left_norm is not None and self.kpts_right_norm is not None, "Need normed keypoints!"
        assert self.T is not None and self.Tp is not None, "Need normalization matrix to de-normalize!"
        print(f"Start building putative fundamental matrix ...")

        # Af=0 homogeneous form
        A = []
        for (x, y), (xp, yp) in zip(self.kpts_left_norm, self.kpts_right_norm):
            A.append([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1])
        A = np.asarray(A)

        assert A.shape[0] >= 8, "Need at least 8 correspondences to build fundamental matrix!"
        assert A.shape[1] == 9, "Wrong columns in matrix A!"

        # solve elements of F matrix
        U, S, VT = np.linalg.svd(A.T @ A)
        f = VT[-1, :]  # last row is the eigen vector to the smallest eigenvalue
        F = f.reshape((3, 3))

        # condition F matrix such that rank(F) = 2
        F = self.condition_F(F)

        # de-normalize
        self.F = self.Tp.T @ F @ self.T
        self.F /= self.F[-1, -1]

        print(f"{self.F=}")
        rank = np.linalg.matrix_rank(self.F)
        print(f"rank(F) = {rank}")

        print(f"Putative fundamental matrix built!")

    @staticmethod
    def condition_F(F):
        U, s, VT = np.linalg.svd(F)
        s[-1] = 0  # zero out the smallest singular value
        F = U @ np.diag(s) @ VT
        F = F / F[-1, -1]
        return F

    @staticmethod
    def _skew_symmetric_matrix(v):
        assert len(v) == 3, "Dimension of vector should be 3 in this homework!"
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @staticmethod
    def null_vec(F, right: bool = True, atol: float = 1e-5):
        """
        Get right or left null vector of a matrix if there exists
        :param F: square matrix with 1 rank deficiency
        :param right: right null vector or left null vector of F
        :param atol: absolute tolerance of determinant of fundamental matrix F
        :return:
        """
        # if abs(np.linalg.det(F)) > atol:
        #     print(f"Matrix is full rank so that no null vector!")
        #     return None

        U, s, VT = np.linalg.svd(F if right else F.T)
        vec = VT[-1]
        vec /= vec[-1]
        # result = F @ vec if right else vec @ F
        # print(f"Check null result: {result}.")
        return vec

    def build_projection_matrix(self):
        assert self.F is not None, "Build fundamental matrix first!"
        print(f"Start building canonical projective matrix ...")

        # epipoles of left and right images are null vector and left null vector of matrix F
        self.e = self.null_vec(self.F, right=True)
        self.ep = self.null_vec(self.F, right=False)
        print(f"Left epipole: {self.e}")
        print(f"Right epipole: {self.ep}")

        self.Pp[:, :-1] = self._skew_symmetric_matrix(self.ep) @ self.F
        self.Pp[:, -1] = self.ep
        print(f"{self.Pp=}")
        assert abs(np.linalg.matrix_rank(self.Pp)) == 3, "Projection matrix of right image should be of rank 3!"

        print(f"Canonical projective matrix built!")

    def reprojection_error(self, projection=None):
        assert self.kpts_left is not None and self.kpts_right is not None, "Build correspondences first!"
        if projection is None:
            projection = self.Pp

        # triangulation to get world points
        world_points = triangulate(self.kpts_left, self.kpts_right, self.P, projection)

        # re-project to right image to calculate error since left image is world coord in canonical form
        errors = []
        for i, wp in enumerate(world_points):
            xp = self.kpts_right[i]
            reproj_xp = self.reproject(wp, projection)
            errors += (xp - reproj_xp).flatten().tolist()
        # print(f"Re-projection errors: {errors}.")
        return np.asarray(errors)

    @staticmethod
    def reproject(world_point, projection):
        assert len(world_point) == 4, "World point in homogeneous coord should be 4!"
        assert projection.shape == (3, 4), "Projection matrix should be of shape (3, 4)!"

        x = projection @ np.asarray(world_point)
        x /= x[-1]
        return x[:-1]

    @staticmethod
    def projection2vec(projection):
        assert projection.shape == (3, 4), "Wrong dimension of projection matrix!"
        p = projection.flatten()
        p = p / p[-1]
        p = p[:-1]  # 11 independent elements to optimize since projection matrix is homogeneous
        return p

    @staticmethod
    def vec2projection(p):
        assert len(p) == 11, "Wrong length of projection vector!"
        p = np.append(p, 1)  # homogeneous
        projection = p.reshape((3, -1))
        projection[-1, -1] = 1
        return projection

    def refine(self):
        print(f"Start refining projection matrix using LM ...")
        p0 = self.projection2vec(self.Pp)  # initial value

        def cost_func(p):
            projection = self.vec2projection(p)
            error = self.reprojection_error(projection)
            return error

        # non-linear optimization with LM algorithm
        result = least_squares(cost_func, p0, method='lm', verbose=2)
        if result.success:
            p_opt = result.x
            self.Pp = self.vec2projection(p_opt)  # refine right projection matrix
            m = self.Pp[:, -1]
            self.F = self._skew_symmetric_matrix(m) @ self.Pp @ np.linalg.pinv(self.P)
            # self.F = self.condition_F(self.F)
            self.e = self.null_vec(self.F, right=True)  # refine left epipole
            self.ep = self.null_vec(self.F, right=False)  # refine right epipole
            print(f"Optimized right camera projection matrix is {self.Pp}")
            print(f"Optimized left epipole: {self.e}")
            print(f"Optimized right epipole: {self.ep}")
        else:
            print(f"Optimization failed!")

        print(f"LM optimisation of projection matrix is finished!")

    def build_homographies(self):
        assert self.ep is not None, "Build right image epipole first!"
        print(f"Start building homographies to rectify two images ...")

        h, w = self.img_right.shape
        # from image coord (positive x right positive y down) to image center coord (pos x right pos y down)
        T1 = np.array([[1, 0, -w/2],
                       [0, 1, -h/2],
                       [0, 0, 1]], dtype=float)

        # from image center coord to image center epipolar coord (pos x along epipolar line)
        print(f"{self.ep=}")
        ex, ey, _ = self.ep
        theta = -np.arctan2(ey - h/2, ex - w/2)
        print(f"theta is {theta / np.pi * 180} deg")
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st, 0],
                      [st, ct, 0],
                      [0, 0, 1]])

        # get scale factor f and transformation G that maps epipole to infinity point [1, 0, 0]
        f = np.linalg.norm([ex - w/2, ey - h/2])
        print(f"{f=}")
        G = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [-1/f, 0, 1]])

        # from epipole horizontal right image center to original image center (pos x down pos y right)
        T2 = np.array([[1, 0, w/2],
                       [0, 1, h/2],
                       [0, 0, 1]])

        # build H_prime that warps right image
        self.Hp = T2 @ G @ R @ T1
        self.Hp /= self.Hp[-1, -1]
        print(f"{self.Hp=}")
        print(f"Right homog epipole: {G @ R @ T1 @ self.ep}")

        # build homography matrix to warp left image
        M = self.Pp @ np.linalg.pinv(self.P)
        H0 = self.Hp @ M
        kpts_left_tf = self.planar_transform(H0, self.kpts_left)  # transformed keypoints in left image
        kpts_right_tf = self.planar_transform(self.Hp, self.kpts_right)  # transformed keypoints in right image
        # solve Ax=b inhomogeneous linear least square problem to get HA matrix
        A = []
        b = []
        for i in range(len(kpts_left_tf)):
            x_hat, y_hat = kpts_left_tf[i]
            xp_hat = kpts_right_tf[i][0]
            A.append([x_hat, y_hat, 1])
            b.append(xp_hat)
        ha = np.linalg.pinv(np.asarray(A)) @ np.asarray(b)
        HA = np.eye(3)
        HA[0, :] = ha
        print(f"{HA=}")
        self.H = HA @ H0
        self.H /= self.H[-1, -1]
        print(f"{self.H=}")
        print(f"Left homog epipole: {self.H @ self.e}")

        print(f"Homographies built!")

    def warp_images_to_rectify(self):
        assert self.H is not None and self.Hp is not None, "Build homography matrices first!"
        print(f"Start rectify images ...")
        self.offset_left[0], self.offset_left[1], wl, hl = self.get_warped_img_shape(self.H, self.img_left)
        self.offset_right[0], self.offset_right[1], wr, hr = self.get_warped_img_shape(self.Hp, self.img_right)
        print(f"New shape for left image, height: {hl}, width: {wl}, offset: {self.offset_left}")
        print(f"New shape for right image, height: {hr}, width: {wr}, offset: {self.offset_right}")
        self.img_left_rectified = cv2.warpPerspective(self.img_left_rgb, self.H, (wl, hl), flags=cv2.INTER_LINEAR)
        self.img_right_rectified = cv2.warpPerspective(self.img_right_rgb, self.Hp, (wr, hr), flags=cv2.INTER_LINEAR)

        # self.img_left_rectified, self.offset_left = self.planar_transform_image(self.H, self.img_left_rgb)
        # self.img_right_rectified, self.offset_right = self.planar_transform_image(self.Hp, self.img_right_rgb)

        # save rectified images
        save_dir = os.path.join(self.dir, '../rectified')
        os.makedirs(save_dir, exist_ok=True)
        left_rect_name = self.left_name + '_rect.png'
        right_rect_name = self.right_name + '_rect.png'
        self.left_rect_path = os.path.join(save_dir, left_rect_name)
        self.right_rect_path = os.path.join(save_dir, right_rect_name)

        cv2.imwrite(self.left_rect_path, cv2.cvtColor(self.img_left_rectified, cv2.COLOR_RGB2BGR))
        cv2.imwrite(self.right_rect_path, cv2.cvtColor(self.img_right_rectified, cv2.COLOR_RGB2BGR))

        print(f"Image rectification finished!")

    def get_warped_img_shape(self, homog, image):
        h, w = image.shape[0], image.shape[1]
        top_left = [0, 0]
        top_right = [w - 1, 0]
        bottom_left = [0, h - 1]
        bottom_right = [w - 1, h - 1]
        corners = np.array([top_left, top_right, bottom_left, bottom_right])  # 4 corners of domain image
        corners_tf = self.planar_transform(homog, corners, round_int=True)  # 4 corners of range image by homography
        xmin, ymin = np.amin(corners_tf, axis=0)
        xmax, ymax = np.amax(corners_tf, axis=0)
        w_tf, h_tf = xmax - xmin + 1, ymax - ymin + 1  # height and width of transformed image
        return xmin, ymin, w_tf, h_tf

    def planar_transform_image(self, homog, image):
        h, w = image.shape[0], image.shape[1]
        print(f"Before transform, image height {h}, width {w}.")
        offset_x, offset_y, w_tf, h_tf = self.get_warped_img_shape(homog, image)
        print(f"After transform, image height {h_tf}, width {w_tf}.")
        img_range = np.zeros((h_tf, w_tf, 3), dtype=np.uint8)
        Hinv = np.linalg.pinv(homog)
        xs, ys = np.meshgrid(range(w_tf), range(h_tf))
        xs, ys = xs.ravel(), ys.ravel()
        range_points = np.vstack((xs, ys)).T  # N x 2
        offset_range_points = np.vstack((xs + offset_x, ys + offset_y)).T  # N x 2
        domain_points = self.planar_transform(Hinv, offset_range_points, round_int=True).T  # 2 x N
        valid_col_indices = (domain_points[0] < w) & (domain_points[0] >= 0) & (domain_points[1] < h) & (domain_points[1] >= 0)
        remaining_domain_points = domain_points[:, valid_col_indices]  # 2 x N_valid
        remaining_range_points = range_points.T[:, valid_col_indices]  # 2 x N_valid
        img_range[remaining_range_points[1], remaining_range_points[0]] = \
            image[remaining_domain_points[1], remaining_domain_points[0]]

        return img_range, np.array([offset_x, offset_y])

    @staticmethod
    def planar_transform(homog, points, round_int: bool = False):
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


if __name__ == '__main__':
    # V
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/img2.png'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/img1.png'

    # madera
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/madera_1.jpg'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/madera_2.jpg'

    # perra
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/perra_7.jpg'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/perra_8.jpg'

    # tissue
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/tissue_left.jpg'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/tissue_right.jpg'

    # sanitizer
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/sanitizer_left.jpg'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/sanitizer_right.jpg'
    
    # duck
    # img_left_path = 'LoopAndZhang/LoopAndZhang/img/duck_left.jpg'
    # img_right_path = 'LoopAndZhang/LoopAndZhang/img/duck_right.jpg'

    # waffle
    img_left_path = 'LoopAndZhang/LoopAndZhang/img/waffle_left.jpg'
    img_right_path = 'LoopAndZhang/LoopAndZhang/img/waffle_right.jpg'

    rectifier = Rectifier(img_left_path, img_right_path)
    rectifier.rectify()



