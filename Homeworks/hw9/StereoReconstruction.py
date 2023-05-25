import numpy as np
import matplotlib.pyplot as plt

from ImageRectification import Rectifier, triangulate
from RectMatcher import RectMatcher


class StereoReconstructor:
    def __init__(self, img_left_path: str, img_right_path: str):
        self.rectifier = Rectifier(img_left_path, img_right_path)
        self.img_left_rgb = self.rectifier.img_left_rgb
        self.img_right_rgb = self.rectifier.img_right_rgb
        self.matcher = None  # constructed when rectification is finished

    def reconstruct(self):
        print(f"Start reconstructing 3D scene ...")

        self.rectifier.rectify()

        self.matcher = RectMatcher(self.rectifier.left_rect_path, self.rectifier.right_rect_path)
        self.matcher.match(window_size=13, search_dist=45)

        # transform keypoints in rectified images to original images and plot
        H = self.rectifier.H
        Hp = self.rectifier.Hp
        # kpts_left = self.matcher.kpts_left + self.rectifier.offset_left
        # kpts_right = self.matcher.kpts_right + self.rectifier.offset_right
        kpts_left = self.matcher.kpts_left
        kpts_right = self.matcher.kpts_right
        kpts_left_ori = Rectifier.planar_transform(np.linalg.pinv(H), kpts_left, round_int=True)
        kpts_right_ori = Rectifier.planar_transform(np.linalg.pinv(Hp), kpts_right, round_int=True)
        RectMatcher.plot_matches(self.img_left_rgb, self.img_right_rgb, kpts_left_ori, kpts_right_ori, 'ori_matches.jpg')

        # update keypoints found in rectified images, as well as right camera canonical projection matrix
        self.rectifier.kpts_left = kpts_left_ori
        self.rectifier.kpts_right = kpts_right_ori
        self.rectifier.normalize_points()
        self.rectifier.build_fundamental_matrix()
        self.rectifier.build_projection_matrix()
        self.rectifier.refine()

        # display triangulated world points
        world_points = triangulate(self.rectifier.kpts_left, self.rectifier.kpts_right,
                                   self.rectifier.P, self.rectifier.Pp)
        step = 50
        fig = plt.figure()
        ax0 = fig.add_subplot(121)
        ax0.imshow(self.img_left_rgb)
        for i in range(len(kpts_left_ori)):
            if i % step == 0:
                x, y = kpts_left_ori[i]
                ax0.plot(x, y, 'r.', markersize=1)
                ax0.text(x, y, str(i // step), weight='bold')

        ax1 = fig.add_subplot(122)
        ax1.imshow(self.img_right_rgb)
        for i in range(len(kpts_right_ori)):
            if i % step == 0:
                x, y = kpts_right_ori[i]
                ax1.plot(x, y, 'r.', markersize=1)
                ax1.text(x, y, str(i // step), weight='bold')

        plt.savefig('keypoints_with_number.jpg', dpi=300)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=world_points[:, 0], ys=world_points[:, 1], zs=world_points[:, 2])

        for i in range(len(world_points)):
            if i % step == 0:
                ax.text(world_points[i, 0], world_points[i, 1], world_points[i, 2], str(i // step))
        # plt.savefig('3d_points_with_number.jpg', dpi=300)
        plt.show()

        print(f"3D reconstruction finished!")


if __name__ == '__main__':
    # waffle
    img_left_path = 'LoopAndZhang/LoopAndZhang/img/waffle_left.jpg'
    img_right_path = 'LoopAndZhang/LoopAndZhang/img/waffle_right.jpg'

    sr = StereoReconstructor(img_left_path, img_right_path)
    sr.reconstruct()


