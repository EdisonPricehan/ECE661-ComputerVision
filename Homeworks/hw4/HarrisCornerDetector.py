
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HarrisCornerDetector:
    def __init__(self, img_path, scale_factor=1.0, sigma=1.2, k=0.04, resp_thr=0.1, use_cv_convolution=True):
        """
        Harris corner detector
        :param img_path: absolute path of input image
        :param scale_factor: scale factor of image, should be (0, 1]
        :param sigma: Gaussian sigma
        :param k: Det(grad) - k * Tr(grad)^2, threshold
        :param resp_thr: percentile threshold of response map
        :param use_cv_convolution: whether use cv2.filter2D or own convolution process, former is faster
        """
        # read image and set params
        self.img_rgb = cv2.imread(img_path)
        self.sigma = sigma
        self.k = k
        self.resp_thr = resp_thr
        # minimum even number that is greater than 4*sigma
        self.haar_width = self.get_haar_width()
        # 5*sigma x 5*sigma neighbourhood to construct C matrix of summation of squared gradients
        self.grad_nbd_width = int(self.sigma * 5)

        # scale image
        dim = (int(self.img_rgb.shape[1] * scale_factor), int(self.img_rgb.shape[0] * scale_factor))
        self.img_rgb = cv2.resize(self.img_rgb, dim)

        # convert image to grayscale then normalize
        self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        self.img = self.img / 255

        # get integral image
        self.intergal_img = self.get_integral_image()

        # get gradient img
        self.grad_img = self.get_gradient_image_cv() if use_cv_convolution else self.get_gradient_image()

    def get_haar_width(self):
        """
        Haar kernel width is the minimum even integer of 4*sigma
        :return: haar kernel width
        """
        haar_width = int(np.ceil(4 * self.sigma))
        haar_width = haar_width + 1 if haar_width % 2 == 1 else haar_width
        return haar_width

    def get_haar_kernel(self, axis=0):
        """
        Haar kernel matrix
        :param axis: 0 along image height, 1 along image width
        :return: haar matrix with pre-calculated width
        """
        kernel = np.ones((self.haar_width, self.haar_width))
        if axis == 0:
            kernel[int(self.haar_width / 2):, :] = -1
        else:
            kernel[:, :int(self.haar_width / 2)] = -1
        return kernel

    def get_integral_image(self):
        """
        Pre-process to get integral image for faster haar kernel convolution
        :return: integral image
        """
        h, w = self.img.shape
        print(f"Image height: {h}, width: {w}")
        integral_img = np.zeros_like(self.img)
        for i in range(h):
            for j in range(w):
                left = integral_img[i, j - 1] if j > 0 else 0
                top = integral_img[i - 1, j] if i > 0 else 0
                top_left = integral_img[i - 1, j - 1] if i > 0 and j > 0 else 0
                cur = self.img[i, j]
                # print(f"{cur=} {left=} {top=} {top_left=}")
                integral_img[i, j] = cur + top + left - top_left
        return integral_img

    def get_rect_sum(self, x1: int, x2: int, y1: int, y2: int) -> int:
        """
        Get sum of pixel values within a rectangle defined by two x and two y values
        :param x1: min or max x value of the rectangle
        :param x2: min or max x value of the rectangle
        :param y1: min or max y value of the rectangle
        :param y2: min or max y value of the rectangle
        :return: sum in int
        """
        h, w = self.img.shape
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        # no overlap with image at all
        if xmax < 0 or xmin >= h or ymax < 0 or ymin >= w:
            return 0
        # clip interested rectangle to the overlapping pixels with image
        xmin, xmax = np.clip([xmin, xmax], 0, h - 1)
        ymin, ymax = np.clip([ymin, ymax], 0, w - 1)

        # get the pixel values sum of the overlapping rectangle inclusively
        top_right = self.intergal_img[xmin - 1, ymax] if xmin > 0 else 0
        bottom_left = self.intergal_img[xmax, ymin - 1] if ymin > 0 else 0
        top_left = self.intergal_img[xmin - 1, ymin - 1] if xmin > 0 and ymin > 0 else 0
        bottom_right = self.intergal_img[xmax, ymax]
        return bottom_right + top_left - top_right - bottom_left

    def get_gradient_image(self):
        """
        Get 2 x h x w gradient image with dx in the first layer and dy in the second layer
        :return: gradient image
        """
        dx_img = np.zeros_like(self.img)
        dy_img = np.zeros_like(self.img)
        h, w = dx_img.shape
        for i in range(h):
            for j in range(w):
                dx_img[i, j] = self.get_haar_grad(i, j, axis=0)
                dy_img[i, j] = self.get_haar_grad(i, j, axis=1)

        grad_img = np.stack((dx_img, dy_img))
        return grad_img

    def get_gradient_image_cv(self):
        """
        OpenCV kernel filtering to get gradient 2 x h x w image, faster
        :return: gradient image
        """
        haar_kernel_x = self.get_haar_kernel(axis=0)
        haar_kernel_y = self.get_haar_kernel(axis=1)
        # print(f"{haar_kernel_x=}")
        # print(f"{haar_kernel_y=}")
        dx_img = cv2.filter2D(self.img, -1, haar_kernel_x)
        dy_img = cv2.filter2D(self.img, -1, haar_kernel_y)

        grad_img = np.stack((dx_img, dy_img))
        return grad_img

    def get_haar_grad(self, xc: int, yc: int, axis: int = 0):
        """
        Get gradient value of certain pixel using Haar kernel
        :param xc: center x value
        :param yc: center y value
        :param axis: 0 along image height, 1 along image width
        :return: pixel gradient (dx or dy) of the pixel
        """
        top = int(xc - self.haar_width / 2)
        bottom = top + self.haar_width - 1
        left = int(yc - self.haar_width / 2)
        right = left + self.haar_width - 1
        if axis == 0:  # dx along height
            pos_sum = self.get_rect_sum(top, xc - 1, left, right)
            neg_sum = self.get_rect_sum(xc, bottom, left, right)
        else:  # dy along width
            pos_sum = self.get_rect_sum(top, bottom, yc, right)
            neg_sum = self.get_rect_sum(top, bottom, left, yc - 1)
        return pos_sum - neg_sum

    def get_corners(self):
        """
        Get all Harris corners in the image
        :return: [[all corners x], [all corners y]]
        """
        # get squared gradients
        dx, dy = self.grad_img
        dxdx = np.multiply(dx, dy)
        dydy = np.multiply(dy, dy)
        dxdy = np.multiply(dx, dy)

        # get summation of squared gradients within some window determined by sigma
        sum_kernel = np.ones((self.grad_nbd_width, self.grad_nbd_width))
        sum_dxdx = cv2.filter2D(dxdx, -1, sum_kernel)
        sum_dydy = cv2.filter2D(dydy, -1, sum_kernel)
        sum_dxdy = cv2.filter2D(dxdy, -1, sum_kernel)

        # get responses of all pixels, higher ones can be corners
        det = sum_dxdx * sum_dydy - sum_dxdy ** 2
        trace = sum_dxdx + sum_dydy
        response = det - self.k * trace ** 2
        # response = cv2.dilate(response, None)

        # do nms
        response = self.nms(response, keep_top_k=1)

        corners = np.where(response > self.resp_thr * response.max())
        print(f"Found {len(corners[0])} corners.")
        return corners

    def nms(self, resp, keep_top_k=1):
        """
        Apply Non-maximum Suppression to the response image to reduce too dense key points
        :param resp: 2D response of image
        :param keep_top_k:
        :return: 2D suppressed response image
        """
        h, w = self.img.shape
        r = int(self.sigma * 10)
        nms_resp = np.zeros_like(resp)

        # cut reps with border length r to inner map
        for x in range(r, h - r, r):
            for y in range(r, w - r, r):
                window = resp[x - r: x + r + 1, y - r: y + r + 1]
                max_val = window.max()
                max_loc = np.where(window == max_val)
                for idx, (i, j) in enumerate(zip(*max_loc)):
                    if idx >= keep_top_k:
                        break
                    nms_resp[x - r + i, y - r + j] = max_val

        return nms_resp

    def display(self, corners):
        """
        Display detected Harris corners
        :param corners:
        :return:
        """
        img_rgb_kps = self.img_rgb.copy()
        print(f"Found {len(corners[0])} corners.")
        img_rgb_kps[corners[0], corners[1]] = [255, 0, 0]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.img_rgb)
        axs[1].imshow(img_rgb_kps)
        axs[1].scatter(corners[1], corners[0], s=50, facecolors='none', edgecolors='r')
        plt.show()


if __name__ == "__main__":
    # hcd = HarrisCornerDetector("/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_1.jpeg",
    #                            scale_factor=1)
    # hcd = HarrisCornerDetector("/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/books_2.jpeg",
    #                            scale_factor=1, resp_thr=0.1, use_cv_convolution=True)

    hcd = HarrisCornerDetector("/home/edison/Research/ECE661/Homeworks/hw4/HW4-Images/Figures/fountain_1.jpg",
                               scale_factor=1, sigma=1)
    corners = hcd.get_corners()
    hcd.display(corners)
