
import cv2
import numpy as np
from glob import glob
import os


class CornerDetector:
    def __init__(self, img_path: str, dim=(5, 4)):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.rows, self.cols = dim
        self.vertical_sorted_lines = None
        self.horizontal_sorted_lines = None
        self.corners = None
        self.debug = False

    def get_lines(self):
        # Step 1: blur gray image
        kernel_size = 5
        blur = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), 0)

        # Step 2: detect Canny edges
        low_threshold = 80  # Hysteresis minVal gradient magnitude below which is sure non-edge
        high_threshold = 220  # Hysteresis maxVal gradient magnitude above which is sure edge
        edges = cv2.Canny(blur, low_threshold, high_threshold)

        if self.debug:
            # cv2.imshow('Canny edges', edges)
            # cv2.waitKey(3000)

            canny_folder = os.path.join(os.path.split(self.img_path)[0], 'canny')
            if not os.path.exists(canny_folder):
                os.mkdir(canny_folder)
            img_name = os.path.split(self.img_path)[-1]
            cv2.imwrite(os.path.join(canny_folder, img_name), edges)

        # Step 3: apply Hough Transform to detect lines
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 45  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 30  # minimum number of pixels making up a line
        max_line_gap = 100  # maximum gap in pixels between connectable line segments
        line_image = np.copy(self.img)  # creating a new image to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        trials = 10
        while trials > 0:
            trials -= 1
            lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
            # print(f"Detected {len(lines)} lines.")
            # print(f"{lines.shape=}")
            lines = lines.squeeze()
            # print(f"{lines.shape=}")
            assert len(lines) >= 2 * (self.rows + self.cols), 'Not enough lines to get all corners!'
            # print(f"{lines=}")

            if self.debug:
                # draw lines on original image
                for x1, y1, x2, y2 in lines:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 4)

                # cv2.imshow('Image with lines', line_image)
                # cv2.waitKey(50000)

                lines_folder = os.path.join(os.path.split(self.img_path)[0], 'lines')
                if not os.path.exists(lines_folder):
                    os.mkdir(lines_folder)
                img_name = os.path.split(self.img_path)[-1]
                cv2.imwrite(os.path.join(lines_folder, img_name), line_image)

            # Step 4: filter lines
            delta_x = lines[:, 2] - lines[:, 0]
            delta_y = lines[:, 3] - lines[:, 1]
            slope = delta_y / delta_x
            # print(f"{slope=}")

            vertical_lines = lines[abs(slope) > 1]
            horizontal_lines = lines[abs(slope) <= 1]
            # print(f"Vertical lines num: {len(vertical_lines)}, horizontal lines num: {len(horizontal_lines)}")

            dist_to_vertical_lines = []
            for vl in vertical_lines:
                dist_to_vertical_lines.append(self.origin_to_line_dist(vl))
            dist_to_vertical_lines = np.array(dist_to_vertical_lines)
            # print(f"{dist_to_vertical_lines=}")
            indices = np.argsort(dist_to_vertical_lines)
            # print(f"{indices=}")
            dist_sorted = dist_to_vertical_lines[indices]
            # print(f"Vertical {dist_sorted=}")
            dist_thr = 10
            vertical_lines_filtered = []
            for i in range(len(dist_sorted)):
                if len(vertical_lines_filtered) == 0:
                    vertical_lines_filtered.append(vertical_lines[indices[i]])
                else:
                    if dist_sorted[i] - dist_sorted[i - 1] > dist_thr:
                        vertical_lines_filtered.append(vertical_lines[indices[i]])
            if len(vertical_lines_filtered) != self.cols * 2:
                print(f'Invalid filtered vertical lines num {len(vertical_lines_filtered)}!')
                continue

            dist_to_horizontal_lines = []
            for hl in horizontal_lines:
                dist_to_horizontal_lines.append(self.origin_to_line_dist(hl))
            dist_to_horizontal_lines = np.array(dist_to_horizontal_lines)
            # print(f"{dist_to_horizontal_lines=}")
            indices = np.argsort(dist_to_horizontal_lines)
            dist_sorted = dist_to_horizontal_lines[indices]
            # print(f"Horizontal {dist_sorted=}")
            # dist_thr = 10
            horizontal_lines_filtered = []
            for i in range(len(dist_sorted)):
                if len(horizontal_lines_filtered) == 0:
                    horizontal_lines_filtered.append(horizontal_lines[indices[i]])
                else:
                    if dist_sorted[i] - dist_sorted[i - 1] > dist_thr:
                        horizontal_lines_filtered.append(horizontal_lines[indices[i]])
            if len(horizontal_lines_filtered) != self.rows * 2:
                print(f'Invalid filtered horizontal lines num {len(horizontal_lines_filtered)}!')
                continue

            if trials == 0:
                print(f"Used up all trials but cannot find exact number of lines!")
                return

            print(f"Exact number of distinctive lines are extracted.")
            self.vertical_sorted_lines = vertical_lines_filtered
            self.horizontal_sorted_lines = horizontal_lines_filtered

            if self.debug:
                img_line_sep = np.copy(self.img)
                for vl in vertical_lines_filtered:
                    x1, y1, x2, y2 = self.extend_line(vl)
                    cv2.line(img_line_sep, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for hl in horizontal_lines_filtered:
                    x1, y1, x2, y2 = self.extend_line(hl)
                    cv2.line(img_line_sep, (x1, y1), (x2, y2), (0, 0, 255), 2)
                filtered_lines_folder = os.path.join(os.path.split(self.img_path)[0], 'lines_filtered')
                if not os.path.exists(filtered_lines_folder):
                    os.mkdir(filtered_lines_folder)
                img_name = os.path.split(self.img_path)[-1]
                cv2.imwrite(os.path.join(filtered_lines_folder, img_name), img_line_sep)

                # cv2.imshow('Image with separated lines', img_line_sep)
                # cv2.waitKey(50000)
            return self.vertical_sorted_lines, self.horizontal_sorted_lines

    def get_corners(self):
        self.get_lines()

        if not self.vertical_sorted_lines or not self.horizontal_sorted_lines:
            print(f"No exact lines detected for image {self.img_path}!")
            return None

        corners = []
        for hl in self.horizontal_sorted_lines:
            hl_homog = self.homog_line(hl)
            for vl in self.vertical_sorted_lines:
                vl_homog = self.homog_line(vl)
                x, y = self.intersect_pixel(vl_homog, hl_homog)
                corners.append([x, y])
        print(f"Detected {len(corners)} corners!")

        if self.debug:
            img_corners = np.copy(self.img)
            for i, (x, y) in enumerate(corners):
                cv2.circle(img_corners, (x, y), color=(0, 0, 255), radius=2, thickness=-1)
                cv2.putText(img_corners, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # cv2.imshow('Image with corners', img_corners)
            # cv2.waitKey(50000)

            corners_folder = os.path.join(os.path.split(self.img_path)[0], 'corners')
            if not os.path.exists(corners_folder):
                os.mkdir(corners_folder)
            img_name = os.path.split(self.img_path)[-1]
            cv2.imwrite(os.path.join(corners_folder, img_name), img_corners)

        self.corners = np.asarray(corners)
        assert len(corners) == 4 * self.rows * self.cols, 'Corners number does not match!'
        return self.corners

    def origin_to_line_dist(self, line):
        l = self.homog_line(line)
        dist = abs(l[2]) / np.sqrt(l[0] ** 2 + l[1] ** 2)
        return dist

    def homog_line(self, line):
        x1, y1, x2, y2 = line
        l = np.cross([x1, y1, 1], [x2, y2, 1])  # homogeneous line
        return l

    def intersect_pixel(self, l1, l2):
        point = np.cross(l1, l2)
        return int(point[0] / point[2]), int(point[1] / point[2])

    def extend_line(self, line, dist=10000):
        angle = np.arctan2(line[3] - line[1], line[2] - line[0])
        x1 = int(line[0] + dist * np.cos(angle))
        y1 = int(line[1] + dist * np.sin(angle))
        x2 = int(line[0] - dist * np.cos(angle))
        y2 = int(line[1] - dist * np.sin(angle))
        return x1, y1, x2, y2


if __name__ == '__main__':
    images_folder = 'HW8-Files/Dataset1'

    # images_folder = 'HW8-Files/Dataset2'

    images_path = glob(images_folder + '/*.jpg')
    print(f"In total {len(images_path)} images.")

    valid_count = len(images_path)
    for img_path in images_path:
        # if 'Pic_40.jpg' not in img_path:
        #     continue
        print(f"Processing {img_path} ...")
        cd = CornerDetector(img_path)
        corners = cd.get_corners()
        if corners is None:
            valid_count -= 1
    print(f"{valid_count} images are valid with corner detection!")

