import numpy as np
import os
import cv2

from FeatureExtractor import Haar


class DataLoader:
    def __init__(self, train_path: str, test_path: str):
        """
        Extract and vectorize feature vectors of images, as well as their labels
        :param train_path:
        :param test_path:
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_pos_path = os.path.join(self.train_path, 'positive')
        self.train_neg_path = os.path.join(self.train_path, 'negative')
        self.test_pos_path = os.path.join(self.test_path, 'positive')
        self.test_neg_path = os.path.join(self.test_path, 'negative')

        self.train_pos_images = [
            cv2.imread(os.path.join(self.train_pos_path, img_name), cv2.COLOR_RGB2GRAY)[..., 0] for img_name in
            os.listdir(self.train_pos_path)]
        self.train_neg_images = [
            cv2.imread(os.path.join(self.train_neg_path, img_name), cv2.COLOR_RGB2GRAY)[..., 0] for img_name in
            os.listdir(self.train_neg_path)]
        self.train_images = self.train_pos_images + self.train_neg_images
        print(f"Read in total {len(self.train_images)} train images, {len(self.train_pos_images)} positive, "
              f"{len(self.train_neg_images)} negative!")
        self.train_y = np.array([1] * len(self.train_pos_images) + [-1] * len(self.train_neg_images))
        assert len(self.train_images) == len(self.train_y)

        self.test_pos_images = [
            cv2.imread(os.path.join(self.test_pos_path, img_name), cv2.COLOR_RGB2GRAY)[..., 0] for img_name in
            os.listdir(self.test_pos_path)]
        self.test_neg_images = [
            cv2.imread(os.path.join(self.test_neg_path, img_name), cv2.COLOR_RGB2GRAY)[..., 0] for img_name in
            os.listdir(self.test_neg_path)]
        self.test_images = self.test_pos_images + self.test_neg_images
        print(f"Read in total {len(self.test_images)} test images, {len(self.test_pos_images)} positive, "
              f"{len(self.test_neg_images)} negative!")
        self.test_y = np.array([1] * len(self.test_pos_images) + [-1] * len(self.test_neg_images))
        assert len(self.test_images) == len(self.test_y)

        self.train_X = []
        for i, img in enumerate(self.train_images):
            # Haar feature
            haar = Haar(img, kernel_size=9)
            self.train_X.append(haar.get_gradient_image_vec())

            # flattened image
            # flat = np.asarray(img).flatten() / 255.
            # self.train_X.append(flat)
        self.train_X = np.asarray(self.train_X)
        self.train_X /= np.linalg.norm(self.train_X, axis=0)  # normalize along feature dimension
        print(f"{self.train_X.shape=}")

        self.test_X = []
        for img in self.test_images:
            # Haar feature
            haar = Haar(img, kernel_size=9)
            self.test_X.append(haar.get_gradient_image_vec())

            # flattened image
            # flat = np.asarray(img).flatten() / 255.
            # self.test_X.append(flat)
        self.test_X = np.asarray(self.test_X)
        self.test_X /= np.linalg.norm(self.test_X, axis=0)  # normalize along feature dimension
        print(f"{self.test_X.shape=}")

