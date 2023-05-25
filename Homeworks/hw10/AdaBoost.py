import numpy as np
from tqdm import tqdm
import pickle

from DataLoader import DataLoader
from DecisionStump import DecisionStump
from Utils import *


class AdaBoost:
    def __init__(self, X, y):
        """
        Implementation of AdaBoost classifier
        :param X: n x m matrix, n is image number, m is feature number
        :param y: 1d array with n labels (1 and -1)
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.num_img, self.num_feature = self.X.shape
        print(f"AdaBoost images: {self.num_img}, features: {self.num_feature}")

        self.data_weights = np.array([1 / self.num_img] * self.num_img)  # uniform weights initially
        self.weak_classifier_weights = []
        self.weak_classifiers = []

        self.X_sorted, self.Y_sorted, self.W_sorted, self.sorted_indices = self._sort()



    def _sort(self):
        """
        sort all columns of X in ascending order to facilitate searching for weak classifier
        all labels and weights are accordingly reordered and stacked horizontally to form n x m matrices
        :return:
        """
        sorted_indices = []
        X_sorted = np.zeros_like(self.X)
        Y_sorted = np.zeros_like(self.X)
        W_sorted = np.zeros_like(self.X)
        for f in range(self.num_feature):
            f_indices = np.argsort(self.X[:, f])
            sorted_indices.append(f_indices)
            X_sorted[:, f] = self.X[f_indices, f]
            Y_sorted[:, f] = self.y[f_indices]
            W_sorted[:, f] = self.data_weights[f_indices]  # maybe uniformly init directly
        return X_sorted, Y_sorted, W_sorted, sorted_indices

    def _make_pred_error_matrix3d(self):
        """
        3d matrix preparation for 3d array multiplication with 2d array, not used since it requires too much memory
        :return:
        """
        error_pred_mat_pos = np.zeros_like(self.X, dtype=bool)
        error_pred_mat3d_pos = np.repeat(error_pred_mat_pos[..., np.newaxis], self.num_img, axis=2)
        print(f"Initializing error rate matrix ...")
        for f in tqdm(range(self.num_feature)):
            y_pred = np.array([1] * self.num_img)
            for n in range(self.num_img):
                if n != 0 and self.X_sorted[n, f] > self.X_sorted[n - 1, f]:
                    y_pred[:n] = -1
                _error_vec = error_vec(self.Y_sorted[:, f], y_pred).astype(bool)
                error_pred_mat3d_pos[n, f] = _error_vec
        error_pred_mat3d_neg = ~error_pred_mat3d_pos
        return error_pred_mat3d_pos, error_pred_mat3d_neg

    def get_best_decision_stump(self):
        """
        Not used since this requires too much memory (> 32 Gb), although should be fast when finding stumps
        :return:
        """
        error_mat3d_pos, error_mat3d_neg = self._make_pred_error_matrix3d()
        weighted_error_mat_pos = np.einsum('ijk,kj->ij', error_mat3d_pos, self.W_sorted)
        weighted_error_mat_neg = np.einsum('ijk,kj->ij', error_mat3d_neg, self.W_sorted)
        weighted_error_mat_stacked = np.asarray([weighted_error_mat_neg, weighted_error_mat_pos])
        p, n, f = np.unravel_index(np.argmin(weighted_error_mat_stacked, axis=None), weighted_error_mat_stacked.shape)
        thr = self.X_sorted[n, f]
        ds = DecisionStump(f=f, threshold=thr, polarity=p)
        self.weak_classifiers.append(ds)
        print(f"Added one decision stump with f: {f}, t: {thr}, p: {p}")
        return ds

    def get_best_decision_stump_slow(self):
        """
        Not used, slowest method
        :return:
        """
        weighted_error_mat_pos = np.zeros_like(self.X)
        weighted_error_mat_neg = np.zeros_like(self.X)
        for f in tqdm(range(self.num_feature), desc='Traversing all data ...'):
            _error_mat = error_mat(self.Y_sorted[:, f], polarize_to_mat(self.X_sorted[:, f]))
            weighted_error_mat_pos[:, f] = _error_mat @ self.W_sorted[:, f]
            weighted_error_mat_neg[:, f] = (1 - _error_mat) @ self.W_sorted[:, f]
        weighted_error_mat_stacked = np.asarray([weighted_error_mat_neg, weighted_error_mat_pos])
        p, n, f = np.unravel_index(np.argmin(weighted_error_mat_stacked, axis=None), weighted_error_mat_stacked.shape)
        thr = self.X_sorted[n, f]
        ds = DecisionStump(f=f, threshold=thr, polarity=p)
        self.weak_classifiers.append(ds)
        print(f"Added one decision stump with f: {f}, t: {thr}, p: {p}")
        return ds

    def get_best_decision_stump_medium(self):
        """
        Adopted, medium speed with low memory usage compared to other two methods
        :return:
        """
        min_error_f = None
        min_error_thr = None
        min_error_p = True
        min_weighted_error = np.inf
        for f in tqdm(range(self.num_feature), desc='Searching for best stump ...'):
            y_pred = np.array([1] * self.num_img)
            for n in range(self.num_img):
                if n != 0 and self.X_sorted[n, f] > self.X_sorted[n - 1, f]:
                    y_pred[:n] = -1
                _error_vec = error_vec(self.Y_sorted[:, f], y_pred)
                weighted_error_pos = np.inner(_error_vec, self.W_sorted[:, f].ravel())
                if weighted_error_pos < min_weighted_error:
                    min_error_f = f
                    min_error_thr = self.X_sorted[n, f]
                    min_error_p = True
                    min_weighted_error = weighted_error_pos
                weighted_error_neg = np.inner(1 - _error_vec, self.W_sorted[:, f].ravel())
                if weighted_error_neg < min_weighted_error:
                    min_error_f = f
                    min_error_thr = self.X_sorted[n, f]
                    min_error_p = False
                    min_weighted_error = weighted_error_neg
        ds = DecisionStump(f=min_error_f, threshold=min_error_thr, polarity=min_error_p)
        self.weak_classifiers.append(ds)
        print(f"Added one decision stump with f: {min_error_f}, t: {min_error_thr}, p: {min_error_p}")
        return ds

    def _update_trust_factor(self):
        """
        Update trust factor for newly added stump based on its misclassification error
        :return:
        """
        ds = self.weak_classifiers[-1]
        h = ds.predict_mat(self.X)
        error = 0.5 * np.inner(np.abs(h - self.y), self.data_weights)
        # print(f"{error=}")
        if error == 0:
            print(f"This stump is the one!")
            error = 1e-12
        alpha = 0.5 * np.log((1 - error) / error)
        self.weak_classifier_weights.append(alpha)
        # print(f"{self.weak_classifier_weights=}")

    def _update_data_weights(self):
        """
        Update training sample weights, increase if that sample is misclassified, decrease otherwise to let the
        next decision stump focuses more the misclassified samples at this iteration
        :return:
        """
        y_pred = self.weak_classifiers[-1].predict_mat(self.X)
        alpha = self.weak_classifier_weights[-1]
        data_weights = self.data_weights * np.exp(-alpha * self.y * y_pred)
        self.data_weights = data_weights / np.sum(data_weights)  # normalize
        # print(f"{self.data_weights=}")
        for f in range(self.num_feature):
            self.W_sorted[:, f] = self.data_weights[self.sorted_indices[f]]

    def step(self):
        """
        Main function to add a decision stump
        needs to be called from external loop since this class mainly works with Cascaded AdaBoost, so itself has no
        control on how many weak classifiers needed or what performance needs to be met
        :return:
        """
        print(f"AdaBoost stepping ...")

        # self.get_best_decision_stump()
        # self.get_best_decision_stump_slow()
        self.get_best_decision_stump_medium()

        self._update_trust_factor()

        self._update_data_weights()

        _tpr, _fpr, _tnr, _fnr = self.metrics(threshold=0)
        print(f"Current {len(self.weak_classifiers)} classifiers, tpr: {_tpr}, fpr: {_fpr}")

    def score(self, x):
        """
        Score of weighted predictions from all weak classifiers
        :param x: data vector
        :return: scalar that can be arbitrarily thresholded to give out binary prediction
        """
        if len(self.weak_classifiers) == 0:
            print(f"No weak classifiers!")
            return 0
        _score = 0
        for alpha, ds in zip(self.weak_classifier_weights, self.weak_classifiers):
            h = ds.predict_data_vec(x)
            _score += alpha * h
        return _score

    def scores_vec(self):
        """
        Scores of all training samples
        :return: scalar vector
        """
        if len(self.weak_classifiers) == 0:
            print(f"No weak classifiers!")
            return None
        return np.array([self.score(self.X[i]) for i in range(self.num_img)])

    def metrics(self, X=None, threshold: float = 0):
        """
        Four performance metrics on the input data samples using all weak classifiers
        :param X: input n x m data matrix, set to training samples if not specified
        :param threshold: polarize score to 1 or -1
        :return: (TPR, FPR, TNR, FNR)
        """
        if len(self.weak_classifiers) == 0:
            print(f"No weak classifiers!")
            return None, None, None, None

        if X is None:
            X = self.X

        y_pred = np.array([self.classify(X[i], threshold) for i in range(X.shape[0])])
        _tpr = tpr(self.y, y_pred)
        _fpr = fpr(self.y, y_pred)
        _tnr = tnr(self.y, y_pred)
        _fnr = fnr(self.y, y_pred)
        # assert _tpr + _fnr == 1
        # assert _fpr + _tnr == 1
        return _tpr, _fpr, _tnr, _fnr

    def classify(self, x, threshold: float = 0):
        """
        Binary classify a data vector to 1 or -1
        :param x: m-dimensional data vector
        :param threshold: scalar
        :return: 1 or -1
        """
        return 1 if self.score(x) > threshold else -1

    def classify_vec(self, X, threshold: float = 0):
        """
        Classify all data samples by some threshold
        :param X: n x m data sample matrix
        :param threshold: scalar
        :return: vector of 1s and -1s
        """
        return np.array([self.classify(X[i], threshold) for i in range(X.shape[0])])


if __name__ == '__main__':
    pass







