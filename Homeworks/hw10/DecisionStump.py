import numpy as np


class DecisionStump:
    def __init__(self, f: int, threshold: float, polarity: bool):
        """
        Simple implementation of a weak classifier (decision stump) that focuses only on one feature
        :param f: stump feature index
        :param threshold: stump threshold value
        :param polarity: stump polarity
        """
        self.f = f
        self.t = threshold
        self.p = polarity

    def predict(self, x: float) -> int:
        """
        Predict a single scalar, assuming it is the value of the stump feature
        :param x:
        :return: 1 or -1
        """
        y_pred = 1 if x >= self.t else -1
        return y_pred if self.p else -y_pred

    def predict_feature_vec(self, features: np.array) -> np.array:
        """
        Predict some feature column of all data samples
        :param features:
        :return: 1d array of 1s and -1s
        """
        return np.array([self.predict(value) for value in features])

    def predict_data_vec(self, x: np.array) -> int:
        """
        Predict some data row of all features, only the value at the same feature column takes effect
        :param x:
        :return: 1 or -1
        """
        return self.predict(x[self.f])

    def predict_mat(self, X):
        """
        Predict the overall data matrix, only the values of the same feature column takes effect
        :param X:
        :return: 1d array of 1s and -1s
        """
        return self.predict_feature_vec(X[:, self.f])

