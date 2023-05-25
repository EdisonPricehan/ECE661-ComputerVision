import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt

from autoencoder import get_acc


class DimReduc:
    def __init__(self, data_path: str, dim: int = 3):
        self.data_path = data_path

        self.image_list = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.png')]
        self.num_train = len(self.image_list)
        print(f"Train image num is {self.num_train}")

        self.p = dim
        assert self.p <= self.num_train, "Subspace dimension should not be greater than training image num!"

        # read all images, resize, and vectorize to form a nxm matrix (n is num_train, m is feature num)
        self.X, self.y = [], []
        for img_name in self.image_list:
            x, y = self.vectorize(img_name)
            self.X.append(x)
            self.y.append(y)
        self.X = np.asarray(self.X)

        self.X_mean = np.mean(self.X, axis=0)  # feature global mean vector
        # self.X = self.X - self.X_mean

        self.W_pca, self.X_pca = self._pca()  # subspace of PCA, W is 4096 x p matrix, X is num_train x p

        self.W_lda, self.X_lda = self._lda()  # same dimension as above

    @staticmethod
    def vectorize(img_path: str):
        img = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)[..., 0]  # use one gray channel
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        X = np.asarray(img, dtype=float) / 255.
        x = X.flatten()  # 4096 elements
        y = int(os.path.basename(img_path).split('_')[0])
        return x, y

    def _pca(self):
        X = self.X - self.X_mean  # num_train x 4096
        XXT = X @ X.T
        _, s, VT = np.linalg.svd(XXT)
        U = VT[:self.p]  # p x num_train
        W = X.T @ U.T  # first p eigenvectors of covariance matrix, 4096 x p
        W = W / np.linalg.norm(W, axis=0)  # normalize, 4096 x p
        X_reduc = W.T @ X.T  # p x num_train
        return W, X_reduc.T

    def _lda(self):
        """
        Yu and Yang's algorithm to solve Linear Discriminant Analysis (LDA) for dimensionality reduction
        :return:
        """
        all_labels = np.sort(np.unique(self.y))  # ascending labels
        labels_num = len(all_labels)
        print(f"{labels_num} labels in train data!")

        X_class_mean = []
        for label in all_labels:
            X_class_mean.append(np.mean(self.X[label == self.y], axis=0))
        X_class_mean = np.asarray(X_class_mean)  # 30 x 4096
        print(f"{X_class_mean.shape=}")
        X = X_class_mean - self.X_mean  # 30 x 4096

        # between-class scatter, but discard subspace where bcs is minimal
        XXT = X @ X.T  # 30 x 30
        _, s, VT = np.linalg.svd(XXT)
        print(f"{s=}")
        if np.min(s) < 1e-6:
            m = np.where(s < 1e-6)[0][0]
        else:
            m = labels_num
        print(f"Retained {m} eigenvectors!")
        Y = X.T @ VT[:m].T  # 4096 x m
        Db = np.diag(np.power(s[:m], -0.5))  # m x m
        Z = Y @ Db  # 4096 x m

        # within-class scatter matrix
        feature_num = self.X.shape[1]  # 4096
        Sw = np.zeros((feature_num, feature_num))  # within-class scatter
        for i, label in enumerate(all_labels):
            X_class = self.X[label == self.y] - X_class_mean[i]  # class_num x 4096
            Sw += (X_class.T @ X_class) / (self.y == label).sum()  # 4096 x 4096
        Sw /= labels_num  # 4096 x 4096

        ZTSwZ = Z.T @ Sw @ Z  # m x m
        eigval, eigvec = np.linalg.eig(ZTSwZ)
        U = eigvec[:, eigval.argsort()]  # ascending eigenvalues' eigenvectors, m x m

        W = (U[:, :self.p].T @ Z.T).T  # 4096 x p
        W = W / np.linalg.norm(W, axis=0)  # normalize, 4096 x p
        X = self.X - self.X_mean  # num_train x 4096
        X_reduc = W.T @ X.T  # p x num_train

        return W, X_reduc.T

    def predict(self, test_img_path: str, method: str = 'pca'):
        test_image_list = [os.path.join(test_img_path, f) for f in os.listdir(test_img_path) if f.endswith('.png')]
        num_test = len(test_image_list)
        print(f"Test image number is {num_test}!")

        predictions = []
        if method == 'pca':
            for test_img in test_image_list:
                x, y = self.vectorize(test_img)
                x = x - self.X_mean
                x_reduc = self.W_pca.T @ x
                pred = self.nn(self.X_pca, x_reduc)
                predictions.append(int(pred == y))
        elif method == 'lda':
            for test_img in test_image_list:
                x, y = self.vectorize(test_img)
                x = x - self.X_mean
                x_reduc = self.W_lda.T @ x
                pred = self.nn(self.X_lda, x_reduc)
                predictions.append(int(pred == y))
        else:
            print(f"Un-implemented method!")

        acc = sum(predictions) / num_test
        print(f"Prediction accuracy with {method} is {acc}")
        return acc

    def nn(self, X_reduc, x_reduc, metric: str = 'l1'):
        """
        1-Nearest Neighbour
        :param X_reduc:
        :param x_reduc:
        :param metric:
        :return: nearest neighbour's label
        """
        assert X_reduc.shape == (self.num_train, self.p), "Wrong dimension of reduced X"
        assert len(x_reduc) == self.p, "Wrong size of reduced x"

        def get_dist(v1, v2, metric: str):
            v1, v2 = np.asarray(v1), np.asarray(v2)
            if metric == 'l1':  # manhattan distance
                return np.sum(np.abs(v1 - v2))
            elif metric == 'l2':  # euclidean distance
                return np.sqrt(np.sum(np.power(v1 - v2, 2)))
            elif metric == 'linf':  # chessboard distance
                return np.max(np.abs(v1 - v2))
            else:
                print(f"Unrecognized metric!")
                return 0

        min_dist = np.inf
        min_label = -1
        for i in range(self.num_train):
            dist = get_dist(X_reduc[i], x_reduc, metric=metric)
            if dist < min_dist:
                min_dist = dist
                min_label = self.y[i]

        assert min_label != -1, "Distance wrong!"
        return min_label


def plot():
    """
    Plot accuracy of PCA and LDA and VAE
    :return:
    """
    p_vec = [3, 8, 16]

    if os.path.exists('pca_acc.pkl'):
        with open('pca_acc.pkl', 'rb') as f:
            pca_acc = pickle.load(f)
    else:
        pca_acc = []

    if os.path.exists('lda_acc.pkl'):
        with open('lda_acc.pkl', 'rb') as f:
            lda_acc = pickle.load(f)
    else:
        lda_acc = []

    if os.path.exists('vae_acc.pkl'):
        with open('vae_acc.pkl', 'rb') as f:
            vae_acc = pickle.load(f)
    else:
        vae_acc = []

    if len(pca_acc) == 0 or len(lda_acc) == 0:
        for p in p_vec:
            dr = DimReduc(training_data_path, dim=p)
            pca_acc.append(dr.predict(test_data_path, method='pca'))
            lda_acc.append(dr.predict(test_data_path, method='lda'))

        with open('pca_acc.pkl', 'wb') as f:
            pickle.dump(pca_acc, f)
        with open('lda_acc.pkl', 'wb') as f:
            pickle.dump(lda_acc, f)

    if len(vae_acc) == 0:
        for p in p_vec:
            acc = get_acc(p)
            print(f"VAE acc for {p=}: {acc}")
            vae_acc.append(acc)

        with open('vae_acc.pkl', 'wb') as f:
            pickle.dump(vae_acc, f)

    plt.figure()
    plt.plot(p_vec, pca_acc)
    plt.plot(p_vec, lda_acc)
    plt.plot(p_vec, vae_acc)
    plt.legend(['PCA acc', 'LDA acc', 'VAE acc'])
    plt.title('Classification Accuracy Comparison Among PCA, LDA and Autoencoder')
    plt.xlabel('Subspace Dimension')
    plt.ylabel('Accuracy')
    plt.xticks(p_vec)
    # plt.show()
    plt.savefig('face_acc_comparison.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    training_data_path = 'FaceRecognition/train'
    test_data_path = 'FaceRecognition/test'

    # dr = DimReduc(training_data_path, dim=3)
    # dr.predict(test_data_path, method='lda')

    plot()
