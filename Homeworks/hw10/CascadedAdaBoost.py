import pickle
import os
import matplotlib.pyplot as plt

from AdaBoost import AdaBoost
from DataLoader import DataLoader
from Utils import *


class CascadedAdaBoost:
    def __init__(self, train_path: str, test_path: str,
                 max_stage: int = 10, max_iter_per_stage: int = 30, min_tpr: float = 0.99, max_fpr: float = 0.3):
        """
        Implementation of Cascaded AdaBoost with preferences on specific TPR and FPR
        :param train_path: training dataset path
        :param test_path: test dataset path
        :param max_stage: maximum number of stages (AdaBoosts)
        :param max_iter_per_stage: maximum number of weak classifier for each AdaBoost
        :param min_tpr: minimum allowable true positive rate for each AdaBoost
        :param max_fpr: maximum allowable false positive rate for each AdaBoost
        """
        self.data_loader = DataLoader(train_path, test_path)  # handles data loading, feature extraction and transform
        self.K = max_stage
        self.N = max_iter_per_stage
        self.min_tpr = min_tpr
        self.max_fpr = max_fpr

        self.stages = []  # stores list of mapping from decision threshold to each AdaBoost classifier

    def train(self):
        """
        Main function for training the Cascaded AdaBoost
        Data samples that are deemed positive will be fed into the next stage
        Stage adaboost that meets at least TPR will be recorded as a valid stage, FPR is not forced to meet with the
        restriction of max iter per stage
        :return:
        """
        X, y = self.data_loader.train_X, self.data_loader.train_y
        for stage in range(self.K):
            print(f"{stage=}")
            adaboost = AdaBoost(X, y)
            need_to_break = False
            for i in range(self.N):
                if need_to_break:
                    break
                adaboost.step()
                scores = adaboost.scores_vec()
                scores_indices = np.argsort(scores)  # ascending order
                scores_sorted = scores[scores_indices]
                y_sorted = y[scores_indices]
                idx = len(scores_sorted) - 1
                while idx >= 0:
                    y_pred = polarize(scores_sorted, idx)
                    _tpr = tpr(y_sorted, y_pred)
                    if _tpr < self.min_tpr:
                        idx -= 1
                        continue

                    _fpr = fpr(y_sorted, y_pred)
                    print(f"TPR met {_tpr}, FPR {_fpr}!")

                    if _fpr < self.max_fpr or i == self.N - 1:
                        need_to_break = True
                        thr = scores_sorted[idx]
                        self.stages.append([thr, adaboost])
                        print(f"Added current adaboost, tpr: {_tpr}, fpr: {_fpr}, threshold: {thr}")

                        # update new dataset
                        X = X[scores_indices[idx:]]
                        y = y[scores_indices[idx:]]
                    break

        print(f"All {len(self.stages)} stages ended!")
        for i in range(len(self.stages)):
            print(f"Stage {i + 1} has {len(self.stages[i][1].weak_classifiers)} decision stumps!")

        with open('all_stages.pkl', 'wb') as f:
            pickle.dump(self.stages, f)
            print(f"Saved pickle of all stages")

    def test(self):
        """
        Main function to test the trained Cascaded AdaBoost
        The negatively predicted label will be set onto the overall prediction vector when sequentially traversing
        all stages, we can decide how many first k stages we need for some task according to all the performance metrics
        up to that stage
        :return:
        """
        if len(self.stages) == 0:
            print(f"No adaboost stages to use, check pickle file!")

        if os.path.exists('all_stages.pkl'):
            with open('all_stages.pkl', 'rb') as f:
                self.stages = pickle.load(f)
            print(f"Loaded all stages, start testing!")
        else:
            print("Get stages first before test!")
            return

        X, y = self.data_loader.test_X, self.data_loader.test_y
        y_pred = np.ones_like(y)
        confusion_list = []
        for i, (thr, adaboost) in enumerate(self.stages):
            y_pred_cur = adaboost.classify_vec(X, threshold=thr)
            y_pred[y_pred_cur == -1] = -1
            _tpr = tpr(y, y_pred)
            _tnr = tnr(y, y_pred)
            _fpr = fpr(y, y_pred)
            _fnr = fnr(y, y_pred)
            confusion_list.append([_tpr, _tnr, _fpr, _fnr])
            print(f"Stage {i + 1}: TPR, TNR, FPR, FNR = {confusion_list[-1]}")

        with open('confusion_list.pkl', 'wb') as f:
            pickle.dump(confusion_list, f)
            print(f"Saved pickle of confusion list of all stages!")

    @staticmethod
    def plot():
        if os.path.exists('confusion_list.pkl'):
            with open('confusion_list.pkl', 'rb') as f:
                confusion_list = pickle.load(f)

            plt.figure()
            stages = np.array(range(len(confusion_list))) + 1
            fprs = [rates[2] for rates in confusion_list]
            fnrs = [rates[3] for rates in confusion_list]
            plt.plot(stages, fprs)
            plt.plot(stages, fnrs)
            plt.legend(['FPR', 'FNR'])
            plt.xlabel('Stage')
            plt.ylabel('Rates')
            plt.title('FPR and FNR up to stages on test set')
            # plt.show()
            plt.savefig('FPR-FNR.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        else:
            print(f"Need to get confusion matrix list first by running test!")


if __name__ == '__main__':
    train_image_path = 'CarDetection/train'
    test_image_path = 'CarDetection/test'

    cascade = CascadedAdaBoost(train_image_path, test_image_path, max_stage=10, max_iter_per_stage=20,
                               min_tpr=0.99, max_fpr=0.3)
    # cascade.train()
    # cascade.test()
    cascade.plot()

