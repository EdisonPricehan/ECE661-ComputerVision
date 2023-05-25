
import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
from sklearn import svm, metrics


def read_data(data_path: str):
    """
    Read data from csv file and return feature matrix and label vector
    :param data_path:
    :return: M x N feature matrix with M samples and N features, M x 1 label vector
    """
    with open(data_path, newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)))

    # last column is label, previous columns are features
    X = data[:, :-1]
    y = data[:, -1]
    # print(f"{X=}")
    # print(f"{y=}")
    return X, y


def classify_svm(training_data: str, test_data: str, save_model_name: str):
    """
    Fit the training dataset with SVM, then predict on test dataset and plot confusion matrix
    :param training_data:
    :param test_data:
    :param save_model_name:
    :return:
    """
    # read in train and test data
    X_train, y_train = read_data(training_data)
    X_test, y_test = read_data(test_data)

    # construct model and fit with training set
    cls = svm.SVC()
    cls.fit(X_train, y_train)

    # save model
    if save_model_name != '':
        pickle.dump(cls, open(save_model_name, 'wb'))

    # predict on the test set, print report and display confusion matrix
    predicted = cls.predict(X_test)
    print(
        f"Classification report for classifier {cls}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()


if __name__ == '__main__':
    # LBP features
    # training_data = 'train_lbp.csv'
    # test_data = 'test_lbp.csv'
    # svm_model = 'lbp_svm.pkl'

    # Gram random features from VGG19
    # training_data = 'HW7-Auxilliary/train_gram.csv'
    # test_data = 'HW7-Auxilliary/test_gram.csv'
    # svm_model = 'gram_svm.pkl'

    # Gram uniform features from VGG19
    training_data = 'HW7-Auxilliary/train_gram_uniform.csv'
    test_data = 'HW7-Auxilliary/test_gram_uniform.csv'
    svm_model = 'gram_uniform_svm.pkl'

    # AdaIN features from VGG19
    # training_data = 'HW7-Auxilliary/train_adain.csv'
    # test_data = 'HW7-Auxilliary/test_adain.csv'
    # svm_model = 'adain_svm.pkl'

    classify_svm(training_data, test_data, save_model_name=svm_model)


