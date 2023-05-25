import numpy as np


def error_rate(y, y_pred) -> float:
    """
    error ratio
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return:
    """
    assert len(y) == len(y_pred)
    return (y != y_pred).sum() / len(y)


def error_vec(y, y_pred) -> np.array:
    """
    error vector with 1s (predicted correctly) and 0s (predicted incorrectly)
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return: 1d array with 1s and 0s
    """
    assert len(y) == len(y_pred)
    return np.asarray((y != y_pred).astype(int))


def error_mat(y, Y_pred) -> np.array:
    assert len(y) == Y_pred.shape[1]
    _error_mat = []
    for y_pred in Y_pred:
        _error_mat.append(y_pred != y)
    _error_mat = np.asarray(_error_mat, dtype=int)
    # print(f"{_error_mat=}")
    return _error_mat


def tpr(y, y_pred) -> float:
    """
    True positive rate
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return:
    """
    assert len(y) == len(y_pred)
    if 1 not in y:
        return 0
    return ((y == 1) & (y_pred == 1)).sum() / (y == 1).sum()


def tnr(y, y_pred) -> float:
    """
    True negative rate
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return:
    """
    assert len(y) == len(y_pred)
    if -1 not in y:
        return 0
    return ((y == -1) & (y_pred == -1)).sum() / (y == -1).sum()


def fpr(y, y_pred) -> float:
    """
    False positive rate
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return:
    """
    assert len(y) == len(y_pred)
    if -1 not in y:
        return 0
    return ((y == -1) & (y_pred == 1)).sum() / (y == -1).sum()


def fnr(y, y_pred) -> float:
    """
    False negative rate
    :param y: true labels, 1d array of 1s and -1s
    :param y_pred: predicted labels, 1d array of 1s and -1s
    :return:
    """
    assert len(y) == len(y_pred)
    if 1 not in y:
        return 0
    return ((y == 1) & (y_pred == -1)).sum() / (y == 1).sum()


def polarize(vec, idx: int):
    """
    Positively polarize an ascending order (feature or score) vector by idx to create fake prediction labels vector
    :param vec: ascending order 1d vector
    :param idx: index of vec
    :return: 1d array of 1s and -1s
    """
    assert 0 <= idx < len(vec)
    return np.array([-1 if i < idx else 1 for i in range(len(vec))])  # assume vec is sorted in ascending order


def polarize_to_mat(vec):
    assert len(vec) > 1
    polarized_mat = [polarize(vec, 0)]
    for i in range(1, len(vec)):
        if vec[i] > vec[i - 1]:
            polarized_mat.append(polarize(vec, i))
        else:
            polarized_mat.append(polarized_mat[-1])
    polarized_mat = np.asarray(polarized_mat)
    assert polarized_mat.shape[0] == polarized_mat.shape[1]
    return polarized_mat


def polarize_by_value(vec, val: float):
    """
    Positively polarize ascending vector with contiguous identical value
    :param vec: ascending order 1d vector
    :param val: stump value
    :return: 1d array of 1s and -1s
    """
    assert val in vec
    return polarize(vec, np.where(vec == val)[0][0])


if __name__ == '__main__':
    y = np.array([1, 1, 1, -1, -1, -1])
    y_pred = np.array([1, -1, 1, -1, -1, -1])
    _tpr = tpr(y, y_pred)
    print(f"{_tpr=}")

    x = np.array([1, 2, 3])
    y_pred = polarize(x, 1)
    print(f"{y_pred=}")


