import numpy as np


def _tp(y_test: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Return the True Positive
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        True Positive
    """
    return (y_pred & y_test).sum()


def _tn(y_test: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Return the True Negatives
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        True Negatives
    """
    return ((y_pred == 0) & (y_test == 0)).sum()


def _fp(y_test: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Return the False Positives
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        False Positives
    """
    return ((y_pred == 1) & (y_test == 0)).sum()


def _fn(y_test: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Return the False Negatives
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        False Negatives
    """
    return ((y_pred == 0) & (y_test == 1)).sum()


def _binary_precision(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Return the Precision
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        Precision
    """
    return _tp(y_test, y_pred) / (_tp(y_test, y_pred) + _fp(y_test, y_pred))


def _binary_recall(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Return the Recall
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        Recall
    """
    return _tp(y_test, y_pred) / (_tp(y_test, y_pred) + _fn(y_test, y_pred))


def _get_classes(data1, data2):
    """
    Get a list of the classes from both datasets
    Args:
        data1 (np.ndarray): Dataset 1 (N*1)
        data2 (np.ndarray): Dataset 2 (N*1)

    Returns:
        A np.ndarray of all the classes in datasets
    """
    return np.unique(np.concatenate((data1, data2)))


def recall(y_test: np.ndarray, y_pred: np.ndarray, return_mean=False) -> float:
    """
    Return the Recall for each class
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)
        return_mean (bool): If True, return mean

    Returns:
        Recall for each class (M*1) where M is the number of classes
    """
    to_return = np.array([_binary_recall(y_test == i, y_pred == i) for i in _get_classes(y_test, y_pred)])
    return to_return if not return_mean else to_return.mean()


def precision(y_test: np.ndarray, y_pred: np.ndarray, return_mean=False) -> np.ndarray:
    """
    Return the Precision for each class
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)
        return_mean (bool): If True, return mean

    Returns:
        Precision for each class (M*1) where M is the number of classes
    """
    to_return = np.array([_binary_precision(y_test == i, y_pred == i) for i in _get_classes(y_test, y_pred)])
    return to_return if not return_mean else to_return.mean()


def f1_score(y_test: np.ndarray, y_pred: np.ndarray, return_mean=False) -> np.ndarray:
    """
    Return the F1 score for each class
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)
        return_mean (bool): If True, return mean

    Returns:
        F1 Score for each class (M*1) where M is the number of classes
    """
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    to_return = (2 * prec * rec) / (prec + rec)
    return to_return if not return_mean else to_return.mean()


def confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Confusion Matrix

    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)
        normalise: Whether to normalise the data. Defaults to True.

    Returns:
        Return the confusion amtric as a matrix
    """
    n_classes = _get_classes(y_test, y_pred).shape[0]
    to_return = np.zeros((n_classes, n_classes))
    for p in range(n_classes):
        for a in range(n_classes):
            to_return[a, p] = _tp(y_test == a + 1, y_pred == p + 1)
    if normalise:
        for r in range(to_return.shape[0]):
            to_return[r] = to_return[r] / to_return[r].sum() + 1e-8
    return to_return


def accuracy(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Return the accuracy
    Args:
        y_test (np.ndarray): The test dataset (Nx1)
        y_pred (np.ndarray): The predicted dataset (Nx1)

    Returns:
        The accuracy
    """
    assert len(y_test) == len(y_pred)
    return np.sum(y_test == y_pred) / len(y_test) + 1e-8


def evaluate(test_db: np.ndarray, trained_tree) -> float:
    """
    Return the accuracy
    Args:
        test_db (np.ndarray): The test dataset (NxF) including all features
        trained_tree (Model): Trained Model instance

    Returns:
        The accuracy
    """
    y_pred = trained_tree.predict(test_db[:, :-1])
    y_test = test_db[:, -1]
    return accuracy(y_test, y_pred)


def evaluate_on(test_db: np.ndarray, trained_tree, metric, **kwargs):
    """
    Evaluate the metric on the test_db and the predictions from the trained_tree
    Args:
        test_db (np.ndarray): The test dataset (NxF) including all features
        trained_tree (Model): Trained Model instance
        metric: A function that returns either a float or a np.ndarray
        **kwargs: A dictionary of keywords to be passed to the metric function

    Returns:
        The value returned by the metric
    """
    # Evaluate the trained_tree on the test_db using the metric fnc passe in
    y_pred = trained_tree.predict(test_db[:, :-1])
    y_test = test_db[:, -1]
    return metric(y_test, y_pred, **kwargs)
