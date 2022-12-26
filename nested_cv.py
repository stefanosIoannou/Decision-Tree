from typing import Dict, Tuple, List, Any
import numpy as np
from decision_tree import Model
from metrics import accuracy, confusion_matrix, evaluate_on, f1_score, precision, recall


def get_split(data, k) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Shuffle the data and split into k parts.

    Args:
        data: Instances + Class Labels, numpy array with shape (N,K)
        k: Amount of splits

    Returns:
        to_return (Tuple[np.ndarray, np.ndarray]) - the k folds.
    """
    data_len = len(data)
    split_len = data_len // k
    assert split_len != 0, 'Not enough data, or k is to large for the data'
    np.random.shuffle(data)
    to_return = []
    for s in range(0, data_len, split_len):
        test_db = data[s:s + split_len]

        mask = np.ones(data.shape[0], dtype=bool)
        mask[s:s + split_len] = False
        rest = data[mask]

        to_return.append((test_db, rest))

    return to_return


def nest_k_cv(data: np.ndarray, model: Model, k=10) -> Dict:
    """
    Perform nested cross validation.

    Args:
        data (np.ndarray): Instances + Class Labels, numpy array with shape (N,K)
        model (Model): Model to perform nested cross validation
        k (int): split amounts

    Returns:
        results (Dict): Dictionary of all the metrics; Confusion Matrix, Accuracy, Precision, Recall, F1 Score and Average Maximum Depth.
    """
    results = dict()

    print(f"Begin {k}-fold nested cross validation")
    for test, rest in get_split(data, k):

        # Metrics to use for evaluation on the 'test' set
        metrics = dict(cm=confusion_matrix, precision=precision, recall=recall,
                       f1=f1_score, accuracy=accuracy)
        # Store all results on the 'test' set
        for train, val in get_split(rest, k - 1):
            model.reset_params()
            model.fit(train[:, :-1], train[:, -1])
            model.post_order_pruning(train, val, model.root)

            # Evaluation on test set
            for key, v in metrics.items():
                results[key] = results.get(key, 0) + (evaluate_on(test, model, v) / (k*(k-1)))
            results['depth'] = results.get('depth', 0) + (model.compute_depth() / (k*(k-1)))
    return results
