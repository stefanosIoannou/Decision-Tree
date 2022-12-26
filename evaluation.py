from typing import Dict
import numpy as np
from decision_tree import Model
from metrics import accuracy, confusion_matrix, evaluate_on, f1_score, precision, recall


def k_fold_cv(data: np.ndarray, model: Model, k: int = 10, verbose=False) -> Dict:
    """
    Performs k-fold cross validation on a model.

    Args:
        data (np.ndarray): Data to perform cross validation with.
        model (Model): Model to perform cross validation to.
        k (int): k-fold parameter
        verbose (boolean): True for more printing information

    Returns:
        results (Dict): Dictionary of all the metrics; Confusion Matrix, Accuracy, Precision, Recall, F1 Score and Average Maximum Depth.
    """
    n = data.shape[0]
    assert k > 0, 'k must be positive'
    split_len = n // k
    assert split_len != 0, 'Not enough data, or k is to large for the data'
    # print("Split Size: {}".format(split_len))
    np.random.shuffle(data)

    metrics = dict(cm=confusion_matrix, precision=precision, recall=recall,
                   f1=f1_score, accuracy=accuracy)
    
    # Store all results on the 'test' set
    results = dict()
    print(f"Begin {k}-fold cross validation")
    for s in range(0, n, split_len):
        model.reset_params()
        # Split the dataset

        test_db = data[s:s + split_len]

        mask = np.ones(data.shape[0], dtype=bool)
        mask[s:s + split_len] = False
        train_db = data[mask]
        model.fit(train_db[:, :-1], train_db[:, -1])

        # Evaluate the model
        for key, v in metrics.items():
            results[key] = results.get(key, 0) + (evaluate_on(test_db, model, v) / k)
        results['depth'] = results.get('depth', 0) + (model.compute_depth() / k)

        # print(f"Fold {int(s / split_len)}")
        if verbose:
            print(f"\n   Val from {s}:{s + split_len} (len: {len(test_db)})",
                  f"\n   Train (len: {len(train_db)})")
    print(f"End of {k}-fold cross validation")
    model.reset_params()
    return results
