import sys
import numpy as np
from decision_tree import DecisionTree
from evaluation import k_fold_cv
from nested_cv import nest_k_cv


def assess_tree_using_data(data):
    """
    Assess tree using the given data.

    Args:
        data: Instances + Class Labels, numpy array with shape (N,K)
    """
    classifier = DecisionTree()

    def print_results(results):
        print("Confusion Matrix: ")
        print(results['cm'].round(3))
        print("Precision: ")
        print(results['precision'].round(3))
        print("Recall: ")
        print(results['recall'].round(3))
        print("F1 score")
        print(results['f1'].round(3))
        print("Accuracy: ")
        print(results['accuracy'].round(3))
        print("Average depth: ")
        print(results['depth'])

    # ------- K FOLD TESTS -------
    results = k_fold_cv(data, classifier)
    print("\n------------------------------------")
    print("* Performance of the UNPRUNED tree\n")
    print_results(results)
    print()

    # # ------- NESTED K FOLD TESTS -------
    results = nest_k_cv(data, classifier)
    print("\n------------------------------------")
    print("* Performance of the PRUNED tree\n")
    print_results(results)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("\nLoading the data\n")
        clean_data = np.loadtxt('data/clean_dataset.txt')
        noisy_data = np.loadtxt('data/noisy_dataset.txt')

        print("-----------------------------------------------------------")
        print("1) Assessing tree on CLEAN data\n")
        assess_tree_using_data(clean_data)

        print("\n-----------------------------------------------------------")
        print("2) Assessing tree on NOISY data\n")
        assess_tree_using_data(noisy_data)
        print()

    elif len(sys.argv) == 2:
        print("Loading custom data")
        custom_data = np.loadtxt(sys.argv[1])
        assess_tree_using_data(custom_data)
    else:
        raise Exception("Too many arguments.\n"
                        "Usage python3 main.py [dataset_name].\n"
                        "Make sure the dataset is within the root folder.")
