# Decision Tree - Intro to Machine Learning Coursework 1
The submitted code is from the group that comprises of Stefanos Ioannou, Alfredo Musumeci, Janek Marczak, Oussama Rakye.
This README file is an introduction to the code and a brief explanation on how to run it.

## Code Structure
- **decision_tree.py** - This is the classifier class, and it contains all the necessary logic to build a Decision Tree.
- **evaluation.py** - The K-Fold Cross Validation used to evaluate the unpruned tree.
- **metrics.py** - All other evaluation metrics comprising accuracy, precision, recall, F1 and confusion matrix.
- **evaluation.ipynb** - A python notebook to visualize the metrics.
- **nested_cv.py** - The Nested K-Fold Cross Validation to evaluate the performance of the unpruned tree.
- **plotting.py** - The utilities needed for visualising the tree.
- **test.py** - Tests to make sure the code is working properly.
- **params.py** - Parameter settings to aid the visualization.
- **main.py** - The main logic where the decision trees are constructed and evaluated.

## How To
To run the code for this coursework navigate to the root folder, where the main.py file resides, and type in the terminal:

    python3 main.py

This will run it first using the clean data and then using the noisy data.
To run the code with a personalised dataset, move the dataset file into the root folder where main.py resides and write the following:

    python3 main.py [path_to_dataset]
