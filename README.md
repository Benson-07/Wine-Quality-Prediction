# Wine-Quality-Prediction
This project aims to predict the quality of wine using a Random Forest Regressor. The dataset contains various features of wine and the target variable is the quality of the wine. We use machine learning techniques to build and tune a Random Forest model to predict wine quality.

1)Project Overview
This project involves the following steps:

Data Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
Model Building: Training a Random Forest Regressor model.
Model Evaluation: Assessing the model performance using Mean Squared Error (MSE).
Hyperparameter Tuning: Using Randomized Search and Grid Search to find the best hyperparameters for the model.

2)Installation
To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
matplotlib
scikit-learn

You can install the required libraries using pip:
pip install pandas numpy matplotlib scikit-learn

3)Usage
Prepare Your Dataset: Ensure your dataset is in the format used in the script. The dataset should have a type column (categorical) and other numerical features, with a quality column as the target variable.

Run the Script:

This script will:

Load and preprocess the dataset.
Train a Random Forest Regressor model.
Perform hyperparameter tuning using Randomized Search and Grid Search.
Evaluate and print the performance metrics of the model


4)Model Evaluation
The script evaluates the model performance using Mean Squared Error (MSE). It prints the following metrics:

MSE for the initial Random Forest model.
Best MSE from Randomized Search.
Best MSE from Grid Search.
Hyperparameter Tuning
The script performs hyperparameter tuning using two methods:

Randomized Search: Samples a fixed number of parameter settings from the specified ranges.
Grid Search: Exhaustively searches through a specified parameter grid.
The parameters tuned include:

n_estimators: Number of trees in the forest.
max_depth: Maximum depth of the trees.
min_samples_split: Minimum number of samples required to split an internal node.
min_samples_leaf: Minimum number of samples required to be at a leaf node.
Save the provided code into a Python script file, e.g., wine_quality_prediction.py, and run it
