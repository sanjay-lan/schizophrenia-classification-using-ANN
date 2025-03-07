# Schizophrenia-Classification-using-ANN

Overview

This project implements a classification model to predict schizophrenia diagnosis using an artificial neural network (ANN). The dataset is preprocessed, balanced, and trained using TensorFlow and Scikit-learn.

Installation

Ensure you have TensorFlow installed before running the code.

Process Overview

1. Importing Libraries

Essential libraries such as TensorFlow, NumPy, Pandas, Matplotlib, and Scikit-learn are imported to facilitate data processing, model training, and evaluation.

2. Data Loading and Preprocessing

The dataset is loaded and cleaned by removing unnecessary columns.

The Diagnosis column is balanced to ensure equal representation of classes.

Features are selected and split into independent (X) and dependent (y) variables.

The dataset is split into training and testing sets.

Feature scaling is applied to normalize the data.

3. Model Building and Training

An ANN model is initialized using the Sequential API.

The model consists of a dense hidden layer with ReLU activation and an output layer with a sigmoid activation function.

The model is compiled using the Adam optimizer and binary cross-entropy loss function.

Early stopping is implemented to prevent overfitting.

The model is trained using the training data with validation.

4. Model Evaluation

The model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score.

The Receiver Operating Characteristic (ROC) curve is plotted to analyze model performance.

A confusion matrix is generated to assess classification results.

5. Performance Metrics and Visualization

The ROC curve is plotted to compare true positive and false positive rates.

A bar chart is created to visualize performance metrics (accuracy, precision, recall, F1-score).

Confusion matrix components (TP, TN, FP, FN) are displayed using a bar plot.

Conclusion

This project successfully implements an ANN model to classify schizophrenia diagnosis. The model is evaluated using multiple performance metrics and visualized through various plots.
