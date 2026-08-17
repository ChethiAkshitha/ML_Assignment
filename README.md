## 1. Problem Statement

The objective of this assignment is to develop and evaluate multiple machine learning classification models on the chosen dataset and compare their performance using different evaluation metrics.
The implemented machine learning models are Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest (Ensemble). The models are evaluated using Accuracy, AUC, Precision, Recall,- F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

The dataset used for this assignment is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The dataset contains 569 instances and 30 numerical features. The target variable is diagnosis, with two classes:
- Benign
- Malignant
The ID column was removed before model training. The data was divided into training and testing sets, and StandardScaler was used for feature scaling where required.

## 3. Github Repository Link
https://github.com/ChethiAkshitha/ML_Assignment

## 3. Models Used

### Logistic Regression
Logistic Regression was used as a baseline classification model.
### Decision Tree
Decision Tree is a tree-based classification model that makes predictions using feature-based decision rules.
### kNN
K-Nearest Neighbors is a distance-based classification algorithm that uses the nearest training samples for prediction.
### Naive Bayes
Gaussian Naive Bayes is a probabilistic classification algorithm used for the classification task.
### Random Forest
Random Forest is an ensemble learning model consisting of multiple decision trees.

| ML Model Name            | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   96.49% | 99.60% |    97.50% | 92.86% | 95.12% | 92.45% |
| Decision Tree            |   92.98% | 92.46% |    90.48% | 90.48% | 90.48% | 84.92% |
| kNN                      |   95.61% | 98.23% |    97.44% | 90.48% | 93.83% | 90.58% |
| Naive Bayes              |   92.11% | 98.91% |    92.31% | 85.71% | 88.89% | 82.92% |
| Random Forest (Ensemble) |   97.37% | 99.29% |   100.00% | 92.86% | 96.30% | 94.42% |

| ML Model Name                        | Observation about model performance                                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**              | Performed very well, with high Accuracy and AUC. It also achieved high Precision, Recall, F1 Score, and MCC, making it a strong baseline model.                                     |
| **Decision Tree**                    | Showed comparatively lower performance than the other models, with lower Accuracy, AUC, F1 Score, and MCC.                                                                          |
| **kNN**                              | Performed well with high Accuracy and AUC. It also achieved high Precision and F1 Score. Feature scaling was used because kNN is distance-based.                                    |
| **Naive Bayes**                      | Achieved a high AUC, but its Accuracy, Recall, F1 Score, and MCC were comparatively lower than the stronger-performing models.                                                      |
| **Random Forest (Ensemble)**         | Achieved the highest Accuracy, Precision, F1 Score, and MCC. It provided the strongest overall performance on the chosen test dataset.                                              |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it achieved the best overall combination of Accuracy (97.37%), AUC (99.29%), Precision (100%), Recall (92.86%), F1 Score (96.30%), and MCC (94.42%). |
