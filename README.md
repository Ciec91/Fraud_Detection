# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains credit card transactions made by European cardholders in September 2013. It includes data from two days, with a total of 284,807 transactions, out of which only 492 are fraudulent (0.172% of all transactions). This highly imbalanced dataset presents a challenge for traditional classification models.

## Dataset Description
The dataset used is the **Credit Card Fraud Detection Dataset** which was collected during a research collaboration between Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB). The dataset includes the following features:

- **V1, V2, … V28**: Principal components resulting from a PCA (Principal Component Analysis) transformation.
- **Time**: The time elapsed (in seconds) between each transaction and the first transaction in the dataset.
- **Amount**: The transaction amount.
- **Class**: The target variable that indicates whether a transaction is fraudulent (1) or not (0).

### Imbalance Issue
The dataset is highly imbalanced with a class distribution of 0.172% fraudulent transactions. This requires the application of techniques like under-sampling, over-sampling, and class weighting to improve the performance of the model.

## Techniques Used

### 1. Data Preprocessing
- **Handling Missing Data**: The dataset is checked for missing values, which are not present in this dataset.
- **Time Feature Removal**: The 'Time' feature is dropped as it is not relevant for the model in this case.
- **Class Mapping**: The target variable 'Class' is mapped from 0 and 1 to False and True for better clarity.

### 2. Data Visualization
- **Histograms**: The distribution of features is visualized using histograms to understand their range and skewness.

### 3. Sampling Techniques
- **Random Undersampling**: The dataset is balanced using random undersampling, reducing the majority class to match the number of fraud cases. This is done to improve the model's ability to learn from both classes.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: SMOTE is applied to create synthetic samples of the minority class (fraudulent transactions) to balance the dataset.

### 4. Model Training
- **Random Forest Classifier**: A Random Forest Classifier is trained on the resampled data to predict fraudulent transactions. The classifier is tuned with various parameters to ensure optimal performance.
- **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for the Random Forest model, improving its predictive performance.

### 5. Evaluation Metrics
- **Confusion Matrix**: Used to evaluate the performance of the classifier by showing the number of correct and incorrect predictions.
- **Classification Report**: Provides precision, recall, and F1 score for both classes (fraudulent and non-fraudulent).
- **ROC-AUC & Average Precision**: Measures the model's ability to discriminate between classes.
- **Precision-Recall Curve**: A plot that shows the trade-off between precision and recall, especially useful for imbalanced datasets.

### 6. Model Saving
- **Model Persistence**: The trained model is saved using `joblib` to be loaded and used for future predictions without retraining.

## Code Implementation

### Import Necessary Libraries

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import joblib
