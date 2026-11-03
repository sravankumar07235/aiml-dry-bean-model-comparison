**Dry Bean Classification - Machine Learning Model Comparison**

**Problem Statement:**
The objective of this project is to classify dry beans into different varieties using morphological features and compare the performance of five machine learning classification algorithms.

The models evaluated are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

**Dataset Description:**
The Dry Bean Dataset from the UCI Machine Learning Repository is used for this project.

The original dataset contains 13,611 observations, 16 input features, and 7 dry bean classes.

During data preprocessing, 68 duplicate records were identified and removed. After duplicate removal, 13,543 observations remained for model development.

The target variable is Class, representing the dry bean variety.

The dataset was divided into training and testing sets using an 80:20 stratified split with random_state=42.

**Models Used**
The following classification algorithms were implemented using scikit-learn:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine
  
StandardScaler was included in the pipelines for Logistic Regression, KNN, and SVM.

**Evaluation Metrics**

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
  
Weighted averaging was used for Precision, Recall, and F1 Score. Multiclass One-vs-Rest (OvR) weighted AUC was used for AUC evaluation.

**Model Performance Comparison**
<img width="789" height="135" alt="image" src="https://github.com/user-attachments/assets/7c8c23d6-24b0-4d5a-879b-38023a5289f2" />

**Model Performance Observations**

**Logistic Regression**
Logistic Regression achieved an accuracy of 0.9192 and an AUC of 0.9934. The model produced strong overall multiclass classification performance.

**Decision Tree**
The Decision Tree achieved an accuracy of 0.8966 and an F1 Score of 0.8964. Its overall performance was lower than Logistic Regression, KNN, and SVM, with an AUC of 0.9363.

**K-Nearest Neighbors**
K-Nearest Neighbors: KNN achieved an accuracy of 0.9155 and an F1 Score of 0.9157. It performed strongly overall, although its Accuracy, F1 Score, and MCC were slightly lower than those of Logistic Regression and SVM.

**Gaussian Naive Bayes**
Gaussian Naive Bayes: Gaussian Naive Bayes achieved an accuracy of 0.7630 and an F1 Score of 0.7607, giving it the lowest Accuracy and F1 Score among the five evaluated models. However, its AUC remained relatively high at 0.9644.

**Support Vector Machine**
SVM achieved an accuracy of 0.9210, F1 Score of 0.9211, and MCC of 0.9045. Among the evaluated models, SVM achieved the highest accuracy, F1 Score, and MCC.


**Overall Model Comparison**

Based on the experimental results, Support Vector Machine provided the strongest overall classification performance based on Accuracy, F1 Score, and MCC.

Logistic Regression achieved a marginally higher AUC in this experiment.

**Streamlit Application**
The Streamlit application allows the user to:

- Upload a CSV test dataset
- Select one of the five trained models
- Evaluate the selected saved model
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- View the confusion matrix
- View the classification report
  
The Streamlit application loads previously trained models saved using Joblib. Models are not retrained when a test dataset is uploaded.

**Repository Contents**
- ML_Assignment_2_Dry_Bean_Classification.ipynb - model development and evaluation notebook
- app.py - Streamlit application
- test_data.csv - test dataset
- requirements.txt - Python dependencies
- logistic_regression.joblib - trained Logistic Regression model
- decision_tree.joblib - trained Decision Tree model
- knn.joblib - trained KNN model
- naive_bayes.joblib - trained Gaussian Naive Bayes model
- svm.joblib - trained SVM model

**Live Streamlit Application link**
https://sravankumar07235-aiml-dry-bean-model-comparison-app-guokpe.streamlit.app/

**GitHub Repository link**
https://github.com/sravankumar07235/aiml-dry-bean-model-comparison

**Data set Source:** UCI Machine Learning Repository - Dry Bean Dataset
