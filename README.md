# Loan Approval Prediction using Machine Learning

## Overview
This project focuses on building a supervised machine learning model to predict whether a loan application should be **Approved** or **Rejected** based on applicant details.

Banks receive thousands of loan applications every month, and manual processing is time-consuming and error-prone. This project demonstrates how data-driven decision-making can improve efficiency and accuracy in loan approval systems.

---

## Problem Statement
HDFC Bank receives a large number of loan applications regularly.  
Manual evaluation of these applications:
- Takes significant time
- Is prone to human bias and errors

The goal is to automate this process using machine learning classification models.

---

## Objective
- Predict **Loan Status** (Approved / Rejected)
- Use applicant information such as:
  - Income
  - Loan Amount
  - Credit History
  - Employment Status
  - Property Area
- Compare multiple ML models and evaluate their performance
- Improve decision-making efficiency using data analysis

---

## Dataset
- Source: **Kaggle**
- Contains applicant demographic and financial details
- Includes both numerical and categorical features

---

## Data Preprocessing
The following preprocessing steps were applied:

- **Handling Missing Values**
  - Numerical features filled using mean/median
  - Categorical features filled using mode

- **Encoding Categorical Variables**
  - Label Encoding for binary features (Gender, Married, Education, Self_Employed)
  - One-Hot Encoding for multi-category features (Property_Area)

- **Feature Scaling**
  - Standardization using `StandardScaler` for numerical variables like:
    - ApplicantIncome
    - LoanAmount

- **Train-Test Split**
  - 80% Training
  - 20% Testing

---

## Models Implemented
The following supervised learning models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

---

## Model Evaluation
Performance was evaluated using:
- Accuracy
- F1 Score
- ROC metrics
- Cross-Validation F1 Score

### Results Summary

| Model               | CV F1 Score (Mean) | Test Accuracy |
|--------------------|-------------------|---------------|
| Logistic Regression | 0.8691            | 0.90          |
| Decision Tree       | 0.8528            | 0.87          |
| Random Forest       | 0.8645            | 0.90          |

---

## Conclusion
Among all models tested, **Logistic Regression** performed the best overall with:
- Higher F1 Score
- Strong generalization performance

This shows that simpler models can sometimes outperform more complex ones when the data is well-preprocessed.

---

## Tools & Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## What I Learned
- End-to-end machine learning workflow
- Data preprocessing techniques
- Feature encoding and scaling
- Training and comparing multiple ML models
- Using evaluation metrics like F1 Score and ROC
- Applying ML to real-world financial problems

---

## Motivation
This project was built to develop **research-oriented data analysis and machine learning skills** and to understand how ML models are applied in real-world banking and finance scenarios.
