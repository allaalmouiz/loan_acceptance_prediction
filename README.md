# Loan Acceptance Prediction Challenge - Module 1

## Overview

This project implements a machine learning model to predict whether a loan application will be accepted or rejected based on applicant profile features. It's a binary classification problem that walks through the complete Data Science lifecycle from data preparation to model evaluation.

**Submitted by:** Alaa Almouiz F. Moh.  
**ID Number:** S2026_176  
**Track:** Machine Learning  
**Organization:** ZAKA ©

## Problem Statement

In a banking context, when someone applies for a loan, how likely are they to get approved based on their profile? This notebook addresses this question by building a predictive model that classifies loan applications as approved or rejected.

## Dataset

The project uses a loan dataset with the following features:

| Column | Description |
|--------|-------------|
| `Loan_ID` | Unique identifier for each loan application |
| `Gender` | Applicant's gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Education level (Graduate/Not Graduate) |
| `Self_Employed` | Employment status |
| `ApplicantIncome` | Applicant's monthly income |
| `CoapplicantIncome` | Co-applicant's monthly income |
| `LoanAmount` | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Loan term (in months) |
| `Credit_History` | Credit history meets guidelines (binary) |
| `Property_Area` | Property location (Urban/Rural/Semi-Urban) |
| `Loan_Status` | **Target variable** - Loan approved (Y/N) |


## Project Structure

### 1. **Data Preparation**
   - Clone dataset from GitHub repository
   - Import necessary Python libraries (NumPy, Pandas, Matplotlib, Scikit-learn)
   - Load training and testing datasets

### 2. **Exploratory Data Analysis (EDA)**
   - Display dataset samples and shape
   - Identify categorical and numerical columns
   - Analyze data types and information
   - Visualize distributions and relationships

### 3. **Data Cleaning**
   - Handle missing values:
     - Categorical features: filled with mode
     - Numerical features: filled with mean
   - Remove unnecessary columns (Loan_ID)
   - Validate null values are resolved

### 4. **Data Encoding & Transformation**
   - Encode categorical variables:
     - Gender: {male: 0, female: 1}
     - Married: {no: 0, yes: 1}
     - Property_Area: {urban: 0, rural: 1, semiurban: 2}
     - Education: {not_graduate: 0, graduate: 1}
     - Self_Employed: {no: 0, yes: 1}
     - Dependents: {0: 0, 1: 1, 2: 2, 3+: 3}
     - Loan_Status: {n: 0, y: 1}
   - Convert categorical columns to category dtype
   - Normalize numerical features using StandardScaler

### 5. **Visualization**
   - Loan status distribution (countplot)
   - Categorical feature distributions (pie charts)
   - Conditional analysis of loan status vs categorical variables
   - Numerical feature distributions (histograms and boxplots)
   - Correlation heatmap for numerical variables

### 6. **Model Building**
   - Train Linear Regression model for binary classification
   - Apply 0.5 threshold to convert continuous predictions to binary outcomes
   - Standardize input features before model training

### 7. **Model Evaluation**
   - Accuracy: **81.19%**
   - Confusion Matrix analysis
   - Precision, Recall, and F1-Score metrics
   - Feature importance visualization


## Key Findings

### Data Insights
- **Imbalanced Dataset:** ~350 approved loans vs. ~150 rejected loans
- **Gender Distribution:** ~80% male applicants
- **Marital Status:** Married applicants are ~2x more common than single applicants
- **Education:** ~80% of applicants are graduates
- **Employment:** ~90% are not self-employed
- **Property Area:** Relatively equal distribution across Urban, Rural, and Semi-Urban areas

### Model Performance
- **Best Performing Feature:** Credit history has the highest importance weight
- **Accuracy:** 81.19% on test set
- **Precision (Approved):** 79% - reasonably reliable approval predictions
- **Recall (Approved):** 99% - captures almost all approved loans

### Notable Observations
- Applicant income shows positive correlation with loan amount
- Loan amount term and credit history should ideally be treated as categorical features
- Linear Regression works reasonably well despite being a continuous regression model adapted for classification


## Technologies Used

- **Python 3**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Scientific computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning model and metrics


## Libraries Required

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | 81.19% |
| Precision (Class 0) | 94% |
| Recall (Class 0) | 47% |
| Precision (Class 1) | 79% |
| Recall (Class 1) | 99% |
| F1-Score (Class 1) | 0.87 |

## Limitations & Future Improvements

1. **Model Choice:** Linear Regression is not ideal for binary classification; logistic regression or tree-based models would be more appropriate
2. **Class Imbalance:** The dataset is imbalanced, which may bias predictions toward approvals
3. **Feature Engineering:** Additional features or feature interactions could improve performance
4. **Hyperparameter Tuning:** No optimization was performed on model parameters
5. **Cross-Validation:** Single train-test split may not be representative


## How to Use

1. Clone the repository containing the datasets
2. Run the notebook cells sequentially
3. Review EDA visualizations to understand data patterns
4. Train the model and evaluate predictions
5. Analyze feature importance for business insights

## Author

**Alaa Almouiz F. Moh.**  
ID: S2026_176  
Submitted for: ZAKA ©

---

## License

This project is submitted as part of a machine learning challenge.
