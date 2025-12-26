Fraud Detection Project (Week 5-6)
Project Overview

This project aims to detect fraudulent transactions in e-commerce and bank datasets using a full machine learning workflow. It covers:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Handling class imbalance

Model building and evaluation

Explainable AI analysis using SHAP

Business-focused recommendations

The goal is to improve fraud detection accuracy, reduce false positives/negatives, and provide interpretable insights for decision-making.

Project Objectives

Data Cleaning: Handle missing values, remove duplicates, and correct data types.

Exploratory Data Analysis (EDA): Understand feature distributions, relationships, and class imbalance.

Geolocation Integration: Map IP addresses to countries and analyze fraud patterns.

Feature Engineering: Create features like:

Transaction frequency and velocity

Time-based features (hour_of_day, day_of_week)

time_since_signup

Data Transformation: Scale numerical features and encode categorical variables.

Handle Class Imbalance: Apply SMOTE or undersampling to balance the target variable.

Model Building & Evaluation:

Baseline model: Logistic Regression

Ensemble models: Random Forest, XGBoost, LightGBM

Stratified K-Fold cross-validation (k=5)

Metrics: AUC-PR, F1-score, confusion matrix

Model Explainability: Use SHAP to interpret predictions and identify key fraud drivers.

Business Recommendations: Provide actionable insights based on SHAP and feature importance.

Data Description
1. E-commerce Dataset: Fraud_Data.csv

user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class

Highly imbalanced dataset.

2. IP Mapping Dataset: IpAddress_to_Country.csv

lower_bound_ip_address, upper_bound_ip_address, country

3. Bank Dataset: creditcard.csv

Time, V1–V28, Amount, Class

Extremely imbalanced dataset.

Project Structure
fraud-detection/
│
├─ data/
│  ├─ raw/         # Original CSV files (not tracked)
│  └─ processed/   # Cleaned and feature-engineered datasets
│
├─ notebooks/
│  ├─ __init__.py
│  ├─ eda-fraud-data.ipynb
│  ├─ eda-creditcard.ipynb
│  ├─ feature-engineering.ipynb
│  ├─ modeling.ipynb
│  ├─ shap-explainability.ipynb
│  └─ README.md
│
├─ scripts/
│  ├─ __init__.py
│  └─ preprocessing.py
│
├─ src/
│  ├─ __init__.py
│  ├─ task1_data_analysis.py
│  └─ task2_modeling.py
│
├─ tests/
│  └─ __init__.py
│
├─ models/          # Saved model artifacts
├─ requirements.txt
├─ .gitignore
└─ README.md