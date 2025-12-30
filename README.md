# 🕵️ Fraud Detection Using Machine Learning

## 📌 Project Overview
This project implements an end-to-end **fraud detection system** using machine learning techniques on highly imbalanced datasets.  
It combines **data analysis, feature engineering, ensemble modeling, explainability with SHAP**, and **business-driven insights** to support real-world fraud prevention.

Datasets used:
- **Fraud_Data.csv** – E-commerce transaction fraud data
- **creditcard.csv** – European cardholder transaction data

---

## 🎯 Project Objectives
- Prepare clean, feature-rich datasets for modeling
- Handle extreme class imbalance effectively
- Build and compare baseline and ensemble models
- Interpret model predictions using SHAP
- Translate model insights into actionable business recommendations

---

## 📁 Project Structure

fraud-detection/
├── .vscode/
│ └── settings.json
├── .github/
│ └── workflows/
│ └── unittests.yml
├── data/ # Gitignored
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned & engineered data
├── notebooks/
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ ├── modeling.ipynb
│ ├── shap-explainability.ipynb
│ └── README.md
├── src/
│ ├── task1_preprocessing.py
│ ├── task2_modeling.py
│ ├── task3_model_explain.py
│ └── init.py
├── tests/
│ └── init.py
├── models/ # Saved trained models
├── scripts/
│ └── README.md
├── requirements.txt
├── .gitignore
└── README.md

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Workflow Summary
✅ Task 1 – Data Analysis & Preprocessing
Missing value handling and duplicate removal

Exploratory Data Analysis (EDA)

Class imbalance analysis

IP address to country integration

Feature engineering:

Transaction velocity & frequency

Time-based features (hour, day of week)

Time since signup

Scaling and encoding

SMOTE applied to training data only

✅ Task 2 – Model Building & Training
Stratified train-test split

Logistic Regression baseline

Ensemble models (Random Forest / XGBoost / LightGBM)

Basic hyperparameter tuning

Stratified 5-fold cross-validation

Evaluation using:

F1-score

AUC-PR

Confusion Matrix

Best model selection and saving

✅ Task 3 – Model Explainability
Built-in feature importance analysis

SHAP global summary plots

SHAP force plots for:

True Positive (correct fraud detection)

False Positive (legitimate flagged as fraud)

False Negative (missed fraud)

SHAP vs built-in importance comparison

Identification of top fraud drivers

Business recommendations grounded in SHAP insights

📊 Key Insights
Fraud signals are driven by non-linear feature interactions

SHAP reveals important features overlooked by built-in importance

Timing, velocity, and behavioral features are strong fraud indicators

💡 Business Recommendations
Apply enhanced verification for transactions shortly after signup

Monitor high-risk feature combinations identified by SHAP

Use SHAP-based alerts to flag rare but high-impact fraud patterns

🧪 Testing & CI
Unit testing enabled via GitHub Actions

Modular, reusable code structure under src/

👤 Author
Rahel Aklog
Senior Analytics Officer | Data Science & Machine Learning

📅 Final Submission
🕗 20:00 UTC – Tuesday, 30 December 2025