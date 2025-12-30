# 📓 Notebooks Overview

This folder contains all Jupyter notebooks used for exploratory analysis, feature engineering, modeling, and explainability in the fraud detection project.

Each notebook is designed to be **self-contained**, **reproducible**, and **clearly documented**.

---

## 📁 Notebook Descriptions

### 1️⃣ `eda-fraud-data.ipynb`
**Purpose:** Exploratory Data Analysis for `Fraud_Data.csv`

Key steps:
- Data quality checks (missing values, duplicates)
- Univariate and bivariate analysis
- Class imbalance analysis
- Time-based behavior exploration
- Country-level fraud patterns (after IP integration)

---

### 2️⃣ `eda-creditcard.ipynb`
**Purpose:** Exploratory Data Analysis for `creditcard.csv`

Key steps:
- Distribution analysis of PCA-transformed features
- Fraud vs non-fraud comparison
- Severe class imbalance quantification
- Initial insights for modeling strategy

---

### 3️⃣ `feature-engineering.ipynb`
**Purpose:** Feature creation and transformation

Key steps:
- Time-based features (hour, day of week)
- Transaction velocity & frequency
- Time since signup
- Scaling numerical features
- Encoding categorical variables
- Train-only SMOTE resampling
- Saving processed datasets

---

### 4️⃣ `modeling.ipynb`
**Purpose:** Model training, evaluation, and selection

Key steps:
- Stratified train-test split
- Baseline Logistic Regression
- Ensemble models (Random Forest / XGBoost / LightGBM)
- Stratified K-Fold Cross-Validation
- Metric comparison (F1, AUC-PR)
- Best model selection and persistence

---

### 5️⃣ `shap-explainability.ipynb`
**Purpose:** Model interpretability and business insight generation

Key steps:
- Built-in feature importance analysis
- SHAP global summary plots
- SHAP force plots:
  - True Positive
  - False Positive
  - False Negative
- SHAP vs built-in comparison
- Key fraud driver identification
- Business recommendations linked to SHAP insights

---

## ✅ Execution Order (Recommended)

1. `eda-fraud-data.ipynb`
2. `eda-creditcard.ipynb`
3. `feature-engineering.ipynb`
4. `modeling.ipynb`
5. `shap-explainability.ipynb`

---

## ⚠️ Notes
- Raw data is not tracked in GitHub (see `.gitignore`)
- All paths assume project root execution
- Models are saved in the `models/` directory
