# Fraud Detection Project (Week 5-6)

## Project Overview
This project aims to detect fraudulent transactions using a structured machine learning pipeline. It covers the full cycle of a data science project including data cleaning, exploratory data analysis (EDA), feature engineering, handling class imbalance, and preparing datasets for modeling.

---

## Project Objectives
1. **Data Cleaning:** Handle missing values, remove duplicates, and correct data types.  
2. **Exploratory Data Analysis (EDA):** Understand feature distributions, relationships, and class imbalance.  
3. **Geolocation Integration:** Merge IP address data to identify fraud patterns by country.  
4. **Feature Engineering:** Create meaningful features such as transaction frequency, time-based features, and time since signup.  
5. **Data Transformation:** Scale numerical features and encode categorical features.  
6. **Handle Class Imbalance:** Apply techniques like SMOTE or undersampling to balance the target variable.

---

## Project Structure
fraud-detection/
│
├─ data/
│ ├─ raw/ # Original CSV files (not tracked in Git)
│ └─ processed/ # Cleaned and preprocessed datasets
│
├─ notebooks/ # Jupyter notebooks for EDA and feature engineering
│ ├─ Task1_EDA.ipynb
│ ├─ Task1_FeatureEngineering.ipynb
│
├─ scripts/ # Python scripts for preprocessing
│ └─ preprocessing.py
│
├─ src/ # Source code modules
│ └─ init.py
│
├─ tests/ # Placeholder for future unit tests
│ └─ init.py
│
├─ requirements.txt # Project dependencies
├─ .gitignore # Files and folders to ignore in Git
└─ README.md # Project overview and documentation

yaml
Copy code

---

## Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/rachel93augesty-del/Fraud-Detection-Week5-6.git
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy code
venv\Scripts\activate
macOS/Linux:

bash
Copy code
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Jupyter Notebooks: Explore notebooks/ for EDA and feature engineering workflows.

Scripts: Use scripts/preprocessing.py to run data cleaning and feature generation.

Data: Raw datasets are in data/raw/ (not tracked in Git). Processed datasets are saved in data/processed/.

Key Notes
Branching Strategy: Each task has its own branch (task-1, task-2, etc.) to track progress separately.

CI/CD: GitHub Actions are set up to ensure environment setup works correctly on Task 1 branch.

Future Work: Add modeling, evaluation, and deployment pipelines in subsequent tasks.