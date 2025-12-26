# src/task2_modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, auc


#1.Data Preparation

def split_features_target(df, target_col):
    """
    Separates features and target from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The dataset
        target_col (str): The name of the target column
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Performs stratified train-test split.
    
    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Fraction for test set
        random_state (int)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, 
                            stratify=y, random_state=random_state)

#2.Build Baseline Model

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model with balanced class weights.
    Returns the trained model.
    """
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return F1-score, AUC-PR, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return {"F1-score": f1, "AUC-PR": ap, "Confusion Matrix": cm}

#3.Ensemble Modeling

# ----------------------------
# Random Forest
# ----------------------------
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier with basic hyperparameters.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

# ----------------------------
# XGBoost
# ----------------------------
def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1):
    """
    Train an XGBoost classifier with basic hyperparameters.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb

# ----------------------------
# LightGBM
# ----------------------------
def train_lightgbm(X_train, y_train, n_estimators=100, max_depth=-1, learning_rate=0.1):
    """
    Train a LightGBM classifier with basic hyperparameters.
    """
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_train, y_train)
    return lgbm

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using F1-score, AUC-PR, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    results = {
        'F1-score': f1_score(y_test, y_pred),
        'AUC-PR': average_precision_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    return results


#4. Cross-Validation

# =====================================================
# Train functions (already consistent with  style)
# =====================================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm_safe(X_train, y_train,
                        n_estimators=100,
                        max_depth=-1,
                        learning_rate=0.1):
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train.astype(int))
    return model


# =====================================================
# Cross-Validation (Task 4)
# =====================================================

def cross_validate_model(model_fn, X, y, k=5, **model_kwargs):
    """
    model_fn: training function (e.g., train_random_forest)
    X, y: full dataset
    k: number of folds
    model_kwargs: hyperparameters for the model
    """

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    f1_scores = []
    auc_pr_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(X_train, y_train, **model_kwargs)
        y_pred = model.predict(X_val)

        f1_scores.append(f1_score(y_val, y_pred))
        auc_pr_scores.append(average_precision_score(y_val, y_pred))

    return {
        "F1-score Mean": np.mean(f1_scores),
        "F1-score Std": np.std(f1_scores),
        "AUC-PR Mean": np.mean(auc_pr_scores),
        "AUC-PR Std": np.std(auc_pr_scores),
    }

#5. Model Comparison

def build_model_comparison_table(cv_results):
    """
    Build a side-by-side comparison table from CV results.

    Parameters
    ----------
    cv_results : list of dict
        Each dict contains model CV metrics.

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    df = pd.DataFrame(cv_results)
    return df


def rank_models(
    df,
    metric="AUC_PR_mean",
    ascending=False
):
    """
    Rank models based on selected metric.

    Parameters
    ----------
    df : pd.DataFrame
        Model comparison table
    metric : str
        Metric used for ranking
    ascending : bool
        Sort order

    Returns
    -------
    pd.DataFrame
        Ranked models
    """
    return df.sort_values(by=metric, ascending=ascending)


def summarize_best_model(df, dataset_name):
    """
    Select best model for a dataset based on AUC-PR.

    Parameters
    ----------
    df : pd.DataFrame
        Model comparison table
    dataset_name : str
        Dataset name (Creditcard or Fraud)

    Returns
    -------
    pd.Series
        Best model summary
    """
    df_subset = df[df["Dataset"] == dataset_name]
    best_model = df_subset.sort_values(
        by="AUC_PR_mean", ascending=False
    ).iloc[0]

    return best_model
