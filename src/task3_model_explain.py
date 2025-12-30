"""
Task 3 - Model Explainability

This module provides utilities for:
1. Built-in feature importance extraction
2. SHAP global and local explanations
3. TP / FP / FN analysis
4. Interpretation helpers
5. Business recommendation generation
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def compute_shap_values(model, X):
    """
    Compute SHAP values for a model and dataset.
    Handles binary classifiers (returns positive class SHAP values).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # If binary classifier, shap_values is a list: [class_0, class_1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]  # positive class (fraud)
    else:
        shap_values_pos = shap_values

    return explainer, shap_values_pos

# ============================================================
# 1️⃣ Feature Importance Baseline
# ============================================================

def get_model_feature_importance(model, feature_names, top_n=10):
    """
    Extract built-in feature importance from tree-based models
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model does not support feature importance.")

    fi = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
    return fi


def plot_feature_importance(fi, title="Top Feature Importance"):
    """
    Bar plot for built-in feature importance
    """
    fi.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# ============================================================
# 2️⃣ SHAP Analysis
# ============================================================

def get_top_shap_features(shap_values, X, top_n=5):
    """
    Returns top N features with the highest mean absolute SHAP values
    shap_values: 2D array (samples x features)
    X: DataFrame of features
    """
    shap_array = np.abs(shap_values)  # absolute values
    mean_shap = np.mean(shap_array, axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:top_n]
    top_features = pd.Series(data=mean_shap[top_idx], index=X.columns[top_idx])
    return top_features

def plot_shap_summary(shap_values, X, plot_type="dot"):
    """
    Global SHAP summary plot
    """
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=True)


def plot_shap_force(explainer, shap_values, X, index):
    """
    SHAP force plot for a single observation
    """
    shap.initjs()

    # Handle expected_value for binary classifiers
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    force_plot = shap.force_plot(
        expected_value,
        shap_values[index],
        X.iloc[index]
    )
    return force_plot


# ============================================================
# 3️⃣ TP / FP / FN Identification
# ============================================================

def get_tp_fp_fn_indices(y_true, y_pred):
    """
    Identify TP, FP, FN indices
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    return tp, fp, fn


# ============================================================
# 4️⃣ Interpretation Helpers
# ============================================================

def get_top_shap_features(shap_values, X, top_n=5):
    """
    Return top N global drivers based on mean |SHAP|
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_features = (
        pd.Series(mean_abs_shap, index=X.columns)
        .sort_values(ascending=False)
        .head(top_n)
    )
    return top_features


def compare_builtin_shap(builtin_fi, shap_top_features):
    """
    Compare built-in feature importance with SHAP importance
    """
    comparison = pd.DataFrame({
        "Built-in Importance": builtin_fi,
        "SHAP Importance": shap_top_features
    })
    return comparison


# ============================================================
# 5️⃣ Business Recommendations
# ============================================================

def generate_business_recommendations(top_shap_features):
    """
    Generate actionable recommendations based on SHAP drivers
    """
    recommendations = []

    for feature in top_shap_features.index:
        if "time" in feature.lower():
            recommendations.append(
                f"Transactions occurring at unusual times related to `{feature}` "
                f"should trigger additional verification."
            )
        elif "amount" in feature.lower():
            recommendations.append(
                f"High-risk transaction amounts influenced by `{feature}` "
                f"should be dynamically thresholded."
            )
        else:
            recommendations.append(
                f"Transactions with extreme values in `{feature}` "
                f"should be flagged for secondary fraud screening."
            )

    return recommendations
