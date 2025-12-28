# src/task3_model_explain.py

import shap
import pandas as pd
import numpy as np

def get_model_feature_importance(model, feature_names, top_n=10):
    """
    Extract built-in feature importance from tree-based models
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    return importance.head(top_n)


def compute_shap_values(model, X):
    """
    Compute SHAP values using TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def get_tp_fp_fn_indices(y_true, y_pred):
    """
    Identify TP, FP, FN indices for explainability
    """
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    return tp, fp, fn
