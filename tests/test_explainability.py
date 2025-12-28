import sys
import os
import numpy as np
import pandas as pd
import pytest

# Make src importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from task3_model_explain import (
    get_model_feature_importance,
    compute_shap_values,
    get_tp_fp_fn_indices
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


@pytest.fixture
def dummy_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    return X, y


@pytest.fixture
def trained_model(dummy_data):
    X, y = dummy_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_feature_importance_output(trained_model):
    model, X, _ = trained_model
    importance = get_model_feature_importance(model, X.columns, top_n=5)

    assert isinstance(importance, pd.Series)
    assert len(importance) == 5


def test_shap_values_shape(trained_model):
    model, X, _ = trained_model
    explainer, shap_values = compute_shap_values(model, X)

    # Binary classification returns list
    assert isinstance(shap_values, list)
    assert shap_values[1].shape == X.shape


def test_tp_fp_fn_indices():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])

    tp, fp, fn = get_tp_fp_fn_indices(y_true, y_pred)

    assert tp.tolist() == [0]
    assert fp.tolist() == [1]
    assert fn.tolist() == [2]
