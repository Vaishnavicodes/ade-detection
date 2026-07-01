"""Tests for ade_detection.models.explain.

Uses a tiny synthetic XGBClassifier trained in-test — no model artifact on disk,
no plot content assertions.  Verifies shape contracts and API behaviour only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from ade_detection.models.explain import (
    explain_model,
    global_importance,
    top_drivers_for_admission,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model_and_X():
    """Train a minimal XGBClassifier on synthetic data and return (model, X_df)."""
    rng = np.random.default_rng(7)
    n_train, n_test, n_feat = 120, 40, 6
    feature_names = [
        "age_at_admission",
        "n_drugs_first24h",
        "n_distinct_drugs_first24h",
        "prior_comorb_congestive_heart_failure",
        "n_prior_conditions",
        "n_prior_drug_exposures",
    ]
    X_tr = rng.random((n_train, n_feat))
    y_tr = (rng.random(n_train) > 0.85).astype(int)
    X_te = rng.random((n_test, n_feat))

    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=0, verbosity=0)
    model.fit(X_tr, y_tr)

    X_df = pd.DataFrame(X_te, columns=feature_names)
    return model, X_df


# ---------------------------------------------------------------------------
# explain_model
# ---------------------------------------------------------------------------


def test_explain_model_returns_2d_array(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    assert isinstance(sv, np.ndarray)
    assert sv.ndim == 2


def test_explain_model_shape_matches_X(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    assert sv.shape == X.shape, f"Expected {X.shape}, got {sv.shape}"


def test_explain_model_finite_values(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    assert np.isfinite(sv).all(), "SHAP values contain NaN or Inf"


# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------


def test_global_importance_length(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    imp = global_importance(sv, X.columns.tolist())
    assert len(imp) == X.shape[1]


def test_global_importance_columns(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    imp = global_importance(sv, X.columns.tolist())
    assert "feature" in imp.columns
    assert "mean_abs_shap" in imp.columns


def test_global_importance_sorted_descending(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    imp = global_importance(sv, X.columns.tolist())
    vals = imp["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_global_importance_non_negative(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    imp = global_importance(sv, X.columns.tolist())
    assert (imp["mean_abs_shap"] >= 0).all()


def test_global_importance_all_features_present(tiny_model_and_X):
    model, X = tiny_model_and_X
    sv = explain_model(model, X)
    imp = global_importance(sv, X.columns.tolist())
    assert set(imp["feature"]) == set(X.columns)


# ---------------------------------------------------------------------------
# top_drivers_for_admission
# ---------------------------------------------------------------------------


def test_top_drivers_returns_dataframe(tiny_model_and_X):
    model, X = tiny_model_and_X
    result = top_drivers_for_admission(model, X, idx=0)
    assert isinstance(result, pd.DataFrame)


def test_top_drivers_non_empty(tiny_model_and_X):
    model, X = tiny_model_and_X
    result = top_drivers_for_admission(model, X, idx=0)
    assert len(result) > 0


def test_top_drivers_columns(tiny_model_and_X):
    model, X = tiny_model_and_X
    result = top_drivers_for_admission(model, X, idx=0)
    assert "feature" in result.columns
    assert "shap_value" in result.columns
    assert "feature_value" in result.columns


def test_top_drivers_n_top_respected(tiny_model_and_X):
    model, X = tiny_model_and_X
    result = top_drivers_for_admission(model, X, idx=0, n_top=3)
    assert len(result) <= 3


def test_top_drivers_sorted_by_abs_shap(tiny_model_and_X):
    model, X = tiny_model_and_X
    result = top_drivers_for_admission(model, X, idx=0)
    abs_vals = result["shap_value"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_top_drivers_different_admissions(tiny_model_and_X):
    """Two different admission rows should (typically) produce different explanations."""
    model, X = tiny_model_and_X
    r0 = top_drivers_for_admission(model, X, idx=0)
    r1 = top_drivers_for_admission(model, X, idx=1)
    # Feature values must differ (different rows of X)
    assert not np.allclose(r0["feature_value"].values, r1["feature_value"].values)
