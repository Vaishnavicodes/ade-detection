"""Tests for ade_detection.models.risk_classifier.

Uses tiny synthetic train/test DataFrames — no real MIMIC data, no model
performance assertions.  We verify the API contract only:
  - train_model runs without error
  - the returned model has predict_proba
  - the metrics dict contains the expected keys
  - AUC values are floats in [0, 1]
  - operating-point sub-dicts contain the required keys
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ade_detection.models.risk_classifier import (
    _find_sensitivity_threshold,
    _get_feature_cols,
    _threshold_metrics,
    train_model,
)

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _make_frame(n: int, pos_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Create a minimal feature DataFrame matching build_features output schema."""
    rng = np.random.default_rng(seed)
    n_pos = max(1, int(n * pos_frac))
    n_neg = n - n_pos
    labels = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(labels)

    return pd.DataFrame(
        {
            "visit_occurrence_id": np.arange(n),
            "person_id": np.arange(n),
            "visit_start_datetime": pd.date_range("2150-01-01", periods=n, freq="D"),
            "index_datetime": pd.date_range("2150-01-02", periods=n, freq="D"),
            "age_at_admission": rng.integers(20, 90, size=n),
            "gender_concept_id": rng.integers(0, 2, size=n),
            "race_concept_id": rng.integers(0, 5, size=n),
            "ethnicity_concept_id": rng.integers(0, 3, size=n),
            "n_drugs_first24h": rng.integers(0, 10, size=n),
            "n_distinct_drugs_first24h": rng.integers(0, 8, size=n),
            "prior_comorb_myocardial_infarction": rng.integers(0, 2, size=n),
            "prior_comorb_congestive_heart_failure": rng.integers(0, 2, size=n),
            "n_prior_conditions": rng.integers(0, 20, size=n),
            "n_prior_drug_exposures": rng.integers(0, 15, size=n),
            "ade_prob": rng.random(size=n),
            "ade_label": labels,
        }
    )


@pytest.fixture()
def train_test_frames():
    train_df = _make_frame(n=200, pos_frac=0.15, seed=1)
    test_df = _make_frame(n=80, pos_frac=0.15, seed=2)
    return train_df, test_df


# ---------------------------------------------------------------------------
# _get_feature_cols
# ---------------------------------------------------------------------------


def test_get_feature_cols_excludes_meta():
    df = _make_frame(10)
    cols = _get_feature_cols(df)
    for excluded in [
        "ade_label",
        "ade_prob",
        "visit_occurrence_id",
        "person_id",
    ]:
        assert excluded not in cols


def test_get_feature_cols_excludes_datetimes():
    df = _make_frame(10)
    cols = _get_feature_cols(df)
    assert "visit_start_datetime" not in cols
    assert "index_datetime" not in cols


def test_get_feature_cols_includes_numerics():
    df = _make_frame(10)
    cols = _get_feature_cols(df)
    for expected in [
        "age_at_admission",
        "n_drugs_first24h",
        "n_prior_conditions",
        "prior_comorb_congestive_heart_failure",
    ]:
        assert expected in cols


# ---------------------------------------------------------------------------
# _threshold_metrics
# ---------------------------------------------------------------------------


def test_threshold_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = _threshold_metrics(y_true, y_prob, threshold=0.5)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_threshold_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.7, 0.4, 0.6])
    m = _threshold_metrics(y_true, y_prob, threshold=0.5)
    for key in [
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]:
        assert key in m


# ---------------------------------------------------------------------------
# _find_sensitivity_threshold
# ---------------------------------------------------------------------------


def test_find_sensitivity_threshold_achieves_target():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    thresh = _find_sensitivity_threshold(y_true, y_prob, target_sensitivity=0.80)
    y_pred = (y_prob >= thresh).astype(int)
    from sklearn.metrics import recall_score

    sens = recall_score(y_true, y_pred, zero_division=0)
    assert sens >= 0.80


def test_find_sensitivity_threshold_returns_float():
    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.7])
    t = _find_sensitivity_threshold(y_true, y_prob, target_sensitivity=0.80)
    assert isinstance(t, float)


# ---------------------------------------------------------------------------
# train_model — API contract
# ---------------------------------------------------------------------------


def test_train_model_returns_tuple(train_test_frames):
    train_df, test_df = train_test_frames
    result = train_model(train_df, test_df)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_train_model_returns_classifier_with_predict_proba(train_test_frames):
    train_df, test_df = train_test_frames
    model, _ = train_model(train_df, test_df)
    feature_cols = _get_feature_cols(test_df)
    probs = model.predict_proba(test_df[feature_cols].values.astype(float))
    assert probs.shape == (len(test_df), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_metrics_dict_top_level_keys(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    for key in ["roc_auc", "pr_auc", "n_features", "feature_cols", "op_05", "op_sens"]:
        assert key in metrics, f"Missing key: {key}"


def test_roc_auc_is_float_in_unit_interval(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    assert isinstance(metrics["roc_auc"], float)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_pr_auc_is_float_in_unit_interval(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    assert isinstance(metrics["pr_auc"], float)
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_op_05_sub_dict_keys(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    required = {
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    }
    assert required.issubset(metrics["op_05"].keys())


def test_op_sens_sub_dict_keys(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    required = {
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    }
    assert required.issubset(metrics["op_sens"].keys())


def test_op_sens_threshold_below_05(train_test_frames):
    """The ~0.80-sensitivity threshold should be lower than 0.5 for imbalanced data."""
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    # Not a hard rule, but the clinical threshold is almost always more permissive
    assert metrics["op_sens"]["threshold"] <= 0.5 + 1e-6


def test_n_features_positive(train_test_frames):
    train_df, test_df = train_test_frames
    _, metrics = train_model(train_df, test_df)
    assert metrics["n_features"] > 0
    assert len(metrics["feature_cols"]) == metrics["n_features"]


def test_train_model_saves_model(train_test_frames, tmp_path):
    train_df, test_df = train_test_frames
    model_path = tmp_path / "test_model.json"
    train_model(train_df, test_df, model_save_path=model_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0
