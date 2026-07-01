"""Tests for ade_detection.models.calibrate.

Verifies the API contract on tiny synthetic data — no real MIMIC data,
no plot assertions, no performance assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ade_detection.models.calibrate import _PrefitCalibratedModel, calibrate_model

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _make_frame(n: int, pos_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = max(2, int(n * pos_frac))
    n_neg = n - n_pos
    labels = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "visit_occurrence_id": np.arange(n),
            "person_id": np.arange(n),
            "visit_start_datetime": pd.date_range("2150-01-01", periods=n, freq="D"),
            "index_datetime": pd.date_range("2150-01-02", periods=n, freq="D"),
            "age_at_admission": rng.integers(20, 90, size=n).astype(float),
            "n_drugs_first24h": rng.integers(0, 10, size=n).astype(float),
            "n_distinct_drugs_first24h": rng.integers(0, 8, size=n).astype(float),
            "prior_comorb_congestive_heart_failure": rng.integers(0, 2, size=n).astype(float),
            "n_prior_conditions": rng.integers(0, 20, size=n).astype(float),
            "n_prior_drug_exposures": rng.integers(0, 15, size=n).astype(float),
            "ade_prob": rng.random(size=n),
            "ade_label": labels,
        }
    )


@pytest.fixture()
def train_test_frames():
    train_df = _make_frame(n=300, pos_frac=0.15, seed=1)
    test_df = _make_frame(n=100, pos_frac=0.15, seed=2)
    return train_df, test_df


# ---------------------------------------------------------------------------
# calibrate_model — return type
# ---------------------------------------------------------------------------


def test_calibrate_returns_tuple(train_test_frames):
    train_df, test_df = train_test_frames
    result = calibrate_model(train_df, test_df)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_best_calibrator_is_prefit_calibrated_model(train_test_frames):
    train_df, test_df = train_test_frames
    best_cal, _ = calibrate_model(train_df, test_df)
    assert isinstance(best_cal, _PrefitCalibratedModel)


def test_best_calibrator_has_predict_proba(train_test_frames):
    train_df, test_df = train_test_frames
    best_cal, _ = calibrate_model(train_df, test_df)
    assert callable(getattr(best_cal, "predict_proba", None))


def test_results_dict_keys(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    required = {
        "brier_raw",
        "brier_isotonic",
        "brier_sigmoid",
        "chosen_method",
        "curve_raw",
        "curve_isotonic",
        "curve_sigmoid",
    }
    assert required.issubset(results.keys())


# ---------------------------------------------------------------------------
# Brier scores
# ---------------------------------------------------------------------------


def test_brier_scores_are_floats(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    for key in ("brier_raw", "brier_isotonic", "brier_sigmoid"):
        assert isinstance(results[key], float), f"{key} is not float"


def test_brier_scores_in_unit_interval(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    for key in ("brier_raw", "brier_isotonic", "brier_sigmoid"):
        assert 0.0 <= results[key] <= 1.0, f"{key} = {results[key]} out of [0,1]"


def test_chosen_method_is_valid(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    assert results["chosen_method"] in ("isotonic", "sigmoid")


def test_chosen_is_lower_brier(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    chosen = results["chosen_method"]
    other = "sigmoid" if chosen == "isotonic" else "isotonic"
    assert results[f"brier_{chosen}"] <= results[f"brier_{other}"]


# ---------------------------------------------------------------------------
# Calibrator predict_proba output
# ---------------------------------------------------------------------------


def test_calibrator_predict_proba_shape(train_test_frames):
    train_df, test_df = train_test_frames
    from ade_detection.models.risk_classifier import _get_feature_cols

    best_cal, _ = calibrate_model(train_df, test_df)
    feature_cols = _get_feature_cols(test_df)
    X_test = test_df[feature_cols].values.astype(float)
    probs = best_cal.predict_proba(X_test)
    assert probs.shape == (len(test_df), 2)


def test_calibrator_probs_in_unit_interval(train_test_frames):
    train_df, test_df = train_test_frames
    from ade_detection.models.risk_classifier import _get_feature_cols

    best_cal, _ = calibrate_model(train_df, test_df)
    feature_cols = _get_feature_cols(test_df)
    X_test = test_df[feature_cols].values.astype(float)
    probs = best_cal.predict_proba(X_test)[:, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_calibrator_probs_sum_to_one(train_test_frames):
    train_df, test_df = train_test_frames
    from ade_detection.models.risk_classifier import _get_feature_cols

    best_cal, _ = calibrate_model(train_df, test_df)
    feature_cols = _get_feature_cols(test_df)
    X_test = test_df[feature_cols].values.astype(float)
    probs = best_cal.predict_proba(X_test)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Calibration curves
# ---------------------------------------------------------------------------


def test_curve_raw_is_tuple_of_arrays(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    frac_pos, mean_pred = results["curve_raw"]
    assert isinstance(frac_pos, np.ndarray)
    assert isinstance(mean_pred, np.ndarray)
    assert len(frac_pos) == len(mean_pred)


def test_curve_arrays_same_length_across_methods(train_test_frames):
    train_df, test_df = train_test_frames
    _, results = calibrate_model(train_df, test_df)
    for method in ("raw", "isotonic", "sigmoid"):
        frac, pred = results[f"curve_{method}"]
        assert len(frac) == len(pred), f"curve_{method} length mismatch"


# ---------------------------------------------------------------------------
# Joblib round-trip
# ---------------------------------------------------------------------------


def test_best_calibrator_is_serialisable(train_test_frames, tmp_path):
    import joblib

    train_df, test_df = train_test_frames
    best_cal, _ = calibrate_model(train_df, test_df)
    save_path = tmp_path / "test_cal.joblib"
    joblib.dump(best_cal, save_path)
    loaded = joblib.load(save_path)
    assert isinstance(loaded, _PrefitCalibratedModel)
