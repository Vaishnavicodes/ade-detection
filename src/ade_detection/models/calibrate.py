"""Probability calibration for the XGBoost ADE risk classifier.

Trains on the 80% fit-split of train.parquet, fits isotonic and sigmoid
calibrators on the 20% calib-split using the prefit pattern (cv="prefit"
was removed in sklearn 1.4; calibrators are fitted manually on the base
model's outputs), evaluates both on test.parquet via Brier score, picks
the winner, and saves:

  data/processed/xgb_ade_calibrated.joblib   — fitted _PrefitCalibratedModel
  data/processed/calibration_reliability.png — reliability diagram

Usage::

    python -m ade_detection.models.calibrate

Prerequisites:
  data/processed/train.parquet  — from temporal_split.py
  data/processed/test.parquet   — from temporal_split.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ade_detection.models.risk_classifier import _get_feature_cols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefit calibrator wrapper
# ---------------------------------------------------------------------------


class _PrefitCalibratedModel:
    """Base model + fitted calibrator with a sklearn-compatible predict_proba.

    Replaces CalibratedClassifierCV(cv='prefit'), which was removed in sklearn
    1.4. The calibrator (isotonic or sigmoid) is fitted on the base model's
    raw probabilities over a held-out calib set, so the base model itself is
    never retrained.
    """

    def __init__(
        self,
        base_model: XGBClassifier,
        calibrator: IsotonicRegression | LogisticRegression,
        method: str,
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(np.asarray(X, dtype=float))[:, 1]
        if self.method == "isotonic":
            cal = self.calibrator.predict(raw)
        else:  # sigmoid / Platt
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


# ---------------------------------------------------------------------------
# Core calibration logic
# ---------------------------------------------------------------------------


def _train_base_model(X_fit: np.ndarray, y_fit: np.ndarray) -> XGBClassifier:
    """Train XGBoost on the fit subset using the same hyperparams as risk_classifier."""
    n_neg = int((y_fit == 0).sum())
    n_pos = int((y_fit == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_fit, y_fit)
    return model


def _fit_calibrator(
    base_model: XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str,
) -> _PrefitCalibratedModel:
    """Fit one calibrator (isotonic or sigmoid) on the calib split."""
    y_prob_calib = base_model.predict_proba(X_calib)[:, 1]

    if method == "isotonic":
        cal: IsotonicRegression | LogisticRegression = IsotonicRegression(out_of_bounds="clip")
        cal.fit(y_prob_calib, y_calib)
    else:  # sigmoid / Platt scaling
        cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        cal.fit(y_prob_calib.reshape(-1, 1), y_calib)

    return _PrefitCalibratedModel(base_model, cal, method)


def calibrate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calib_frac: float = 0.20,
    seed: int = 42,
) -> tuple[_PrefitCalibratedModel, dict]:
    """Calibrate an XGBoost ADE classifier and return the better calibrator.

    Parameters
    ----------
    train_df:
        Full training set (output of temporal_split). Must contain 'ade_label'.
    test_df:
        Hold-out test set for final Brier evaluation.
    calib_frac:
        Fraction of train_df reserved for fitting the calibrators (default 0.20).
    seed:
        Random seed for the stratified fit/calib split.

    Returns
    -------
    (best_calibrator, results) : tuple
        best_calibrator — _PrefitCalibratedModel (isotonic or sigmoid)
        results — dict with keys:
            brier_raw, brier_isotonic, brier_sigmoid,
            chosen_method,
            curve_raw, curve_isotonic, curve_sigmoid
            (each curve is (fraction_of_positives, mean_predicted_value))
    """
    feature_cols = _get_feature_cols(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in train_df")

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["ade_label"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["ade_label"].values.astype(int)

    # Stratified split: fit subset trains the base model, calib subset fits calibrators
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=calib_frac,
        stratify=y_train,
        random_state=seed,
    )
    logger.info(
        "Fit subset: %d rows (%d pos)  |  Calib subset: %d rows (%d pos)",
        len(X_fit),
        int(y_fit.sum()),
        len(X_calib),
        int(y_calib.sum()),
    )

    base_model = _train_base_model(X_fit, y_fit)

    # Raw (uncalibrated) probabilities on test
    y_prob_raw = base_model.predict_proba(X_test)[:, 1]
    brier_raw = float(brier_score_loss(y_test, y_prob_raw))

    # Fit each calibrator on calib subset, evaluate on test
    calibrators: dict[str, _PrefitCalibratedModel] = {}
    briers: dict[str, float] = {}
    curves: dict[str, tuple] = {}

    for method in ("isotonic", "sigmoid"):
        cal = _fit_calibrator(base_model, X_calib, y_calib, method)
        y_prob_cal = cal.predict_proba(X_test)[:, 1]
        briers[method] = float(brier_score_loss(y_test, y_prob_cal))
        curves[method] = calibration_curve(y_test, y_prob_cal, n_bins=10)
        calibrators[method] = cal

    curves["raw"] = calibration_curve(y_test, y_prob_raw, n_bins=10)

    chosen = "isotonic" if briers["isotonic"] <= briers["sigmoid"] else "sigmoid"
    logger.info(
        "Brier — raw: %.4f  isotonic: %.4f  sigmoid: %.4f  → chosen: %s",
        brier_raw,
        briers["isotonic"],
        briers["sigmoid"],
        chosen,
    )

    results = {
        "brier_raw": brier_raw,
        "brier_isotonic": briers["isotonic"],
        "brier_sigmoid": briers["sigmoid"],
        "chosen_method": chosen,
        "curve_raw": curves["raw"],
        "curve_isotonic": curves["isotonic"],
        "curve_sigmoid": curves["sigmoid"],
    }
    return calibrators[chosen], results


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def _save_reliability_plot(results: dict, out_path: Path) -> None:
    """Reliability diagram: raw / isotonic / sigmoid curves + perfect diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6))

    styles = {
        "raw": dict(color="#e41a1c", linestyle="--", marker="o", label="Raw (uncalibrated)"),
        "isotonic": dict(color="#377eb8", linestyle="-", marker="s", label="Isotonic"),
        "sigmoid": dict(color="#4daf4a", linestyle="-", marker="^", label="Sigmoid (Platt)"),
    }

    for key, style in styles.items():
        frac_pos, mean_pred = results[f"curve_{key}"]
        ax.plot(mean_pred, frac_pos, **style)

    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("ADE risk classifier — reliability diagram")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved reliability plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Calibrate the XGBoost ADE risk classifier")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    logger.info("Loading train: %s", train_path)
    train_df = pd.read_parquet(train_path)
    logger.info("Loading test:  %s", test_path)
    test_df = pd.read_parquet(test_path)

    best_cal, results = calibrate_model(train_df, test_df)

    model_path = processed_dir / "xgb_ade_calibrated.joblib"
    joblib.dump(best_cal, model_path)
    logger.info("Saved calibrated model → %s", model_path)

    plot_path = processed_dir / "calibration_reliability.png"
    _save_reliability_plot(results, plot_path)

    chosen = results["chosen_method"]
    print("\n" + "=" * 50)
    print("  Calibration results (Brier score, lower = better)")
    print("=" * 50)
    print(f"  Raw (uncalibrated) : {results['brier_raw']:.4f}")
    print(f"  Isotonic           : {results['brier_isotonic']:.4f}")
    print(f"  Sigmoid (Platt)    : {results['brier_sigmoid']:.4f}")
    print(f"  Chosen             : {chosen}")
    print("=" * 50)
    print(f"\nArtifacts written to {processed_dir}:")
    print("  xgb_ade_calibrated.joblib")
    print("  calibration_reliability.png")


if __name__ == "__main__":
    main()
