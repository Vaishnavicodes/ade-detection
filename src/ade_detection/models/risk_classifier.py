"""XGBoost ADE risk classifier.

Trains on the temporally-split feature matrix produced by
build_features.py + temporal_split.py and evaluates at two operating points:

  1. Default 0.5 threshold  — standard classification metrics
  2. ~0.80-sensitivity threshold — clinically meaningful: catching 80% of
     true ADEs at the cost of some false positives (clinicians prefer high
     recall for a safety-critical label)

Usage::

    python -m ade_detection.models.risk_classifier

Prerequisites: data/processed/train.parquet and test.parquet must exist.
Run build_features.py then temporal_split.py (or the main() of temporal_split)
to produce them.

The trained model is saved to data/processed/xgb_ade_model.json.
That file is NOT committed — it is a data artifact (gitignored via /data/).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Columns that are identifiers, targets, or datetimes — never used as features
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "ade_label",
        "ade_prob",
        "visit_occurrence_id",
        "person_id",
        "visit_start_datetime",
        "index_datetime",
    }
)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names, excluding meta/target/datetime columns."""
    cols = []
    for col in df.columns:
        if col in _NON_FEATURE_COLS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Sensitivity, specificity, precision, F1 at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _find_sensitivity_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.80,
) -> float:
    """Return the lowest threshold that achieves >= target_sensitivity.

    Sweeps candidate thresholds from high to low.  Returns the threshold at
    which sensitivity first reaches the target — i.e. the operating point where
    we catch at least target_sensitivity fraction of true ADEs.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    best_thresh = thresholds[-1]
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        if sens >= target_sensitivity:
            best_thresh = t
            break
    return float(best_thresh)


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_sensitivity: float = 0.80,
    model_save_path: str | Path | None = None,
) -> tuple[XGBClassifier, dict]:
    """Train an XGBoost ADE risk classifier and evaluate on the test set.

    Parameters
    ----------
    train_df:
        Training feature matrix (output of build_features + temporal_split).
        Must contain 'ade_label' and numeric feature columns.
    test_df:
        Hold-out test set (same schema as train_df).
    target_sensitivity:
        Sensitivity level for the clinical operating point (default 0.80).
    model_save_path:
        If provided, save the trained model to this path as XGBoost JSON.
        Should be inside data/ so it is gitignored.

    Returns
    -------
    (model, metrics) : tuple[XGBClassifier, dict]
        model   — fitted XGBClassifier with predict_proba
        metrics — dict with keys:
            roc_auc, pr_auc,
            op_05 (dict): threshold=0.5 operating point metrics
            op_sens (dict): ~target_sensitivity operating point metrics
    """
    feature_cols = _get_feature_cols(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in train_df")

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["ade_label"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["ade_label"].values.astype(int)

    logger.info(
        "Training on %d admissions (%d features); testing on %d admissions",
        len(X_train),
        len(feature_cols),
        len(X_test),
    )

    # class imbalance: weight positives to compensate for ~8% ADE rate
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(
        "Class balance — neg: %d  pos: %d  scale_pos_weight: %.2f",
        n_neg,
        n_pos,
        scale_pos_weight,
    )

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
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Operating point 1: fixed 0.5 threshold
    op_05 = _threshold_metrics(y_test, y_prob, threshold=0.5)

    # Operating point 2: threshold chosen to achieve ~target_sensitivity
    sens_thresh = _find_sensitivity_threshold(y_test, y_prob, target_sensitivity)
    op_sens = _threshold_metrics(y_test, y_prob, threshold=sens_thresh)

    logger.info("ROC-AUC: %.4f  PR-AUC: %.4f", roc_auc, pr_auc)

    # Confusion matrix at default threshold
    print("\n--- Confusion Matrix (threshold=0.50) ---")
    cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int), labels=[0, 1])
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))
        logger.info("Model saved to %s", model_save_path)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "op_05": op_05,
        "op_sens": op_sens,
    }
    return model, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Train XGBoost ADE risk classifier on temporal train/test split"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.80,
        help="Sensitivity target for the clinical operating point (default 0.80)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    model_path = processed_dir / "xgb_ade_model.json"

    logger.info("Loading train: %s", train_path)
    train_df = pd.read_parquet(train_path)
    logger.info("Loading test:  %s", test_path)
    test_df = pd.read_parquet(test_path)

    model, metrics = train_model(
        train_df,
        test_df,
        target_sensitivity=args.target_sensitivity,
        model_save_path=model_path,
    )

    op5 = metrics["op_05"]
    ops = metrics["op_sens"]

    print("\n" + "=" * 55)
    print("  ADE Risk Classifier — Evaluation Report")
    print("=" * 55)
    print(f"  Features used     : {metrics['n_features']}")
    print(f"  Train admissions  : {len(train_df):,}")
    print(f"  Test admissions   : {len(test_df):,}")
    print(f"  Test ADE rate     : {100 * test_df['ade_label'].mean():.1f}%")
    print()
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print()
    print("  --- Operating point 1: threshold = 0.50 ---")
    print(f"  Sensitivity       : {op5['sensitivity']:.3f}")
    print(f"  Specificity       : {op5['specificity']:.3f}")
    print(f"  Precision         : {op5['precision']:.3f}")
    print(f"  F1                : {op5['f1']:.3f}")
    print()
    print(
        f"  --- Operating point 2: threshold = {ops['threshold']:.3f}"
        f" (~{args.target_sensitivity:.0%} sensitivity) ---"
    )
    print(f"  Sensitivity       : {ops['sensitivity']:.3f}")
    print(f"  Specificity       : {ops['specificity']:.3f}")
    print(f"  Precision         : {ops['precision']:.3f}")
    print(f"  F1                : {ops['f1']:.3f}")
    print("=" * 55)
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    main()
