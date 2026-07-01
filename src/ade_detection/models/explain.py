"""SHAP explainability for the XGBoost ADE risk classifier.

Produces three artifacts (saved to data/processed/, gitignored):
  shap_global_importance.csv  — mean(|SHAP|) per feature, ranked
  shap_global_importance.png  — horizontal bar plot of the above
  shap_summary.png            — beeswarm plot (value + direction per feature)

And prints the top features to console so clinicians can read them immediately.

Usage::

    python -m ade_detection.models.explain

Prerequisites:
  data/processed/xgb_ade_model.json  — trained model from risk_classifier.py
  data/processed/test.parquet        — held-out test set from temporal_split.py

Output artifacts are NOT committed — they live under /data/ which is gitignored.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from xgboost import XGBClassifier

from ade_detection.models.risk_classifier import _get_feature_cols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------


def _fix_base_score(model: XGBClassifier) -> None:
    """Normalise XGBoost 3.x base_score in-place before passing to SHAP.

    XGBoost 3.x auto-computes base_score from training data and stores it as a
    bracketed JSON array (e.g. "[1.5E-1]") rather than a scalar.  SHAP's
    TreeExplainer reads this value via save_config() and calls float() on it,
    which raises ValueError on the bracket format in some version combinations.
    Setting base_score explicitly on the booster forces a scalar representation.
    """
    import json

    booster = model.get_booster()
    cfg = json.loads(booster.save_config())
    raw = cfg.get("learner", {}).get("learner_model_param", {}).get("base_score", "0.5")
    # raw may be "[1.5E-1]" (array) or "1.5E-1" (scalar) — normalise to float
    if raw.startswith("["):
        raw = raw.strip("[]")
    booster.set_param("base_score", str(float(raw)))


def explain_model(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for every sample in X using TreeExplainer.

    TreeExplainer is exact (not sampling-based) for tree ensembles and
    orders of magnitude faster than KernelExplainer on XGBoost models.

    Parameters
    ----------
    model:
        Fitted XGBClassifier.
    X:
        Feature matrix (n_samples × n_features) as a DataFrame.
        Column names are used to label the SHAP output.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        SHAP values for the positive (ADE=1) class.
    """
    _fix_base_score(model)
    # Guard: ensure all feature columns are float — rejects string-y dtypes early.
    X = X.astype(float)
    explainer = shap.TreeExplainer(model)
    shap_out = explainer.shap_values(X)

    # XGBoost binary classifiers return a 2-D array directly (one column per feature);
    # multi-output wrappers return a list — take index 1 (positive class).
    if isinstance(shap_out, list):
        return np.array(shap_out[1])
    return np.array(shap_out)


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------


def global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a DataFrame of mean(|SHAP|) per feature, sorted descending.

    Parameters
    ----------
    shap_values:
        Array of shape (n_samples, n_features) from explain_model().
    feature_names:
        Column names matching the second axis of shap_values.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap (sorted descending).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-admission explanation
# ---------------------------------------------------------------------------


def top_drivers_for_admission(
    model: XGBClassifier,
    X: pd.DataFrame,
    idx: int,
    n_top: int = 10,
) -> pd.DataFrame:
    """Return the top SHAP drivers for a single admission.

    This is the per-patient explanation a pharmacovigilance reviewer would see:
    positive SHAP values push toward ADE=1; negative values push toward ADE=0.

    Parameters
    ----------
    model:
        Fitted XGBClassifier.
    X:
        Feature matrix; row at position *idx* is the admission to explain.
    idx:
        Integer position (not label) of the admission in X.
    n_top:
        Number of top features to return (by absolute SHAP value).

    Returns
    -------
    pd.DataFrame
        Columns: feature, shap_value, feature_value.
        Sorted by |shap_value| descending (most influential first).
    """
    _fix_base_score(model)
    row = X.iloc[[idx]].astype(float)
    explainer = shap.TreeExplainer(model)
    shap_out = explainer.shap_values(row)

    if isinstance(shap_out, list):
        vals = np.array(shap_out[1]).ravel()
    else:
        vals = np.array(shap_out).ravel()

    feature_vals = row.values.ravel()
    result = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "shap_value": vals,
            "feature_value": feature_vals,
        }
    )
    result["abs_shap"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap", ascending=False).head(n_top)
    return result[["feature", "shap_value", "feature_value"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_global_bar(importance_df: pd.DataFrame, out_path: Path, n_top: int = 20) -> None:
    """Horizontal bar plot of top-N features by mean(|SHAP|)."""
    top = importance_df.head(n_top).iloc[::-1]  # reverse so highest is at top
    fig, ax = plt.subplots(figsize=(9, max(4, n_top * 0.35)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{n_top} global feature importances (ADE risk)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved global importance bar plot → %s", out_path)


def _save_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    out_path: Path,
    n_top: int = 20,
) -> None:
    """Beeswarm summary plot (value magnitude + direction per feature)."""
    fig, ax = plt.subplots(figsize=(9, max(4, n_top * 0.4)))
    shap.summary_plot(
        shap_values,
        X,
        max_display=n_top,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary (beeswarm) plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="SHAP explainability for the XGBoost ADE risk classifier"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--n-top",
        type=int,
        default=20,
        help="Number of top features to show in plots and console (default 20)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    model_path = processed_dir / "xgb_ade_model.json"
    test_path = processed_dir / "test.parquet"

    # Load model
    logger.info("Loading model from %s", model_path)
    model = XGBClassifier()
    model.load_model(str(model_path))

    # Load test features (same column selection as risk_classifier)
    logger.info("Loading test set from %s", test_path)
    test_df = pd.read_parquet(test_path)
    feature_cols = _get_feature_cols(test_df)
    X_test = test_df[feature_cols]
    logger.info("Explaining %d admissions × %d features", len(X_test), len(feature_cols))

    # SHAP values
    shap_values = explain_model(model, X_test)

    # Global importance table
    imp_df = global_importance(shap_values, feature_cols)

    csv_path = processed_dir / "shap_global_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    logger.info("Saved global importance table → %s", csv_path)

    bar_path = processed_dir / "shap_global_importance.png"
    _save_global_bar(imp_df, bar_path, n_top=args.n_top)

    summary_path = processed_dir / "shap_summary.png"
    _save_shap_summary(shap_values, X_test, summary_path, n_top=args.n_top)

    # Console report
    print("\n" + "=" * 55)
    print(f"  SHAP Global Feature Importance (top {args.n_top})")
    print("=" * 55)
    for i, row in imp_df.head(args.n_top).iterrows():
        bar = "█" * int(row["mean_abs_shap"] / imp_df["mean_abs_shap"].max() * 30)
        print(f"  {i+1:2d}. {row['feature']:<40s} {row['mean_abs_shap']:.4f}  {bar}")
    print("=" * 55)

    print(f"\nArtifacts written to {processed_dir}:")
    print(f"  {csv_path.name}")
    print(f"  {bar_path.name}")
    print(f"  {summary_path.name}")


if __name__ == "__main__":
    main()
