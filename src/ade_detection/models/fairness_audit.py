"""Subgroup fairness audit for the ADE risk classifier.

Measures PERFORMANCE parity across demographic subgroups — sensitivity,
specificity, precision, and PR-AUC at the ~0.80-sensitivity operating
threshold derived from the calibrated model's probabilities.

IMPORTANT: Low SHAP feature importance for demographic columns does NOT
guarantee performance parity across subgroups.  Correlated features
(e.g. age correlating with both polypharmacy and ADE risk) and differential
label completeness across sites or demographic groups can produce meaningful
sensitivity gaps even when the model never directly uses race or gender as
inputs.  This audit measures outcome metrics empirically to catch those gaps.

Audit dimensions:
  is_female      — derived from PATIENTS.GENDER (M → 0, F → 1); model feature
  ethnicity_group — bucketed from ADMISSIONS.ETHNICITY (WHITE/BLACK/HISPANIC/
                    ASIAN/OTHER/UNKNOWN); audit-only, NOT a model feature
  age_band       — derived from age_at_admission (<40/40-64/65-79/80+)

Output artifacts (all gitignored, written to data/processed/):
  fairness_<dimension>.csv — per-subgroup metrics table
  fairness_<dimension>.png — sensitivity bar chart annotated with n

Usage::

    python -m ade_detection.models.fairness_audit
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
from sklearn.metrics import average_precision_score, confusion_matrix

from ade_detection.models.calibrate import _PrefitCalibratedModel
from ade_detection.models.risk_classifier import _find_sensitivity_threshold, _get_feature_cols

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit dimensions
# ---------------------------------------------------------------------------

# is_female: 0/1 numeric model feature derived from PATIENTS.GENDER (M/F)
_IS_FEMALE_LABELS: dict[int, str] = {0: "Male", 1: "Female"}

# Dimensions and their optional int→str label maps.
# ethnicity_group and age_band use string values directly (no int mapping needed).
_DIMENSIONS: dict[str, dict[int, str] | None] = {
    "is_female": _IS_FEMALE_LABELS,
    "ethnicity_group": None,  # already readable strings (WHITE/BLACK/HISPANIC/…)
    "age_band": None,  # already readable strings (<40/40-64/…)
}

# ---------------------------------------------------------------------------
# Age banding
# ---------------------------------------------------------------------------

_AGE_BANDS = [
    (0, 40, "<40"),
    (40, 65, "40–64"),
    (65, 80, "65–79"),
    (80, 999, "80+"),
]


def _add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'age_band' string column derived from age_at_admission."""
    df = df.copy()
    labels = pd.Series("Unknown", index=df.index, dtype=str)
    for lo, hi, name in _AGE_BANDS:
        mask = (df["age_at_admission"] >= lo) & (df["age_at_admission"] < hi)
        labels[mask] = name
    df["age_band"] = labels
    return df


# ---------------------------------------------------------------------------
# Per-subgroup metrics
# ---------------------------------------------------------------------------

_MIN_POS_FOR_AUC = 5  # minimum positives to report PR-AUC


def _subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute fairness metrics for one subgroup."""
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n == 0:
        return {
            "n": 0,
            "n_positive": 0,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "pr_auc": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(int)

    if n_pos == 0 or n_pos == n:
        # Can't compute confusion matrix with one class
        return {
            "n": n,
            "n_positive": n_pos,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "pr_auc": float("nan"),
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    if n_pos >= _MIN_POS_FOR_AUC:
        pr_auc: float | str = float(average_precision_score(y_true, y_prob))
    else:
        pr_auc = "N/A"

    return {
        "n": n,
        "n_positive": n_pos,
        "sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else float("nan"),
        "specificity": round(specificity, 4) if not np.isnan(specificity) else float("nan"),
        "precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
        "pr_auc": round(pr_auc, 4) if isinstance(pr_auc, float) else pr_auc,
    }


# ---------------------------------------------------------------------------
# Audit one demographic dimension
# ---------------------------------------------------------------------------

_SMALL_N = 100  # threshold for "small sample" warning


def audit_dimension(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    dim_col: str,
    label_map: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Compute per-subgroup metrics for one demographic dimension.

    Parameters
    ----------
    df : DataFrame with 'ade_label' and dim_col columns.
    y_prob : calibrated probabilities aligned with df rows.
    threshold : operating threshold for sensitivity/specificity/precision.
    dim_col : column to group by (e.g. 'gender_concept_id', 'age_band').
    label_map : optional int→str mapping for concept IDs.

    Returns
    -------
    (table, gap)
        table — DataFrame with columns: subgroup, label, n, n_positive,
                sensitivity, specificity, precision, pr_auc, note
        gap   — max-min sensitivity across subgroups with n >= _SMALL_N,
                or None if fewer than 2 qualifying subgroups.
    """
    y_true = df["ade_label"].values.astype(int)
    rows = []

    for val, idx in df.groupby(dim_col, sort=False).groups.items():
        yt = y_true[idx]
        yp = y_prob[idx]
        m = _subgroup_metrics(yt, yp, threshold)
        label = label_map.get(int(val), str(val)) if label_map else str(val)
        note = "small sample — interpret with caution" if m["n"] < _SMALL_N else ""
        rows.append({"subgroup": val, "label": label, **m, "note": note})

    table = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    # Sensitivity gap across subgroups with n >= _SMALL_N
    large = table[table["n"] >= _SMALL_N]
    sens_vals = pd.to_numeric(large["sensitivity"], errors="coerce").dropna()
    gap: float | None = None
    if len(sens_vals) >= 2:
        gap = float(sens_vals.max() - sens_vals.min())

    return table, gap


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save_bar_chart(table: pd.DataFrame, dim_name: str, out_path: Path) -> None:
    """Bar chart of sensitivity by subgroup, annotated with n."""
    plot_df = table.copy()
    plot_df["sensitivity_num"] = pd.to_numeric(plot_df["sensitivity"], errors="coerce")
    plot_df = plot_df.dropna(subset=["sensitivity_num"])

    if plot_df.empty:
        logger.warning("No numeric sensitivity values for %s — skipping plot", dim_name)
        return

    labels = plot_df["label"].astype(str).tolist()
    values = plot_df["sensitivity_num"].tolist()
    ns = plot_df["n"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white")

    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0.80, color="red", linestyle="--", linewidth=1, label="0.80 target")
    ax.set_ylim(0, min(1.05, max(values) + 0.15))
    ax.set_xlabel(dim_name.replace("_", " ").title())
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ADE sensitivity by {dim_name.replace('_', ' ')}")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart → %s", out_path)


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------


def run_fairness_audit(
    test_df: pd.DataFrame,
    cal_model: _PrefitCalibratedModel,
    target_sensitivity: float = 0.80,
) -> dict[str, tuple[pd.DataFrame, float | None]]:
    """Run the full fairness audit across all demographic dimensions.

    Parameters
    ----------
    test_df : Hold-out test set (test.parquet schema).
    cal_model : Fitted _PrefitCalibratedModel from calibrate.py.
    target_sensitivity : Target sensitivity for threshold derivation (default 0.80).

    Returns
    -------
    dict mapping dimension name → (table DataFrame, sensitivity gap or None)
    """
    feature_cols = _get_feature_cols(test_df)
    X_test = test_df[feature_cols].values.astype(float)
    y_true = test_df["ade_label"].values.astype(int)

    y_prob = cal_model.predict_proba(X_test)[:, 1]
    threshold = _find_sensitivity_threshold(y_true, y_prob, target_sensitivity)
    logger.info(
        "Operating threshold: %.4f  (target sensitivity %.0f%%)",
        threshold,
        target_sensitivity * 100,
    )

    # Derive age_band
    test_df = _add_age_band(test_df)

    results: dict[str, tuple[pd.DataFrame, float | None]] = {}
    for dim_col, label_map in _DIMENSIONS.items():
        if dim_col not in test_df.columns:
            logger.warning("Column %s not found in test_df — skipping", dim_col)
            continue
        table, gap = audit_dimension(test_df, y_prob, threshold, dim_col, label_map)
        results[dim_col] = (table, gap)

    return results


# ---------------------------------------------------------------------------
# Console + file output
# ---------------------------------------------------------------------------


def _print_table(dim_col: str, table: pd.DataFrame, gap: float | None) -> None:
    unique_vals = table["subgroup"].nunique()

    print(f"\n{'=' * 65}")
    print(f"  Fairness audit: {dim_col}")
    print(f"{'=' * 65}")

    if unique_vals <= 1:
        print(
            f"  NOTE: All {len(table)} records share a single value "
            f"({table['subgroup'].iloc[0]!r})."
        )
        print("  MIMIC OMOP ETL did not populate this field — subgroup comparison not possible.")

    print(table.to_string(index=False))

    if gap is not None:
        flag = " *** HIGH GAP — investigate ***" if gap > 0.10 else ""
        print(f"\n  Sensitivity GAP (n≥{_SMALL_N}): {gap:.4f}{flag}")
    else:
        print(f"\n  Sensitivity GAP: N/A (fewer than 2 subgroups with n≥{_SMALL_N})")

    small = table[table["n"] < _SMALL_N]
    if not small.empty:
        print(f"  Small-sample subgroups (n<{_SMALL_N}): {small['label'].tolist()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Subgroup fairness audit for ADE risk classifier")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.80,
        help="Sensitivity target for operating threshold (default 0.80)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])

    logger.info("Loading test set…")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Loading calibrated model…")
    cal_model = joblib.load(processed_dir / "xgb_ade_calibrated.joblib")

    results = run_fairness_audit(test_df, cal_model, args.target_sensitivity)

    for dim_col, (table, gap) in results.items():
        _print_table(dim_col, table, gap)

        csv_path = processed_dir / f"fairness_{dim_col}.csv"
        table.to_csv(csv_path, index=False)
        logger.info("Saved table → %s", csv_path)

        png_path = processed_dir / f"fairness_{dim_col}.png"
        _save_bar_chart(table, dim_col, png_path)

    print("\nDone. Artifacts written to", processed_dir)


if __name__ == "__main__":
    main()
