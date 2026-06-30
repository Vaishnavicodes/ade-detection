"""Apply Snorkel LFs and train a LabelModel to produce probabilistic ADE labels.

Pipeline:
  1. Apply the 5 local LFs via PandasLFApplier -> label matrix L (n, 5).
  2. Print LFAnalysis coverage / overlap / conflict summary.
  3. Train LabelModel(cardinality=2, class_balance=[1-ADE_BASE_RATE, ADE_BASE_RATE]) on L.
  4. predict_proba -> ade_prob (P(ADE=1)), threshold at 0.5 -> ade_label.
  5. Return input frame augmented with ade_prob and ade_label.

Usage::

    # Normal run — trains one model, writes ade_labels.parquet
    python -m ade_detection.labeling.run_label_model

    # Compare-priors experiment — prints a summary table, does NOT write parquet
    python -m ade_detection.labeling.run_label_model --compare 0.05 0.10 0.15 0.20

Prerequisite: labeling_frame.parquet must exist in processed_data_dir.
Run build_labeling_frame first::

    python -m ade_detection.labeling.build_labeling_frame
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from ade_detection.labeling.labeling_functions import get_local_lfs

logger = logging.getLogger(__name__)

# Snorkel cardinality-2 convention: column 0 = NOT_ADE, column 1 = ADE
_ADE_COL = 1

# Chosen ADE base-rate prior after running --compare 0.10 0.15 0.20:
#   0.10 → positive rate ~12%, LF accuracies skewed low for positive LFs
#   0.15 → positive rate ~7.9%, LF accuracies balanced across all 5 LFs  ← chosen
#   0.20 → positive rate ~7.9%, same realized rate but prior exceeds it (overconfident)
# 0.15 stabilizes at the same realized rate as 0.20 while keeping the prior consistent
# with the outcome and giving fairer learned accuracies to the positive LFs.
ADE_BASE_RATE = 0.15


def run(labeling_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply 5 local LFs, train LabelModel, return frame with ade_prob + ade_label.

    Parameters
    ----------
    labeling_frame:
        Output of build_labeling_frame() — one row per admission.

    Returns
    -------
    pd.DataFrame
        Input frame plus:
          ade_prob   float  P(ADE=1) from LabelModel.predict_proba
          ade_label  int    1 if ade_prob >= 0.5 else 0
    """
    from snorkel.labeling import LFAnalysis, PandasLFApplier
    from snorkel.labeling.model import LabelModel

    lfs = get_local_lfs()

    # Apply LFs -> label matrix  shape (n_admissions, n_lfs)
    logger.info("Applying %d LFs to %d admissions...", len(lfs), len(labeling_frame))
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(labeling_frame)

    # Coverage / overlap / conflict report
    print("\n--- LF Coverage / Overlap / Conflict Summary ---")
    print(LFAnalysis(L=L, lfs=lfs).lf_summary())

    # Train LabelModel with a realistic class-balance prior.
    # Without it, admissions where all 5 LFs ABSTAIN default to ~50% positive,
    # inflating the ADE rate to 80%+.  The 90/10 prior anchors uncovered rows
    # to the true base rate (~5-15% in literature).
    logger.info(
        "Training LabelModel (cardinality=2, n_epochs=500, seed=42, "
        "class_balance=[%.2f, %.2f])...",
        1 - ADE_BASE_RATE,
        ADE_BASE_RATE,
    )
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(
        L_train=L,
        n_epochs=500,
        seed=42,
        log_freq=100,
        class_balance=[1 - ADE_BASE_RATE, ADE_BASE_RATE],
    )

    # Probabilistic labels: column 1 = P(ADE=1)
    probs = label_model.predict_proba(L)
    ade_probs = probs[:, _ADE_COL]

    result = labeling_frame.copy()
    result["ade_prob"] = ade_probs
    result["ade_label"] = (ade_probs >= 0.5).astype(int)

    # Summary
    n_total = len(result)
    n_ade = int(result["ade_label"].sum())
    print("\n--- ADE Label Summary ---")
    print(f"  Total admissions      : {n_total:,}")
    print(f"  Labeled ADE (>=0.5)   : {n_ade:,} ({100 * n_ade / n_total:.1f}%)")
    print(f"  Mean ade_prob         : {ade_probs.mean():.4f}")

    # LF learned accuracies (API varies between snorkel versions)
    try:
        weights = label_model.get_weights()
        print("\n  LF learned accuracies:")
        for lf, w in zip(lfs, weights):
            print(f"    {lf.name}: {w:.4f}")
    except AttributeError:
        pass  # older snorkel versions do not expose get_weights()

    return result


def compare_priors(labeling_frame: pd.DataFrame, priors: list[float]) -> pd.DataFrame:
    """Train a fresh LabelModel per prior and return a side-by-side comparison table.

    Applies the LFs once (the label matrix L is the same regardless of class_balance),
    then trains a separate LabelModel for each prior value. Useful for choosing
    ADE_BASE_RATE before committing to a final labeling run.

    Parameters
    ----------
    labeling_frame:
        Output of build_labeling_frame() — one row per admission.
    priors:
        ADE base-rate values to sweep (e.g. [0.05, 0.10, 0.15, 0.20]).
        Each p is used as class_balance=[1-p, p].

    Returns
    -------
    pd.DataFrame
        One row per prior. Columns:
          prior               float  — the ADE base-rate value used
          positive_rate_pct   float  — % admissions with ade_label == 1
          mean_ade_prob       float  — mean P(ADE=1) across all admissions
          acc_<lf_name>       float  — learned accuracy per LF (NaN if unavailable)
    """
    from snorkel.labeling import LFAnalysis, PandasLFApplier
    from snorkel.labeling.model import LabelModel

    lfs = get_local_lfs()
    lf_names = [lf.name for lf in lfs]

    # Apply LFs once — L does not change across priors
    logger.info(
        "Applying %d LFs to %d admissions (once for all priors)...",
        len(lfs),
        len(labeling_frame),
    )
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(labeling_frame)

    print("\n--- LF Coverage / Overlap / Conflict Summary ---")
    print(LFAnalysis(L=L, lfs=lfs).lf_summary())

    rows = []
    for p in priors:
        logger.info("Training LabelModel with ADE prior=%.2f ...", p)
        lm = LabelModel(cardinality=2, verbose=False)
        lm.fit(
            L_train=L,
            n_epochs=500,
            seed=42,
            log_freq=500,  # suppressed; verbose=False already silences output
            class_balance=[1 - p, p],
        )
        probs = lm.predict_proba(L)[:, _ADE_COL]
        labels = (probs >= 0.5).astype(int)

        record: dict = {
            "prior": p,
            "positive_rate_pct": round(100.0 * float(labels.mean()), 2),
            "mean_ade_prob": round(float(probs.mean()), 4),
        }

        try:
            weights = lm.get_weights()
            for name, w in zip(lf_names, weights):
                record[f"acc_{name}"] = round(float(w), 4)
        except AttributeError:
            for name in lf_names:
                record[f"acc_{name}"] = float("nan")

        rows.append(record)

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Train Snorkel LabelModel and write probabilistic ADE labels"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--compare",
        nargs="+",
        type=float,
        metavar="PRIOR",
        help=(
            "Compare-priors mode: train one LabelModel per prior and print a summary "
            "table. Does NOT write ade_labels.parquet. "
            "Example: --compare 0.05 0.10 0.15 0.20"
        ),
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])

    frame_path = processed_dir / "labeling_frame.parquet"
    logger.info("Loading labeling frame from %s", frame_path)
    frame = pd.read_parquet(frame_path)

    if args.compare:
        summary = compare_priors(frame, args.compare)
        print("\n--- Prior Comparison ---")
        print(summary.to_string(index=False))
        return

    # Default: single run, write parquet
    result = run(frame)

    out_path = processed_dir / "ade_labels.parquet"
    result.to_parquet(out_path, index=False)

    n_total = len(result)
    n_ade = int(result["ade_label"].sum())
    print(f"\nWrote {n_total:,} rows -> {out_path}")
    print(f"Positive rate: {100 * n_ade / n_total:.1f}%")


if __name__ == "__main__":
    main()
