"""Temporal (time-aware) train/test split for ADE admission data.

A random split would leak future admissions into the training set — patients
admitted in the future share population-level temporal trends (seasonal illness,
formulary changes, coding drift) that the model should not see at train time.

The time-aware split sorts admissions by visit_start_datetime and puts the
earliest (1 - test_frac) fraction in train and the most-recent test_frac in test.
This mirrors deployment: the model is trained on historical data and evaluated on
admissions it has never seen, preserving the causal direction of time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def time_aware_split(
    features_df: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature matrix into train and test by visit_start_datetime.

    Admissions are sorted chronologically; the earliest (1 - test_frac) become
    the training set and the most recent test_frac become the test set.  No
    random shuffling is applied — shuffling would allow future admissions to
    appear in the training fold, constituting temporal leakage.

    Parameters
    ----------
    features_df:
        Feature matrix from build_features().  Must contain a
        ``visit_start_datetime`` column.
    test_frac:
        Fraction of admissions to place in the test set (default 0.2 = 20%).

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        Both DataFrames retain the same columns as *features_df*.
        train_df contains the earliest admissions; test_df the most recent.
    """
    if "visit_start_datetime" not in features_df.columns:
        raise ValueError("features_df must contain a 'visit_start_datetime' column")
    if not (0 < test_frac < 1):
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")

    df = features_df.sort_values("visit_start_datetime").reset_index(drop=True)
    n = len(df)
    cutoff = int(n * (1.0 - test_frac))

    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()

    train_start = train_df["visit_start_datetime"].min()
    train_end = train_df["visit_start_datetime"].max()
    test_start = test_df["visit_start_datetime"].min()
    test_end = test_df["visit_start_datetime"].max()

    print(f"Train: {len(train_df):,} admissions | " f"{train_start.date()} to {train_end.date()}")
    print(f"Test:  {len(test_df):,} admissions | " f"{test_start.date()} to {test_end.date()}")

    if "ade_label" in df.columns:
        train_pos = 100.0 * train_df["ade_label"].mean()
        test_pos = 100.0 * test_df["ade_label"].mean()
        print(f"ADE positive rate  train: {train_pos:.1f}%  test: {test_pos:.1f}%")

    return train_df, test_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Time-aware train/test split of the ADE feature matrix"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of (most-recent) admissions to hold out for testing (default 0.2)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    features_path = processed_dir / "features.parquet"

    logger.info("Loading feature matrix from %s", features_path)
    features = pd.read_parquet(features_path)

    train_df, test_df = time_aware_split(features, test_frac=args.test_frac)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"\nWrote {len(train_df):,} rows → {train_path}")
    print(f"Wrote {len(test_df):,} rows  → {test_path}")


if __name__ == "__main__":
    main()
