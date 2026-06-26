"""ADE model training orchestration entrypoint — STUB.

ENVIRONMENT: local (ETL + XGBoost) or Colab (full pipeline with NLP).

PURPOSE
-------
End-to-end pipeline:
  1. build_omop_tables    → processed parquets (local, 4 small tables)
  2. extract_ade_labels   → probabilistic ADE labels (Colab for note LF)
  3. build_features       → temporally valid feature matrix
  4. risk_classifier      → trained XGBoost/TabNet + MLflow metrics

Usage (local, after ETL):
  python -m ade_detection.train_ade --config config/config.yaml --stage features
  python -m ade_detection.train_ade --config config/config.yaml --stage train

TODO
----
  1. Parse --stage argument: etl | labels | features | train | all.
  2. Load config.yaml.
  3. Dispatch to the appropriate module based on stage.
  4. Wire MLflow experiment name from config or CLI flag.
"""

from __future__ import annotations


def main() -> None:
    """CLI entrypoint — TODO: implement argparse + stage dispatch."""
    raise NotImplementedError("TODO: implement train_ade.main()")


if __name__ == "__main__":
    main()
