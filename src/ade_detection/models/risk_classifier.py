"""XGBoost / TabNet ADE risk classifier — STUB.

ENVIRONMENT: local (XGBoost CPU) or Colab (TabNet GPU).

PURPOSE
-------
Train a tabular risk classifier on the feature matrix produced by
build_features.py.  Two candidate models:

  XGBoostClassifier
    Fast, interpretable via SHAP, runs locally.
    Input: structured OMOP features (comorbidities, drug counts, measurements).
    Output: ADE risk probability per admission.

  TabNetClassifier (pytorch-tabnet)
    Attention-based tabular DL model; better on high-cardinality sparse features.
    Requires GPU — run on Colab.
    TODO: pip install pytorch-tabnet on Colab.

INTENDED INPUTS
---------------
  feature_matrix_df  : output of build_features.build_feature_matrix()
                       columns: person_id, visit_occurrence_id, [features...], ade_label
  model_type         : "xgboost" | "tabnet"
  config             : dict from config/config.yaml (for hyperparams)

INTENDED OUTPUT
---------------
  Trained model artifact (saved to mlruns/ via omop_mlops.mlflow_config.log_run).
  Evaluation metrics: AUROC, AUPRC, F1 at 0.5 threshold.

TODO
----
  1. Split feature_matrix into train/val/test respecting temporal order
     (admissions before cutoff date → train; after → test). Do NOT use
     random splits — temporal leakage would inflate metrics.
  2. Train XGBoostClassifier; log hyperparams + AUROC to MLflow via log_run.
  3. (Colab) train TabNet; compare AUROC.
  4. Save best model and feature importance / SHAP values.
"""

from __future__ import annotations

import pandas as pd


def temporal_train_test_split(
    feature_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    test_cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature_df into train/test by admission date (never random).

    TODO: join visit_start_date on visit_occurrence_id, split at test_cutoff_date.
    """
    raise NotImplementedError("TODO: implement temporal_train_test_split")


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None):
    """Train an XGBoostClassifier and return the fitted model.

    TODO:
      from xgboost import XGBClassifier
      model = XGBClassifier(**(params or {}))
      model.fit(X_train, y_train)
      return model
    """
    raise NotImplementedError("TODO: implement train_xgboost")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute AUROC, AUPRC, F1 on test set.

    TODO: use sklearn.metrics.roc_auc_score, average_precision_score, f1_score.
    """
    raise NotImplementedError("TODO: implement evaluate")


def train_and_log(
    feature_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    test_cutoff_date: str,
    model_type: str = "xgboost",
    mlflow_experiment: str = "ade-detection",
) -> None:
    """Full train+eval pipeline with MLflow logging.

    TODO:
      from omop_mlops.mlflow_config import init_experiment, log_run
      init_experiment(mlflow_experiment)
      X_train, X_test, y_train, y_test = temporal_train_test_split(...)
      model = train_xgboost(X_train, y_train)
      metrics = evaluate(model, X_test, y_test)
      with log_run("xgboost-v1", params={...}, metrics=metrics):
          pass
    """
    raise NotImplementedError("TODO: implement train_and_log")
