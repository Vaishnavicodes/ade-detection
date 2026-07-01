"""FastAPI inference server for the calibrated ADE risk model.

Serves pre-computed admission features — a feature service would sit upstream
in production, joining OMOP CDM tables to produce the 13-column input vector.

Endpoints
---------
GET  /         brief service info + link to /docs
GET  /health   liveness + model-loaded flag
POST /predict  takes AdmissionFeatures, returns RiskResponse with calibrated
               probability, clinical risk flag, and per-admission SHAP top drivers

Run
---
    uvicorn ade_detection.serving.serve_ade:app --reload
    python -m ade_detection.serving.serve_ade
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from ade_detection.models.risk_classifier import _find_sensitivity_threshold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
    "age_at_admission",
    "is_female",
    "n_drugs_first24h",
    "n_distinct_drugs_first24h",
    "prior_comorb_myocardial_infarction",
    "prior_comorb_congestive_heart_failure",
    "prior_comorb_diabetes_uncomplicated",
    "prior_comorb_chronic_pulmonary_disease",
    "n_prior_conditions",
    "n_prior_drug_exposures",
]

# Fallback threshold when test.parquet is absent at serve time.
# Computed from the calibrated model on the MIMIC test split (2026-07); 80% sensitivity.
_THRESHOLD_FALLBACK: float = 0.0747

_PROCESSED_DIR = Path(
    os.getenv(
        "ADE_PROCESSED_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "processed"),
    )
)

# ---------------------------------------------------------------------------
# App state — populated once at startup
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _load_artifacts() -> None:
    """Load model files and compute threshold; raise clearly if anything is missing."""
    cal_path = _PROCESSED_DIR / "xgb_ade_calibrated.joblib"
    xgb_path = _PROCESSED_DIR / "xgb_ade_model.json"

    missing = [p for p in (cal_path, xgb_path) if not p.exists()]
    if missing:
        paths = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required model artifacts not found: {paths}\n"
            "Run `python -m ade_detection.models.calibrate` to generate them."
        )

    logger.info("Loading calibrated model from %s", cal_path)
    # calibrate.py was run as __main__, so pickle stored the class as
    # __main__._PrefitCalibratedModel. Register it before loading so joblib
    # can find it regardless of the calling context (pytest, uvicorn, etc.).
    import sys

    from ade_detection.models.calibrate import _PrefitCalibratedModel

    sys.modules["__main__"].__dict__.setdefault("_PrefitCalibratedModel", _PrefitCalibratedModel)
    _state["calibrated"] = joblib.load(cal_path)

    logger.info("Loading raw XGBoost model for SHAP TreeExplainer from %s", xgb_path)
    raw_xgb = XGBClassifier()
    raw_xgb.load_model(str(xgb_path))
    _state["explainer"] = shap.TreeExplainer(raw_xgb)

    # Threshold: recompute from test set at startup; fall back to constant if absent
    test_path = _PROCESSED_DIR / "test.parquet"
    if test_path.exists():
        import pandas as pd

        test_df = pd.read_parquet(test_path)
        X_test = test_df[FEATURE_COLS].values.astype(float)
        y_test = test_df["ade_label"].values.astype(int)
        y_prob = _state["calibrated"].predict_proba(X_test)[:, 1]
        threshold = _find_sensitivity_threshold(y_test, y_prob, target_sensitivity=0.80)
        logger.info("80%%-sensitivity threshold computed from test set: %.6f", threshold)
    else:
        threshold = _THRESHOLD_FALLBACK
        logger.warning(
            "test.parquet not found at %s; using hardcoded fallback threshold %.4f",
            test_path,
            threshold,
        )

    _state["threshold"] = threshold
    _state["model_loaded"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    _load_artifacts()
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AdmissionFeatures(BaseModel):
    gender_concept_id: int = Field(
        ...,
        examples=[8507],
        description="OMOP gender concept (8507=Male, 8532=Female)",
    )
    race_concept_id: int = Field(
        ...,
        examples=[8527],
        description="OMOP race concept ID",
    )
    ethnicity_concept_id: int = Field(
        ...,
        examples=[38003564],
        description="OMOP ethnicity concept ID",
    )
    age_at_admission: int = Field(
        ...,
        examples=[65],
        description="Patient age in years at admission",
    )
    is_female: int = Field(
        ...,
        examples=[0],
        description="Binary sex flag: 1=female, 0=male/other",
    )
    n_drugs_first24h: int = Field(
        ...,
        examples=[8],
        description="Total drug administrations recorded in the first 24 h",
    )
    n_distinct_drugs_first24h: int = Field(
        ...,
        examples=[5],
        description="Distinct drug concept IDs administered in the first 24 h",
    )
    prior_comorb_myocardial_infarction: int = Field(
        ...,
        examples=[0],
        description="Prior myocardial infarction comorbidity flag (0/1)",
    )
    prior_comorb_congestive_heart_failure: int = Field(
        ...,
        examples=[1],
        description="Prior congestive heart failure comorbidity flag (0/1)",
    )
    prior_comorb_diabetes_uncomplicated: int = Field(
        ...,
        examples=[0],
        description="Prior uncomplicated diabetes comorbidity flag (0/1)",
    )
    prior_comorb_chronic_pulmonary_disease: int = Field(
        ...,
        examples=[0],
        description="Prior chronic pulmonary disease comorbidity flag (0/1)",
    )
    n_prior_conditions: int = Field(
        ...,
        examples=[4],
        description="Total prior OMOP condition occurrences",
    )
    n_prior_drug_exposures: int = Field(
        ...,
        examples=[12],
        description="Total prior OMOP drug exposures",
    )


class DriverDetail(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class RiskResponse(BaseModel):
    ade_risk_probability: float
    risk_flag: bool
    threshold_used: float
    top_drivers: list[DriverDetail]


# ---------------------------------------------------------------------------
# App + endpoints
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADE Risk Scorer",
    version="0.1.0",
    description=(
        "Calibrated adverse drug event (ADE) risk scoring from OMOP CDM admission "
        "features. Serves pre-computed feature vectors — a feature service would sit "
        "upstream in production."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "ADE Risk Scorer",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _state.get("model_loaded", False)}


@app.post("/predict", response_model=RiskResponse)
def predict(admission: AdmissionFeatures) -> RiskResponse:
    calibrated = _state["calibrated"]
    explainer: shap.TreeExplainer = _state["explainer"]
    threshold: float = _state["threshold"]

    row = np.array(
        [[getattr(admission, col) for col in FEATURE_COLS]],
        dtype=float,
    )

    prob = float(calibrated.predict_proba(row)[0, 1])
    risk_flag = prob >= threshold

    shap_out = explainer.shap_values(row)
    if isinstance(shap_out, list):
        shap_vals = np.array(shap_out[1]).ravel()
    else:
        shap_vals = np.array(shap_out).ravel()

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_drivers = [
        DriverDetail(
            feature=FEATURE_COLS[i],
            shap_value=float(shap_vals[i]),
            direction="increases_risk" if shap_vals[i] >= 0 else "decreases_risk",
        )
        for i in top_idx
    ]

    return RiskResponse(
        ade_risk_probability=prob,
        risk_flag=risk_flag,
        threshold_used=threshold,
        top_drivers=top_drivers,
    )


if __name__ == "__main__":
    uvicorn.run(
        "ade_detection.serving.serve_ade:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
