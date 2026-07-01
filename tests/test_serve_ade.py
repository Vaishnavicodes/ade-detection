"""Tests for the ADE risk scoring FastAPI service (ade_detection.serving.serve_ade).

Uses fastapi.testclient.TestClient backed by httpx.  The /predict tests require
model artifacts in data/processed/ — they are skipped automatically if the files
are absent so CI stays green without committing data artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Sentinel: are model artifacts present?
# ---------------------------------------------------------------------------

_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_MODELS_PRESENT = (_PROCESSED_DIR / "xgb_ade_calibrated.joblib").exists() and (
    _PROCESSED_DIR / "xgb_ade_model.json"
).exists()

_SKIP_MSG = "model artifacts not present in data/processed/ — skipped in CI"

# ---------------------------------------------------------------------------
# Example payload (matches FEATURE_COLS order; values are clinically plausible)
# ---------------------------------------------------------------------------

_EXAMPLE_PAYLOAD = {
    "gender_concept_id": 8507,
    "race_concept_id": 8527,
    "ethnicity_concept_id": 38003564,
    "age_at_admission": 65,
    "is_female": 0,
    "n_drugs_first24h": 8,
    "n_distinct_drugs_first24h": 5,
    "prior_comorb_myocardial_infarction": 0,
    "prior_comorb_congestive_heart_failure": 1,
    "prior_comorb_diabetes_uncomplicated": 0,
    "prior_comorb_chronic_pulmonary_disease": 0,
    "n_prior_conditions": 4,
    "n_prior_drug_exposures": 12,
}

# ---------------------------------------------------------------------------
# Shared client fixture — skips the whole module when models are absent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    if not _MODELS_PRESENT:
        pytest.skip(_SKIP_MSG)
    from fastapi.testclient import TestClient

    from ade_detection.serving.serve_ade import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict — valid payload
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_returns_200(client):
    resp = client.post("/predict", json=_EXAMPLE_PAYLOAD)
    assert resp.status_code == 200


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_probability_in_unit_interval(client):
    resp = client.post("/predict", json=_EXAMPLE_PAYLOAD)
    prob = resp.json()["ade_risk_probability"]
    assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_risk_flag_is_bool(client):
    body = client.post("/predict", json=_EXAMPLE_PAYLOAD).json()
    assert isinstance(body["risk_flag"], bool)


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_threshold_used_is_positive(client):
    body = client.post("/predict", json=_EXAMPLE_PAYLOAD).json()
    assert body["threshold_used"] > 0.0


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_top_drivers_nonempty(client):
    body = client.post("/predict", json=_EXAMPLE_PAYLOAD).json()
    assert len(body["top_drivers"]) > 0


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_top_drivers_structure(client):
    body = client.post("/predict", json=_EXAMPLE_PAYLOAD).json()
    for driver in body["top_drivers"]:
        assert "feature" in driver
        assert isinstance(driver["shap_value"], float)
        assert driver["direction"] in ("increases_risk", "decreases_risk")


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_risk_flag_consistent_with_probability(client):
    body = client.post("/predict", json=_EXAMPLE_PAYLOAD).json()
    expected_flag = body["ade_risk_probability"] >= body["threshold_used"]
    assert body["risk_flag"] == expected_flag


# ---------------------------------------------------------------------------
# /predict — invalid payload (validation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in _EXAMPLE_PAYLOAD.items() if k != "age_at_admission"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


@pytest.mark.skipif(not _MODELS_PRESENT, reason=_SKIP_MSG)
def test_predict_wrong_type_returns_422(client):
    payload = {**_EXAMPLE_PAYLOAD, "age_at_admission": "old"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
