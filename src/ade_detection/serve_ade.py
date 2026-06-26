"""FastAPI inference server for ADE risk scoring — STUB.

ENVIRONMENT: local or any Docker/cloud host with the trained model artifact.

PURPOSE
-------
Expose the trained XGBoost ADE risk model as a REST API so downstream
clinical dashboards or alerting systems can query risk scores in real time.

INTENDED ENDPOINTS
------------------
  POST /predict
    Input JSON:
      {
        "person_id": int,
        "visit_occurrence_id": int,
        "features": { <column>: <value>, ... }  # same schema as feature matrix
      }
    Output JSON:
      { "ade_prob": float, "ade_label": 0|1 }

  GET  /health  → {"status": "ok", "model_version": str}

INTENDED DEPENDENCIES
---------------------
  fastapi>=0.110
  uvicorn>=0.29
  xgboost>=2.0
  (These are not in pyproject.toml yet — add when implementing.)

TODO
----
  1. Load trained XGBoost model from MLflow artifact or local path on startup.
  2. Define Pydantic request/response models.
  3. Implement /predict: validate input → run model.predict_proba → return score.
  4. Add /health endpoint returning model version from omop_mlops.__version__.
  5. Dockerfile for containerised deployment.
"""

from __future__ import annotations


def create_app():
    """Build and return the FastAPI application instance.

    TODO:
      from fastapi import FastAPI
      app = FastAPI(title="ADE Risk Scorer", version="0.1.0")
      # register routers
      return app
    """
    raise NotImplementedError("TODO: implement FastAPI app — add fastapi/uvicorn to deps first")


if __name__ == "__main__":
    raise NotImplementedError("TODO: uvicorn.run(create_app(), host='0.0.0.0', port=8000)")
