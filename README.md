# ade-detection

[![CI](https://github.com/Vaishnavicodes/ade-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaishnavicodes/ade-detection/actions/workflows/ci.yml)

**Phase 1** of a healthcare ML platform: Adverse Drug Event (ADE) detection on MIMIC-III data,
mapped to OMOP-CDM format.

Depends on the shared library **[omop-mlops-platform](https://github.com/Vaishnavicodes/omop-mlops-platform)**
(pinned to `v0.1.0`) for all OMOP feature engineering and MLflow configuration.

---

## Hero Contributions

### Hero #1 — Weak-supervision ADE labeling (`extract_ade_labels.py`)
MIMIC-III has no ground-truth ADE labels. Instead of hand-labeling thousands of admissions, we
combine three noisy labeling functions via a **Snorkel LabelModel**:

| LF | Signal | Notes |
|----|--------|-------|
| LF-1 | ICD-9 E-codes E930–E949 | Structured, high-precision, low-recall |
| LF-2 | Clinical note mention matching | Requires NOTEEVENTS (Colab) |
| LF-3 | SIDER known drug–ADE pairs | External knowledge base |

The LabelModel learns each LF's accuracy and correlation, yielding a probabilistic `ade_prob`
per admission without any manual labeling.

### Hero #2 — Temporal validity rigor (`build_features.py` + `config.yaml`)
Every feature is computed using **only data recorded before the prediction timepoint**:

```
index_datetime = visit_start_datetime + prediction_timepoint_hours (default: 24h)
```

This is enforced by passing `index_dates` and `lookback_days` to every `omop_mlops.features`
call, preventing label leakage from post-prediction records into the training set.
The knobs live in `config/config.yaml` so experiments with different windows require zero
code changes.

---

## Architecture

```
MIMIC-III CSVs
     │
     ▼
build_omop_tables.py   ← ETL: PATIENTS/ADMISSIONS/DIAGNOSES_ICD/PRESCRIPTIONS → parquet
     │                   (NOTEEVENTS → note table on Colab only)
     ▼
extract_ade_labels.py  ← Snorkel weak supervision → ade_prob per admission (Colab)
     │
     ▼
build_features.py      ← omop_mlops.features (temporally gated at 24h post-admit)
     │
     ▼
risk_classifier.py     ← XGBoost (local) / TabNet (Colab GPU)
     │
     ▼
serve_ade.py           ← FastAPI /predict endpoint
```

**Dependency on shared platform:**

```
omop-mlops-platform @ git+https://github.com/Vaishnavicodes/omop-mlops-platform.git@v0.1.0
```

All OMOP feature engineering (`comorbidity_flags`, `medication_counts`, `make_feature_matrix`)
and MLflow wiring (`init_experiment`, `log_run`) come from the shared library.

---

## Install

```bash
git clone https://github.com/Vaishnavicodes/ade-detection.git
cd ade-detection
pip install -e .[dev]
```

The platform dependency resolves automatically from its GitHub tag.

## Run ETL (local — 4 small MIMIC tables)

```bash
# Place PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv in data/raw/
python -m ade_detection.data.build_omop_tables --config config/config.yaml
```

## Tests

```bash
pytest -q
ruff check .
black .
```

---

## Related repos

| Repo | Role |
|------|------|
| [omop-mlops-platform](https://github.com/Vaishnavicodes/omop-mlops-platform) | Shared OMOP + MLOps library |
| `trial-cohort` _(upcoming)_ | Clinical trial cohort selection |
