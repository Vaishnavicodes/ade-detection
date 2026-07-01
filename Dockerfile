FROM python:3.11-slim

# libgomp1: XGBoost OpenMP runtime — not present on slim images by default.
# git: pip needs it to clone omop-mlops-platform from its GitHub tag.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package metadata and source first so pip can build and install the wheel.
# data/ is excluded via .dockerignore — model artifacts are NEVER baked into the
# image (MIMIC DUA).  They must be mounted at runtime:
#   docker run -p 8000:8000 -v ${PWD}/data/processed:/app/data/processed ade-detection:latest
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# When the package is installed non-editably, serve_ade.py resolves __file__ to
# site-packages, not /app, so the parents[3] fallback would miss the mount point.
# This env var overrides it to the correct container path.
ENV ADE_PROCESSED_DIR=/app/data/processed

EXPOSE 8000

CMD ["uvicorn", "ade_detection.serving.serve_ade:app", "--host", "0.0.0.0", "--port", "8000"]
