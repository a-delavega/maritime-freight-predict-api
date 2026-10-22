<div align="center">

# Maritime Freight & Navigation Predict API

[![CI/CD Pipeline](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml/badge.svg)](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-success.svg)](https://github.com/a-delavega/maritime-freight-predict-api)
[![Security](https://img.shields.io/badge/security-Zero%20Pickle%20%2F%20Safe%20JSON-brightgreen.svg)](#design-decisions)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C.svg?logo=prometheus&logoColor=white)](http://127.0.0.1:8000/metrics)
[![Architecture](https://img.shields.io/badge/architecture-Clean%20%2F%20Hexagonal-orange.svg)](#architecture)

A hardened, production-grade ML inference microservice that classifies the overall difficulty of maritime navigation exercises (`LOW`, `MEDIUM`, or `HIGH`) from situational encounters. Built with FastAPI, native XGBoost (Zero-Pickle / JSON), and strict Clean Architecture.

[Overview](#overview) • [Architecture](#architecture) • [Playground](#playground) • [Quickstart](#quickstart) • [API](#api) • [Design Decisions](#design-decisions)

---

</div>

## Overview

This project serves as an **educational showcase and production-grade engineering reference** for taking machine learning models out of experimental scripts and deploying them into secure, resilient, and observable web microservices.

Rather than relying on common data science shortcuts (e.g. `pickle`, ad-hoc data transformations in HTTP handlers, or unconstrained decision trees), it strictly applies modern software engineering and MLOps practices:

- **Clean / Hexagonal Architecture:** The domain and application logic are completely decoupled from frameworks (FastAPI) and ML runtimes (XGBoost). The model is consumed strictly via an abstract port (`DifficultyClassifierPort`).
- **Zero-Pickle / Anti-RCE Security:** No Python object deserialization (`joblib`/`pickle`). The model uses native C++ `model.json` format and feature scaling parameters are isolated in a static `scaler_params.json`.
- **Zero Data Leakage Pipeline:** The training workflow (`ml_pipeline/train_model.py`) trains on stratified splits without synthetic distortion (no SMOTE) and exports frozen z-score constants.
- **Monotonic Risk Physics:** Features were mathematically curated (`mean_difficulty`, `max_difficulty`, `high_risk_count`, `situation_count`) to guarantee monotonic risk ordering—adding high-risk situations strictly increases risk probability.
- **Observability & Metrics:** Instrumented with Prometheus (`/metrics`) exposing latency quantiles and prediction counters, a live dashboard (`/dashboard`), and an interactive UI (`/playground`).
- **Container Hardening:** Multi-stage Docker image executed under an unprivileged user (`appuser`, UID 1001).

---

## Architecture

Dependencies strictly point inward: business entities and use cases know nothing about FastAPI or XGBoost.

```
┌────────────────────────────────────────────────────────┐
│                      API Layer                         │
│   FastAPI routes · Pydantic v2 schemas · middleware    │
│   Playground · Live Dashboard · Prometheus Metrics     │
└──────────────────────────┬─────────────────────────────┘
                           │ (HTTP / DTO)
┌──────────────────────────▼─────────────────────────────┐
│                  Application Layer                     │
│        ClassifyExerciseDifficultyUseCase               │
└──────────────────────────┬─────────────────────────────┘
                           │ (Calls Port Interface)
┌──────────────────────────▼─────────────────────────────┐
│                    Domain Layer                        │
│   Entities · AggregatedFeatures · ClassifierPort (ABC) │
└──────────────────────────▲─────────────────────────────┘
                           │ (Implements Port)
┌──────────────────────────┴─────────────────────────────┐
│                 Infrastructure Layer                   │
│      XGBoostAdapter · Secure JSON Model Loader         │
└────────────────────────────────────────────────────────┘
```

### Directory Structure

```
maritime-freight-predict-api/
├── data/
│   └── labeled/                # Labeled exercise CSV datasets (100 exercises, 1,150 situations)
├── ml_pipeline/
│   ├── train_model.py          # Leakage-free training script (Stratified K-Fold, XGBoost)
│   └── requirements_ml.txt     # Training dependencies
├── models/                     # Production bridge assets
│   ├── model.json              # Native XGBoost booster weights (zero pickle)
│   ├── scaler_params.json      # Frozen Z-score scaling parameters
│   └── metadata.json           # Model card and feature schema metadata
├── src/predict/
│   ├── domain/                 # Invariant entities, value objects and ports
│   ├── application/            # Use cases and real-time feature aggregation
│   ├── infrastructure/         # XGBoost adapter and secure JSON loader
│   └── api/                    # FastAPI routes, schemas, metrics, views, UI
└── tests/
    ├── unit/                   # Schema validation, value objects, and use case unit tests
    └── integration/            # Model loader, XGBoost adapter, and API endpoint integration tests
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/a-delavega/maritime-freight-predict-api.git
cd maritime-freight-predict-api

python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -e ".[dev]"
```

### 2. Run the Service

```bash
uvicorn predict.api.main:app --host 127.0.0.1 --port 8000 --reload
```

| Endpoint | URL | Description |
|---|---|---|
| **Interactive Playground** | [/playground](http://127.0.0.1:8000/playground) | Graphical interface to test scenarios with live probability bars |
| **API Documentation** | [/docs](http://127.0.0.1:8000/docs) | OpenAPI 3.1 Swagger UI (custom dark theme) |
| **Alternative Docs** | [/redoc](http://127.0.0.1:8000/redoc) | Clean ReDoc specification reference |
| **Analytics Dashboard** | [/dashboard](http://127.0.0.1:8000/dashboard) | Live Chart.js visualization of served inference requests |
| **Prometheus Metrics** | [/metrics](http://127.0.0.1:8000/metrics) | Scrapeable Prometheus metrics and latency histograms |
| **Health & Readiness** | [/health](http://127.0.0.1:8000/health) | Liveness probe and model readiness status |

---

## API Usage

### Single Prediction

`POST /predict`

Request:
```json
{
  "situations": ["HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "HIGH"]
}
```

Response (`200 OK`):
```json
{
  "exercise_difficulty": "HIGH",
  "probabilities": {
    "LOW": 0.0600,
    "MEDIUM": 0.3252,
    "HIGH": 0.6148
  },
  "situation_count": 6,
  "model_version": "1.0.0"
}
```

### Batch Prediction

`POST /predict/batch`

Request:
```json
{
  "exercises": {
    "exercise_01": ["LOW", "LOW", "MEDIUM"],
    "exercise_02": ["HIGH", "HIGH", "HIGH", "MEDIUM"]
  }
}
```

Response (`200 OK`):
```json
{
  "predictions": {
    "exercise_01": {
      "exercise_difficulty": "LOW",
      "probabilities": { "LOW": 0.7884, "MEDIUM": 0.1829, "HIGH": 0.0286 },
      "situation_count": 3,
      "model_version": "1.0.0"
    },
    "exercise_02": {
      "exercise_difficulty": "MEDIUM",
      "probabilities": { "LOW": 0.0635, "MEDIUM": 0.5298, "HIGH": 0.4068 },
      "situation_count": 4,
      "model_version": "1.0.0"
    }
  },
  "total_exercises": 2,
  "model_version": "1.0.0"
}
```

---

## Test Suite

The project includes **52 automated tests** covering edge cases, Pydantic DTO boundaries, frozen dataclass immutability, corrupted asset handling, and monotonic inference checks:

```bash
# Run test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=predict --cov-report=term-missing
```

```
======================== 52 passed, 1 warning in 3.16s ========================
```

---

## Docker Deployment

The microservice builds into a lightweight, hardened container using a multi-stage `python:3.11-slim` image:

```bash
# Build the Docker image
docker build -t maritime-predict-api:latest .

# Run containerized microservice
docker run -d -p 8000:8000 --name maritime-api maritime-predict-api:latest

# Verify non-root execution
docker exec maritime-api whoami
# appuser
```

---

## Design Decisions

| Decision | Implementation | Why |
|---|---|---|
| **Security (Anti-RCE)** | Native `model.json` + `scaler_params.json` | Traditional `pickle`/`joblib` allows Remote Code Execution (CWE-502) via malicious payloads. Native JSON serialization isolates execution safely. |
| **Statistical Monotonicity** | `high_risk_count` replaces `min_difficulty` | Training exercises with low encounter ratings created an inverse correlation with `min_difficulty`. Replacing it with the count of high-risk encounters restores natural physical risk monotonicity ($r = +0.716$). |
| **Overfitting Prevention** | Constrained `max_depth=3`, No SMOTE | Eliminates synthetic hallucinations from SMOTE and limits tree depth for sub-millisecond inference and high generalization. |
| **Concurrency** | Synchronous worker thread delegation (`def`) | CPU-bound tree evaluation is executed inside AnyIO worker threads, leaving the async event loop non-blocking. |
| **Boundary Contracts** | Pydantic v2 Enums (`LOW`, `MEDIUM`, `HIGH`) | Input arrays must contain between 1 and 100 valid situations; malformed requests are halted at the network perimeter. |
| **Clean Decoupling** | Ports and Adapters (Hexagonal) | Changing the underlying ML engine (e.g., to ONNX or LightGBM) requires zero changes to use cases or HTTP routes. |

---

## License

MIT License. Designed and developed by Alejandro de la Vega ([@a-delavega](https://github.com/a-delavega)).
