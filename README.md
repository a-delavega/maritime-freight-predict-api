<div align="center">

# Maritime Freight & Navigation Predict API

[![CI/CD Pipeline](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml/badge.svg)](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-success.svg)](https://github.com/a-delavega/maritime-freight-predict-api)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C.svg?logo=prometheus&logoColor=white)](http://127.0.0.1:8000/metrics)
[![Architecture](https://img.shields.io/badge/architecture-Clean%20%2F%20Hexagonal-orange.svg)](#architecture)

An ML inference service that classifies the difficulty of a maritime navigation exercise (`LOW`, `MEDIUM` or `HIGH`) from the risk levels of the situations it contains. Built with FastAPI, XGBoost and a Clean Architecture layout.

[API](#api) • [Playground](#playground) • [Quickstart](#quickstart) • [Architecture](#architecture) • [Design decisions](#design-decisions)

---

</div>

## Overview

This project wraps an XGBoost classification pipeline (`StandardScaler + XGBClassifier`) in a small web service. You send it the list of situation risk levels for an exercise, and it returns the predicted difficulty along with the probability of each class.

I wanted it to be more than a notebook behind an endpoint, so most of the effort went into the engineering around the model:

- **Clean Architecture.** The domain layer has no dependency on FastAPI or XGBoost. The model is reached through a port, so the XGBoost adapter can be swapped without touching the use case.
- **Ordinal encoding.** Risk levels are mapped as `LOW = 1.0 < MEDIUM = 2.0 < HIGH = 3.0`. A plain `LabelEncoder` sorts alphabetically and gives `HIGH = 0`, which breaks aggregates like mean, max and min as a measure of risk.
- **Non-blocking inference.** Prediction is CPU-bound, so the endpoints are plain `def` handlers. FastAPI runs them in a worker thread and the event loop stays free.
- **Input validation.** Pydantic v2 schemas reject malformed requests at the API boundary, before they get anywhere near the model.
- **Observability.** Prometheus metrics at `/metrics`, a live Chart.js dashboard at `/dashboard`, and JSON request logs.
- **Container.** Multi-stage Docker build, runs as a non-root user, with a health check.

There is also a small interactive UI at `/playground` for trying the model without writing JSON by hand.

---

## Architecture

Dependencies point inward: the domain knows nothing about the frameworks around it.

```
┌────────────────────────────────────────────────────────┐
│                      API Layer                         │
│   FastAPI routes · Pydantic v2 schemas · middleware    │
│   Playground · Dashboard · Prometheus                  │
└──────────────────────────┬─────────────────────────────┘
                           │ (HTTP / DTO)
┌──────────────────────────▼─────────────────────────────┐
│                  Application Layer                     │
│        ClassifyExerciseDifficultyUseCase               │
└──────────────────────────┬─────────────────────────────┘
                           │ (calls port interface)
┌──────────────────────────▼─────────────────────────────┐
│                    Domain Layer                        │
│   Entities · AggregatedFeatures · ClassifierPort (ABC) │
└──────────────────────────▲─────────────────────────────┘
                           │ (implements port)
┌──────────────────────────┴─────────────────────────────┐
│                 Infrastructure Layer                   │
│         XGBoostAdapter · joblib model loader           │
└────────────────────────────────────────────────────────┘
```

Project layout:

```
maritime-freight-predict-api/
├── .github/workflows/          # CI pipeline (GitHub Actions)
├── models/                     # Serialized model and metadata (joblib / json)
├── scripts/                    # Model serialization and client demo scripts
├── src/predict/
│   ├── domain/                 # Entities, value objects and ports
│   ├── application/            # Use cases
│   ├── infrastructure/         # XGBoost adapter and joblib loader
│   └── api/                    # Routes, schemas, middleware, playground, metrics
└── tests/
    ├── unit/                   # Schema validation and use case tests
    └── integration/            # Model loader and end-to-end API tests
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/a-delavega/maritime-freight-predict-api.git
cd maritime-freight-predict-api

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### 2. Run the server

```bash
uvicorn predict.api.main:app --host 127.0.0.1 --port 8000 --reload
```

| Service | URL | What it is |
|---|---|---|
| Playground | [/playground](http://127.0.0.1:8000/playground) | Visual UI to try predictions, with live charts |
| Swagger docs | [/docs](http://127.0.0.1:8000/docs) | OpenAPI docs (dark theme) |
| ReDoc | [/redoc](http://127.0.0.1:8000/redoc) | Alternative API reference |
| Dashboard | [/dashboard](http://127.0.0.1:8000/dashboard) | Doughnut chart of the predictions served so far |
| Metrics | [/metrics](http://127.0.0.1:8000/metrics) | Prometheus endpoint for scraping |
| Health | [/health](http://127.0.0.1:8000/health) | Liveness and model readiness check |

### 3. Try the demo client

From another terminal:

```bash
python scripts/client_demo.py
```

---

## Playground

`/playground` lets you build an exercise from the browser instead of sending raw JSON:

- Pick each situation's risk level from a dropdown (LOW, MEDIUM, HIGH, color coded).
- Add or remove situations with `+ Add Situation` and `✕`.
- The model is called every time you change something. You get the class probabilities as a bar chart and the server latency in milliseconds.
- Presets for common cases, such as *Standard* or *Stress Test (High Risk)*.

---

## API

### Single prediction

`POST /predict`

Classifies one exercise from the list of its situation risk levels.

Request:

```json
{
  "situations": ["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
}
```

Response (`200 OK`):

```json
{
  "exercise_difficulty": "MEDIUM",
  "probabilities": {
    "LOW": 0.0224,
    "MEDIUM": 0.9745,
    "HIGH": 0.0031
  },
  "situation_count": 6,
  "model_version": "1.0.0"
}
```

### Batch prediction

`POST /predict/batch`

Classifies several exercises in a single request. Each exercise is keyed by an ID of your choice.

Request:

```json
{
  "exercises": {
    "exercise_101": ["LOW", "LOW", "MEDIUM"],
    "exercise_102": ["HIGH", "HIGH", "HIGH", "MEDIUM"]
  }
}
```

Response (`200 OK`):

```json
{
  "predictions": {
    "exercise_101": {
      "exercise_difficulty": "LOW",
      "probabilities": { "LOW": 0.9886, "MEDIUM": 0.0111, "HIGH": 0.0003 },
      "situation_count": 3,
      "model_version": "1.0.0"
    },
    "exercise_102": {
      "exercise_difficulty": "MEDIUM",
      "probabilities": { "LOW": 0.3927, "MEDIUM": 0.5554, "HIGH": 0.0519 },
      "situation_count": 4,
      "model_version": "1.0.0"
    }
  },
  "total_exercises": 2,
  "model_version": "1.0.0"
}
```

---

## Tests

There are 45 tests covering schema validation, use case logic and edge cases in the domain, the model loader, and the API end to end. The project was developed test-first.

```bash
# Run everything
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=predict --cov-report=term-missing
```

Expected result:

```
collected 45 items
...
============================== 45 passed ==============================
```

---

## Docker

The image is a multi-stage build on `python:3.11-slim`. Build tools stay in the builder stage, so the final image is smaller and has less in it.

```bash
# Build the image
docker build -t maritime-predict-api:latest .

# Run it (port 8000)
docker run -d -p 8000:8000 --name maritime-api maritime-predict-api:latest

# Check that it is not running as root
docker exec maritime-api whoami
# appuser
```

---

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Concurrency | Plain `def` handlers | XGBoost inference is CPU-bound. FastAPI runs sync endpoints in AnyIO worker threads, so the event loop is never starved. |
| Model lifecycle | Loaded once in the FastAPI `lifespan` | The model sits in memory after startup, so there is no load cost per request. If it can't be loaded, the app fails at startup instead of on the first call. |
| Ordinal encoding | `LOW: 1 < MEDIUM: 2 < HIGH: 3` | `LabelEncoder` orders classes alphabetically (`HIGH = 0`). With an explicit mapping, mean/max/min over the situations actually follow risk. |
| Decoupling | Domain enums and ports | Use cases and business rules don't import FastAPI or XGBoost, so they can be tested and reused on their own. |
| Container security | Non-root `appuser` (UID 1001) | The process doesn't run with root privileges inside the container, which limits the damage if something goes wrong. |
| Observability | Prometheus + JSON logging middleware | Every request gets a unique `X-Request-ID`, its duration is logged in ms, and Prometheus counters and latency histograms are updated. |

---

## License

MIT. Written by Alejandro de la Vega ([@a-delavega](https://github.com/a-delavega)).
