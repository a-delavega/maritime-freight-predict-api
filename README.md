<div align="center">

# 🚢 Maritime Freight & Navigation Predict API

[![CI/CD Pipeline](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml/badge.svg)](https://github.com/a-delavega/maritime-freight-predict-api/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-success.svg)](https://github.com/a-delavega/maritime-freight-predict-api)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C.svg?logo=prometheus&logoColor=white)](http://127.0.0.1:8000/metrics)
[![Architecture](https://img.shields.io/badge/architecture-Clean%20%2F%20Hexagonal-orange.svg)](#-clean-architecture)

**A high-throughput, production-grade ML inference microservice designed following Big Tech engineering standards (Clean Architecture, Strict TDD, Multi-stage Hardened Docker, Semantic Ordinal ML Pipeline, and Full Observability).**

[Explore API Docs](#-api-documentation--endpoints) • [Interactive Studio](#-interactive-prediction-studio) • [Quickstart](#-quickstart) • [Architecture](#-clean-architecture) • [Engineering Decisions](#-key-engineering-decisions--trade-offs)

---

</div>

## 📌 Executive Summary

Modern machine learning platforms demand rigorous software engineering beyond exploratory scripts. **Maritime Predict API** serves an optimized **XGBoost classification pipeline** (`StandardScaler + XGBClassifier`) to evaluate maritime risk encounters and classify overall exercise complexity (`LOW`, `MEDIUM`, `HIGH`).

Built from the ground up prioritizing **decoupling, determinism, security, and low-latency inference**:
- **Clean Architecture & DIP**: Zero framework coupling in the domain layer; model drivers are interchangeable via adapter ports.
- **Semantic Ordinal ML Scaling**: Enforces physical risk monotonicity (`LOW: 1.0 < MEDIUM: 2.0 < HIGH: 3.0`) eliminating alphabetical encoding artifacts.
- **Fail-Safe Concurrency**: CPU-bound inference run off-loop via synchronous threadpool offloading, keeping the async event loop non-blocking.
- **Defensive API Contracts**: Pydantic v2 schemas validating boundaries and rejecting malformed inputs before reaching memory.
- **Full Observability & Interactive Studio**: Built-in Prometheus `/metrics` exposition, live Chart.js `/dashboard`, and an interactive visual `/playground` with dynamic dropdowns.
- **Production Hardened**: Containerized under an unprivileged `appuser` (non-root), health check probes, and structured JSON telemetry.

---

## 🏛 Clean Architecture

The codebase strictly adheres to the **Dependency Inversion Principle (DIP)**. Dependencies point inward—domain logic is fully isolated from external frameworks like FastAPI or XGBoost.

```
┌────────────────────────────────────────────────────────┐
│                      API Layer                         │
│   FastAPI Routes · Pydantic v2 Schemas · Middleware    │
│   Studio Playground · Live Dashboard · Prometheus      │
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
│         XGBoostAdapter · Joblib Model Loader           │
└────────────────────────────────────────────────────────┘
```

```
maritime-freight-predict-api/
├── .github/workflows/          # CI/CD Automated Test Pipeline (GitHub Actions)
├── models/                     # Serialized Model Artifacts & Metadata (joblib / json)
├── scripts/                    # Model serialization & client demo scripts
├── src/predict/
│   ├── domain/                 # Pure domain entities, value objects, and ports
│   ├── application/            # Orchestration use cases
│   ├── infrastructure/         # XGBoost adapters and joblib loaders
│   └── api/                    # FastAPI routes, schemas, middleware, playground, metrics
└── tests/
    ├── unit/                   # Schema validation & Use case unit tests
    └── integration/            # Model loader & End-to-end API integration tests
```

---

## ⚡ Quickstart

### 1. Prerequisites & Virtual Environment

```bash
# Clone the repository
git clone https://github.com/a-delavega/maritime-freight-predict-api.git
cd maritime-freight-predict-api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install package with developer dependencies
pip install -e ".[dev]"
```

### 2. Launch Local Server

```bash
uvicorn predict.api.main:app --host 127.0.0.1 --port 8000 --reload
```

| Service | URL | Description |
|---|---|---|
| **🎨 Interactive Studio** | [http://127.0.0.1:8000/playground](http://127.0.0.1:8000/playground) | Visual prediction UI with dynamic dropdowns & live charts |
| **📑 OpenAPI Swagger Docs** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Customized dark developer portal documentation |
| **📖 ReDoc Documentation** | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | Alternative clean API specification portal |
| **📊 Visual Telemetry Dashboard** | [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) | Real-time Doughnut chart tracking inference distribution |
| **📈 Prometheus Metrics** | [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics) | Standard CNCF metrics exposition for scraping |
| **🩺 Health Probe** | [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | Liveness & model readiness check |

### 3. Run Client Demo

In a separate terminal, execute the demo client:

```bash
python scripts/client_demo.py
```

---

## 🎨 Interactive Prediction Studio

Rather than writing raw JSON requests, the service provides an **Interactive Studio (`/playground`)**:

- **Visual Dropdown Selectors**: Choose risk encounters (`🟢 LOW`, `🟡 MEDIUM`, `🔴 HIGH`) with real-time color coding.
- **Dynamic Situation Controls**: Add or remove situations on the fly (`+ Add Situation`, `✕`).
- **Instant Inference & Feedback**: Automatically calls the model upon selection change, displaying confidence distribution bar charts and exact server latency in milliseconds.
- **One-click Presets**: Quick-load scenarios like *Stress Test (High Risk)* or *Standard*.

---

## 📡 API Documentation & Endpoints

### Single Prediction
`POST /predict`

Evaluates a single maritime navigation exercise given a sequence of situation risk levels.

#### Request Payload
```json
{
  "situations": ["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
}
```

#### Response (`200 OK`)
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

---

### Batch Prediction
`POST /predict/batch`

High-throughput endpoint to evaluate multiple exercises concurrently in a single network roundtrip.

#### Request Payload
```json
{
  "exercises": {
    "exercise_101": ["LOW", "LOW", "MEDIUM"],
    "exercise_102": ["HIGH", "HIGH", "HIGH", "MEDIUM"]
  }
}
```

#### Response (`200 OK`)
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

## 🧪 Test Suite (Strict TDD)

The service is backed by an automated test matrix with **45 tests** across unit, validation, domain edge-cases, and integration layers:

```bash
# Execute entire test suite
pytest tests/ -v

# Run with test coverage report
pytest tests/ -v --cov=predict --cov-report=term-missing
```

```
============================== test session starts ==============================
collected 45 items

tests/integration/test_api.py::TestHealthEndpoint ... PASSED
tests/integration/test_api.py::TestPredictEndpoint ... PASSED
tests/integration/test_api.py::TestPredictEndpointFailures ... PASSED
tests/integration/test_model_loader.py::TestModelLoader ... PASSED
tests/integration/test_model_loader.py::TestXGBoostAdapterIntegration ... PASSED
tests/unit/test_schemas.py::TestPredictionRequestValid ... PASSED
tests/unit/test_schemas.py::TestPredictionRequestInvalid ... PASSED
tests/unit/test_schemas.py::TestPredictionRequestEdgeCases ... PASSED
tests/unit/test_use_cases.py::TestUseCaseHappyPath ... PASSED
tests/unit/test_use_cases.py::TestUseCaseEdgeCases ... PASSED
tests/unit/test_use_cases.py::TestAggregatedFeatures ... PASSED

======================== 45 passed, 100% test success ==========================
```

---

## 🐳 Docker Deployment

The application uses a **multi-stage build** based on `python:3.11-slim`, discarding build-time toolchains and reducing attack surface.

```bash
# Build production image
docker build -t maritime-predict-api:latest .

# Run container (listening on port 8000)
docker run -d -p 8000:8000 --name maritime-api maritime-predict-api:latest

# Verify non-root user security context
docker exec maritime-api whoami
# Output: appuser
```

---

## 💡 Key Engineering Decisions & Trade-offs

| Decision | Selection | Rationale |
|---|---|---|
| **Concurrency Strategy** | Synchronous `def` handlers | XGBoost inference is CPU-bound. FastAPI automatically offloads standard `def` endpoints to AnyIO worker threads, preventing event-loop starvation. |
| **Model Lifecycle** | FastAPI `lifespan` context | The model is loaded once at startup and kept in memory (`fail-fast`). Avoids cold-start latency per request. |
| **Semantic Ordinal Scaling** | `LOW: 1 < MEDIUM: 2 < HIGH: 3` | Fixes standard `LabelEncoder` alphabetical flaws (which assign `HIGH=0`), ensuring feature aggregation (`mean`, `max`, `min`) mirrors genuine risk escalation. |
| **Domain Decoupling** | Domain Enums & Ports | Business rules and use cases remain purely Pythonic without framework dependencies (FastAPI or XGBoost). |
| **Security Hardening** | Unprivileged Linux User | The Docker image runs under an isolated non-root `appuser` (UID 1001) preventing container breakout escalation. |
| **Observability** | Prometheus + JSON Middleware | Each request receives a unique `X-Request-ID`, emits duration in ms, and updates Prometheus counters and latency histograms. |

---

## 📄 License
Distributed under the MIT License. Developed by **Alejandro de la Vega** ([@a-delavega](https://github.com/a-delavega)).
