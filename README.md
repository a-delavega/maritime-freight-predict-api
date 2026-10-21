# Maritime Navigation Predict API

Microservice for classifying the difficulty of maritime navigation exercises (FórmaTECH simulation).

## Architecture

```
Clean Architecture (Dependency Rule: dependencies point inwards to domain)

┌─────────────────────────────────────────────┐
│  API Layer (FastAPI)                        │
│  routes.py · schemas.py · dependencies.py   │
├─────────────────────────────────────────────┤
│  Application Layer                          │
│  ClassifyExerciseDifficultyUseCase          │
├─────────────────────────────────────────────┤
│  Domain Layer (pure, zero framework deps)   │
│  entities.py · ports.py                     │
├─────────────────────────────────────────────┤
│  Infrastructure Layer                       │
│  XGBoostAdapter · model_loader.py           │
└─────────────────────────────────────────────┘
```

## Technical Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| ML Model | XGBoost (`XGBClassifier`) + scikit-learn `Pipeline` |
| Validation | Pydantic v2 |
| Testing | Pytest (TDD) |
| Container | Docker (multi-stage, non-root) |
| Serialization | Joblib |

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -e ".[dev]"
```

### 2. Serialize model (only required once)

```bash
python scripts/serialize_model.py
```

Trains the `XGBClassifier` and saves the pipeline to `models/model_pipeline.joblib`.

### 3. Run API

```bash
uvicorn predict.api.main:app --reload
```

Interactive docs: `http://localhost:8000/docs`  
Health check: `http://localhost:8000/health`

### 4. Run prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"situations": ["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]}'
```

Example response:

```json
{
  "exercise_difficulty": "MEDIUM",
  "probabilities": {"HIGH": 0.23, "LOW": 0.12, "MEDIUM": 0.65},
  "situation_count": 6,
  "model_version": "1.0.0"
}
```

## Testing

```bash
# Run entire test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=predict --cov-report=term-missing
```

## Docker

```bash
# Build
docker build -t maritime-predict-api .

# Run
docker run -p 8000:8000 maritime-predict-api
```
