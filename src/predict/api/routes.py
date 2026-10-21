"""
API routes for the maritime navigation predict service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response

from predict.api.batch_schemas import BatchPredictionRequest, BatchPredictionResponse
from predict.api.dependencies import get_use_case
from predict.api.metrics import (
    INFERENCE_DURATION_SECONDS,
    PREDICTIONS_TOTAL,
    get_metrics_snapshot,
)
from predict.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    SituationDifficultySchema,
)
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.domain.entities import SituationDifficulty
from predict.infrastructure.xgboost_adapter import PredictionError


MODEL_VERSION = "1.0.0"

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    use_case: ClassifyExerciseDifficultyUseCase | None = Depends(get_use_case),
) -> HealthResponse:
    """Health check endpoint. Returns model loading status."""
    return HealthResponse(
        status="healthy" if use_case is not None else "degraded",
        model_loaded=use_case is not None,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    use_case: ClassifyExerciseDifficultyUseCase | None = Depends(get_use_case),
) -> PredictionResponse:
    """Classify the difficulty of a navigation exercise.

    Receives the difficulty of each situation in the exercise,
    aggregates them, and predicts the overall exercise difficulty.
    """
    if use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is unavailable.",
        )

    # Map schema enum to domain enum
    domain_situations = [
        SituationDifficulty(s.value) for s in request.situations
    ]

    try:
        import time
        start_inf = time.perf_counter()
        prediction = use_case.execute(domain_situations)
        inf_duration = time.perf_counter() - start_inf
        INFERENCE_DURATION_SECONDS.observe(inf_duration)
        PREDICTIONS_TOTAL.labels(
            predicted_class=prediction.difficulty.value,
            endpoint="predict",
        ).inc()
    except PredictionError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return PredictionResponse(
        exercise_difficulty=prediction.difficulty.value,
        probabilities=prediction.probabilities,
        situation_count=prediction.situation_count,
        model_version=MODEL_VERSION,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    use_case: ClassifyExerciseDifficultyUseCase | None = Depends(get_use_case),
) -> BatchPredictionResponse:
    """Classify multiple exercises in a single batch request."""
    if use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is unavailable.",
        )

    results: dict[str, PredictionResponse] = {}

    for exercise_id, situation_strs in request.exercises.items():
        if not situation_strs:
            raise HTTPException(
                status_code=422,
                detail=f"Exercise '{exercise_id}' has empty situations list",
            )
        try:
            domain_situations = [
                SituationDifficulty(SituationDifficultySchema(s).value)
                for s in situation_strs
            ]
            pred = use_case.execute(domain_situations)
            PREDICTIONS_TOTAL.labels(
                predicted_class=pred.difficulty.value,
                endpoint="predict_batch",
            ).inc()
            results[exercise_id] = PredictionResponse(
                exercise_difficulty=pred.difficulty.value,
                probabilities=pred.probabilities,
                situation_count=pred.situation_count,
                model_version=MODEL_VERSION,
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid value in exercise '{exercise_id}': {e}",
            )

    return BatchPredictionResponse(
        predictions=results,
        total_exercises=len(results),
        model_version=MODEL_VERSION,
    )


@router.get("/metrics", response_class=Response)
def metrics() -> Response:
    """Prometheus exposition endpoint for scraping model and API telemetry."""
    data = get_metrics_snapshot()
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Visual telemetry dashboard displaying real-time metrics and charts."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Maritime Predict API - Live Telemetry</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .stat-title { font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
    .stat-val { font-size: 28px; font-weight: 700; color: var(--accent); }
    .chart-container {
      position: relative;
      height: 240px;
      width: 100%;
    }
    .btn {
      background: var(--accent);
      color: #000;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 style="font-size: 22px; margin-bottom: 4px;">🚢 Maritime Inference Telemetry</h1>
      <p style="color: var(--text-muted); font-size: 13px;">Real-time Prometheus metrics & inference distribution</p>
    </div>
    <div style="display:flex; gap: 12px; align-items:center;">
      <div class="badge"><span class="badge-dot"></span> Pipeline Active</div>
      <button class="btn" onclick="fetchMetrics()">Refresh</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="stat-title">Total Inferences</div>
      <div class="stat-val" id="totalPredictions">-</div>
    </div>
    <div class="card">
      <div class="stat-title">Model State</div>
      <div class="stat-val" style="color: var(--success);" id="modelStatus">Loaded</div>
    </div>
    <div class="card">
      <div class="stat-title">Architecture</div>
      <div class="stat-val" style="font-size: 20px; color: var(--text);">Clean Architecture</div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr 1fr;">
    <div class="card">
      <div class="stat-title">Class Distribution (LOW / MEDIUM / HIGH)</div>
      <div class="chart-container">
        <canvas id="classChart"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="stat-title">Metrics Preview</div>
      <pre id="rawMetrics" style="font-size: 12px; color: var(--text-muted); overflow-y: auto; max-height: 240px; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">Loading Prometheus buffer...</pre>
    </div>
  </div>

  <script>
    let chart;
    async function fetchMetrics() {
      try {
        const res = await fetch('/metrics');
        const text = await res.text();
        document.getElementById('rawMetrics').textContent = text;

        let high = 0, low = 0, med = 0;
        const lines = text.split('\\n');
        for (const line of lines) {
          if (line.startsWith('model_predictions_total{')) {
            const val = parseFloat(line.split(' ').pop());
            if (line.includes('predicted_class="HIGH"')) high += val;
            if (line.includes('predicted_class="LOW"')) low += val;
            if (line.includes('predicted_class="MEDIUM"')) med += val;
          }
        }

        const total = high + low + med;
        document.getElementById('totalPredictions').textContent = total;

        const ctx = document.getElementById('classChart').getContext('2d');
        if (chart) chart.destroy();
        chart = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['LOW', 'MEDIUM', 'HIGH'],
            datasets: [{
              data: [low, med, high],
              backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'bottom', labels: { color: '#94a3b8' } }
            }
          }
        });
      } catch (e) {
        console.error(e);
      }
    }
    fetchMetrics();
    setInterval(fetchMetrics, 5000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
