"""
Interactive Playground UI template.

Provides a visual dashboard with dropdowns to configure situations
and obtain immediate exercise difficulty predictions from the model.
"""

PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Maritime AI - Interactive Prediction Studio</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card: #131b2e;
      --card-inner: #1a243b;
      --border: #24324f;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --low: #10b981;
      --medium: #f59e0b;
      --high: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 32px 20px;
    }
    .container {
      max-width: 1040px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .badge {
      background: rgba(56, 189, 248, 0.12);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.25);
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
    }
    .card-title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .situations-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 6px;
    }
    .situation-row {
      display: flex;
      align-items: center;
      gap: 12px;
      background: var(--card-inner);
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      transition: border-color 0.2s;
    }
    .situation-row:hover { border-color: #3b82f6; }
    .situation-index {
      font-weight: 700;
      font-size: 13px;
      color: var(--text-muted);
      width: 28px;
    }
    .situation-label {
      flex: 1;
      font-size: 14px;
      font-weight: 500;
    }
    select.difficulty-select {
      background: #0b1120;
      color: #fff;
      border: 1px solid #334155;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      outline: none;
    }
    select.difficulty-select.val-LOW { color: var(--low); border-color: rgba(16, 185, 129, 0.4); }
    select.difficulty-select.val-MEDIUM { color: var(--medium); border-color: rgba(245, 158, 11, 0.4); }
    select.difficulty-select.val-HIGH { color: var(--high); border-color: rgba(239, 68, 68, 0.4); }
    .btn-delete {
      background: transparent;
      border: none;
      color: #64748b;
      cursor: pointer;
      padding: 6px;
      border-radius: 6px;
      font-size: 16px;
      line-height: 1;
    }
    .btn-delete:hover { color: var(--high); background: rgba(239, 68, 68, 0.1); }
    .actions-bar {
      display: flex;
      gap: 12px;
      margin-top: 18px;
    }
    .btn {
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-primary {
      background: var(--primary);
      color: #030712;
      flex: 1;
      justify-content: center;
      box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-secondary {
      background: var(--card-inner);
      color: var(--text);
      border-color: var(--border);
    }
    .btn-secondary:hover { background: #22304d; }
    .result-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
      background: var(--card-inner);
      border-radius: 12px;
      border: 1px solid var(--border);
      text-align: center;
      margin-bottom: 20px;
    }
    .result-badge {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-top: 8px;
    }
    .result-badge.LOW { color: var(--low); text-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
    .result-badge.MEDIUM { color: var(--medium); text-shadow: 0 0 20px rgba(245, 158, 11, 0.3); }
    .result-badge.HIGH { color: var(--high); text-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    .chart-box {
      height: 220px;
      position: relative;
    }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      margin-left: 14px;
    }
    .nav-links a:hover { color: var(--primary); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🚢 Maritime Prediction Studio <span class="badge">XGBoost v1.0</span></h1>
        <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">
          Configure risk encounters with visual dropdowns and calculate exercise difficulty in real time
        </p>
      </div>
      <div class="nav-links">
        <a href="/docs">Swagger API Docs</a>
        <a href="/dashboard">Telemetry Dashboard</a>
      </div>
    </header>

    <div class="layout">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-title">
          <span>Exercise Situations (<span id="situationCount">4</span>)</span>
          <button class="btn btn-secondary" onclick="addSituation()" style="padding: 6px 12px; font-size: 12px;">+ Add Situation</button>
        </div>

        <div class="situations-list" id="situationsContainer">
          <!-- Rows will be rendered dynamically -->
        </div>

        <div class="actions-bar">
          <button class="btn btn-primary" onclick="submitPrediction()">
            ⚡ Classify Exercise Difficulty
          </button>
          <button class="btn btn-secondary" onclick="loadPreset('stress')">Stress Test (High Risk)</button>
          <button class="btn btn-secondary" onclick="loadPreset('standard')">Standard</button>
        </div>
      </div>

      <!-- Output Panel -->
      <div class="card">
        <div class="card-title">
          <span>Predicted Outcome</span>
          <span id="latencySpan" style="font-size: 12px; color: var(--text-muted);">Latency: -</span>
        </div>

        <div class="result-box">
          <span style="font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Overall Difficulty</span>
          <div class="result-badge MEDIUM" id="predictedClass">MEDIUM</div>
        </div>

        <div class="card-title" style="font-size: 14px; margin-bottom: 8px;">Confidence Distribution</div>
        <div class="chart-box">
          <canvas id="probaChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <script>
    let situations = ["LOW", "MEDIUM", "HIGH", "MEDIUM"];
    let chartInstance = null;

    function renderSituations() {
      const container = document.getElementById("situationsContainer");
      container.innerHTML = "";
      document.getElementById("situationCount").textContent = situations.length;

      situations.forEach((sit, index) => {
        const row = document.createElement("div");
        row.className = "situation-row";
        row.innerHTML = `
          <div class="situation-index">#${index + 1}</div>
          <div class="situation-label">Navigation Encounter Risk</div>
          <select class="difficulty-select val-${sit}" onchange="updateSituation(${index}, this.value)">
            <option value="LOW" ${sit === "LOW" ? "selected" : ""}>🟢 LOW</option>
            <option value="MEDIUM" ${sit === "MEDIUM" ? "selected" : ""}>🟡 MEDIUM</option>
            <option value="HIGH" ${sit === "HIGH" ? "selected" : ""}>🔴 HIGH</option>
          </select>
          <button class="btn-delete" title="Remove" onclick="removeSituation(${index})" ${situations.length <= 1 ? "disabled style='opacity:0.2'" : ""}>✕</button>
        `;
        container.appendChild(row);
      });
    }

    function updateSituation(index, value) {
      situations[index] = value;
      renderSituations();
      submitPrediction();
    }

    function addSituation() {
      if (situations.length >= 50) return alert("Maximum 50 situations per exercise.");
      situations.push("MEDIUM");
      renderSituations();
      submitPrediction();
    }

    function removeSituation(index) {
      if (situations.length <= 1) return;
      situations.splice(index, 1);
      renderSituations();
      submitPrediction();
    }

    function loadPreset(type) {
      if (type === 'stress') {
        situations = ["HIGH", "HIGH", "HIGH", "MEDIUM", "HIGH"];
      } else {
        situations = ["LOW", "LOW", "MEDIUM", "LOW"];
      }
      renderSituations();
      submitPrediction();
    }

    async function submitPrediction() {
      const start = performance.now();
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ situations })
        });
        const elapsed = (performance.now() - start).toFixed(1);
        document.getElementById("latencySpan").textContent = `Latency: ${elapsed}ms`;

        if (!res.ok) throw new Error("Prediction request failed");
        const data = await res.json();

        // Update badge
        const badge = document.getElementById("predictedClass");
        badge.textContent = data.exercise_difficulty;
        badge.className = `result-badge ${data.exercise_difficulty}`;

        // Update probabilities chart
        updateChart(data.probabilities);
      } catch (err) {
        console.error(err);
      }
    }

    function updateChart(probMap) {
      const ctx = document.getElementById("probaChart").getContext("2d");
      const labels = ["LOW", "MEDIUM", "HIGH"];
      const values = [
        (probMap["LOW"] * 100).toFixed(1),
        (probMap["MEDIUM"] * 100).toFixed(1),
        (probMap["HIGH"] * 100).toFixed(1)
      ];

      if (chartInstance) chartInstance.destroy();

      chartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Probability (%)",
            data: values,
            backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` ${ctx.raw}% probability`
              }
            }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: "#94a3b8", font: { weight: "600" } }
            },
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: "rgba(51, 65, 85, 0.4)" },
              ticks: { color: "#64748b", callback: v => v + "%" }
            }
          }
        }
      });
    }

    // Initial load
    renderSituations();
    submitPrediction();
  </script>
</body>
</html>
"""
