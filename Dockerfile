# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# Security: create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set PYTHONPATH so the predict package is importable
ENV PYTHONPATH=/app/src
ENV MODELS_DIR=/app/models

# Security: switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn (disable server header to avoid server fingerprinting)
CMD ["uvicorn", "predict.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
