"""
Structured logging and observability middleware.

Adds unique request IDs and logs duration, status, and path for every request.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from predict.api.metrics import HTTP_REQUEST_DURATION_SECONDS

logger = logging.getLogger("predict.observability")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to inject correlation IDs and log structured request telemetry."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        duration_ms = round(duration * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        # Record Prometheus metric
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code),
        ).observe(duration)

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        logger.info(json.dumps(log_data))

        return response
