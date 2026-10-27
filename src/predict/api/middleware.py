"""
Structured logging, observability, security headers, rate limiting, and payload protection middleware.

Adds unique request IDs, limits payload sizes (anti-DoS), tracks client IP request rates,
and logs duration, status, and path for every request.
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
import threading
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from predict.api.metrics import HTTP_REQUEST_DURATION_SECONDS

logger = logging.getLogger("predict.observability")

# Maximum permitted payload size: 1 MB (prevents memory exhaustion DoS)
MAX_CONTENT_LENGTH = 1_048_576

# In-memory sliding window rate limiter: 60 requests per minute per IP
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60


class InMemoryRateLimiter:
    """Thread-safe sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """Check if request is permitted under rate limit.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Purge expired timestamps for this IP
            timestamps = [t for t in self._requests[client_ip] if t > cutoff]
            if len(timestamps) >= self.max_requests:
                self._requests[client_ip] = timestamps
                return False, 0

            timestamps.append(now)
            self._requests[client_ip] = timestamps
            remaining = self.max_requests - len(timestamps)
            return True, remaining


_rate_limiter = InMemoryRateLimiter()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to inject correlation IDs, enforce anti-DoS limits, and log structured telemetry."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Enforce payload size protection (Anti-DoS Content-Length check)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_CONTENT_LENGTH:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large. Maximum allowed size is 1MB."},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )

        # 2. Rate Limiting check (per client IP)
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = _rate_limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Maximum 60 requests per minute."},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

        # 3. Enforce sanitization of X-Request-ID to prevent header injection or unbounded lengths
        raw_request_id = request.headers.get("X-Request-ID")
        if raw_request_id and len(raw_request_id) <= 64 and raw_request_id.replace("-", "").isalnum():
            request_id = raw_request_id
        else:
            request_id = str(uuid.uuid4())

        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        duration_ms = round(duration * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        # 4. Defense-in-depth: Standard OWASP security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

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
