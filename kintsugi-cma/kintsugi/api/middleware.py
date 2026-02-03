"""Auth, PII-redaction, and request-logging middleware for FastAPI."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kintsugi.config.settings import settings
from kintsugi.security.pii import PIIRedactor

logger = logging.getLogger("kintsugi.api")

# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID and log method/path/status/duration."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "method=%s path=%s status_code=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

_redactor = PIIRedactor()


class PIIRedactionMiddleware(BaseHTTPMiddleware):
    """Scan JSON response bodies for PII and redact before sending."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        response: Response = await call_next(request)

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            response.headers["X-PII-Redacted"] = "false"
            return response

        # Read the full body
        body_bytes = b""
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            if isinstance(chunk, str):
                body_bytes += chunk.encode("utf-8")
            else:
                body_bytes += chunk

        body_text = body_bytes.decode("utf-8")
        result = _redactor.redact(body_text)

        pii_found = bool(result.types_found)
        out_body = result.redacted_text.encode("utf-8") if pii_found else body_bytes

        return Response(
            content=out_body,
            status_code=response.status_code,
            headers={
                **dict(response.headers),
                "X-PII-Redacted": str(pii_found).lower(),
                "content-length": str(len(out_body)),
            },
            media_type=response.media_type,
        )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_EXEMPT_PREFIXES = ("/api/health", "/docs", "/openapi.json", "/ws/")


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token authentication via HMAC-SHA256 against SECRET_KEY."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        path = request.url.path

        # Exempt paths
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        # Development mode bypass
        if settings.SECRET_KEY == "CHANGE-ME-in-production":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"detail": "Missing or malformed Authorization header"}, status_code=401)

        token = auth_header[7:]
        if not token:
            return JSONResponse({"detail": "Empty bearer token"}, status_code=401)

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        key_hash = hashlib.sha256(settings.SECRET_KEY.encode()).hexdigest()

        if not hmac.compare_digest(token_hash, key_hash):
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        return await call_next(request)
