"""Tests for kintsugi.api.websocket and kintsugi.api.middleware modules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from kintsugi.api.websocket import ConnectionManager, MessageType


# ---------------------------------------------------------------------------
# MessageType
# ---------------------------------------------------------------------------

class TestMessageType:
    def test_values(self):
        assert MessageType.AGENT_RESPONSE == "agent_response"
        assert MessageType.SHADOW_STATUS == "shadow_status"
        assert MessageType.TEMPORAL_EVENT == "temporal_event"
        assert MessageType.CONSENSUS_UPDATE == "consensus_update"
        assert MessageType.ERROR == "error"
        assert MessageType.HEARTBEAT == "heartbeat"


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connect_increments(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        await mgr.connect(ws, "org1")
        assert mgr.get_connection_count("org1") == 1

    @pytest.mark.asyncio
    async def test_disconnect_decrements(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        await mgr.connect(ws, "org1")
        await mgr.disconnect(ws, "org1")
        assert mgr.get_connection_count("org1") == 0

    @pytest.mark.asyncio
    async def test_disconnect_removes_org_key(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        await mgr.connect(ws, "org1")
        await mgr.disconnect(ws, "org1")
        assert mgr.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_send_to_org(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        await mgr.connect(ws, "org1")
        await mgr.send_to_org("org1", {"type": "test"})
        ws.send_json.assert_awaited_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_send_to_org_disconnects_on_error(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("closed")
        await mgr.connect(ws, "org1")
        await mgr.send_to_org("org1", {"type": "test"})
        assert mgr.get_connection_count("org1") == 0

    @pytest.mark.asyncio
    async def test_send_personal(self):
        mgr = ConnectionManager()
        ws = AsyncMock()
        await mgr.send_personal(ws, {"type": "hi"})
        ws.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_connection_count_total(self):
        mgr = ConnectionManager()
        for org in ("a", "b"):
            ws = AsyncMock()
            await mgr.connect(ws, org)
        assert mgr.get_connection_count() == 2

    @pytest.mark.asyncio
    async def test_get_connection_count_empty_org(self):
        mgr = ConnectionManager()
        assert mgr.get_connection_count("nope") == 0


# ---------------------------------------------------------------------------
# Middleware helpers
# ---------------------------------------------------------------------------

def _make_app():
    """Build a minimal FastAPI app with all three middleware layers."""
    from kintsugi.api.middleware import (
        AuthMiddleware,
        PIIRedactionMiddleware,
        RequestLoggingMiddleware,
    )

    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/data")
    async def data():
        return {"email": "user@example.com", "name": "Alice"}

    @app.get("/api/plain")
    async def plain():
        from starlette.responses import PlainTextResponse
        return PlainTextResponse("hello")

    # Order matters: outermost middleware runs first.
    # Auth -> Logging -> PII
    app.add_middleware(PIIRedactionMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuthMiddleware)

    return app


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------

class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_exempt_path_passes(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/health")
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_dev_mode_bypass(self):
        """Default SECRET_KEY is 'CHANGE-ME-in-production' -> bypass auth."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/data")
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_token(self):
        from kintsugi.config.settings import settings

        original = settings.SECRET_KEY
        try:
            settings.SECRET_KEY = "my-secret"
            app = _make_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/api/data", headers={"Authorization": "Bearer my-secret"})
                assert r.status_code == 200
        finally:
            settings.SECRET_KEY = original

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        from kintsugi.config.settings import settings

        original = settings.SECRET_KEY
        try:
            settings.SECRET_KEY = "my-secret"
            app = _make_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/api/data", headers={"Authorization": "Bearer wrong"})
                assert r.status_code == 401
        finally:
            settings.SECRET_KEY = original

    @pytest.mark.asyncio
    async def test_missing_auth_header(self):
        from kintsugi.config.settings import settings

        original = settings.SECRET_KEY
        try:
            settings.SECRET_KEY = "my-secret"
            app = _make_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/api/data")
                assert r.status_code == 401
        finally:
            settings.SECRET_KEY = original

    @pytest.mark.asyncio
    async def test_empty_bearer_token(self):
        from kintsugi.config.settings import settings

        original = settings.SECRET_KEY
        try:
            settings.SECRET_KEY = "my-secret"
            app = _make_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/api/data", headers={"Authorization": "Bearer "})
                assert r.status_code == 401
        finally:
            settings.SECRET_KEY = original


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------

class TestRequestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_adds_request_id_header(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/health")
            assert "x-request-id" in r.headers


# ---------------------------------------------------------------------------
# PIIRedactionMiddleware
# ---------------------------------------------------------------------------

class TestPIIRedactionMiddleware:
    @pytest.mark.asyncio
    async def test_redacts_email_from_json(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/data")
            assert "user@example.com" not in r.text
            assert r.headers.get("x-pii-redacted") == "true"

    @pytest.mark.asyncio
    async def test_non_json_gets_false_header(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/plain")
            assert r.headers.get("x-pii-redacted") == "false"
