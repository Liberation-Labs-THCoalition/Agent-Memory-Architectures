"""WebSocket handlers for real-time streaming to connected clients."""

from __future__ import annotations

import enum
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("kintsugi.api")

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class MessageType(str, enum.Enum):
    AGENT_RESPONSE = "agent_response"
    SHADOW_STATUS = "shadow_status"
    TEMPORAL_EVENT = "temporal_event"
    CONSENSUS_UPDATE = "consensus_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Tracks active WebSocket connections per organisation."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, org_id: str) -> None:
        await websocket.accept()
        self._connections.setdefault(org_id, []).append(websocket)

    async def disconnect(self, websocket: WebSocket, org_id: str) -> None:
        conns = self._connections.get(org_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self._connections.pop(org_id, None)

    async def send_to_org(self, org_id: str, message: dict) -> None:
        for ws in list(self._connections.get(org_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                await self.disconnect(ws, org_id)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        await websocket.send_json(message)

    def get_connection_count(self, org_id: str | None = None) -> int:
        if org_id is not None:
            return len(self._connections.get(org_id, []))
        return sum(len(v) for v in self._connections.values())


# Module-level singleton
manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.websocket("/ws/{org_id}")
async def websocket_endpoint(websocket: WebSocket, org_id: str) -> None:
    await manager.connect(websocket, org_id)
    subscriptions: set[str] = set()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    "type": MessageType.ERROR,
                    "detail": "invalid JSON",
                })
                continue

            msg_type = data.get("type", "")

            if msg_type == "subscribe":
                channel = data.get("channel", "")
                if channel:
                    subscriptions.add(channel)
                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "channels": sorted(subscriptions),
                })

            elif msg_type == "ping":
                await manager.send_personal(websocket, {"type": "pong"})

            elif msg_type == "message":
                # Placeholder for agent interaction pipeline
                await manager.send_personal(websocket, {
                    "type": MessageType.AGENT_RESPONSE,
                    "detail": "message received (handler not yet implemented)",
                })

            else:
                await manager.send_personal(websocket, {
                    "type": MessageType.ERROR,
                    "detail": f"unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, org_id)


# ---------------------------------------------------------------------------
# Convenience broadcast helpers
# ---------------------------------------------------------------------------


async def broadcast_temporal_event(org_id: str, event: dict) -> None:
    """Push a temporal-memory event to all connections for *org_id*."""
    await manager.send_to_org(org_id, {
        "type": MessageType.TEMPORAL_EVENT,
        "payload": event,
    })


async def broadcast_shadow_status(org_id: str, status: dict) -> None:
    """Push a Kintsugi shadow-governance status update."""
    await manager.send_to_org(org_id, {
        "type": MessageType.SHADOW_STATUS,
        "payload": status,
    })
