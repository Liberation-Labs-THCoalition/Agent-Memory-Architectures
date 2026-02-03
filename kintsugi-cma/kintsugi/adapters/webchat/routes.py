"""FastAPI routes for WebChat widget endpoints.

This module provides HTTP and WebSocket endpoints for the WebChat widget,
including configuration retrieval, session management, and embed code
generation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from kintsugi.adapters.webchat.config import WebChatConfig
from kintsugi.adapters.webchat.handler import WebChatHandler, WebChatMessageType, WebChatSession
from kintsugi.adapters.webchat.static import get_widget_css, get_widget_loader_js
from kintsugi.adapters.webchat.widget import WidgetConfigGenerator, WidgetPosition, WidgetTheme

logger = logging.getLogger("kintsugi.adapters.webchat")

router = APIRouter(prefix="/webchat", tags=["webchat"])


# ---------------------------------------------------------------------------
# Module-level state (would typically be injected via dependency)
# ---------------------------------------------------------------------------

# Store configurations per org - in production, load from database
_org_configs: dict[str, WebChatConfig] = {}

# Store handlers per org
_handlers: dict[str, WebChatHandler] = {}

# Store message history per session (in production, use persistent storage)
_message_history: dict[str, list[dict]] = {}


def get_or_create_handler(org_id: str) -> WebChatHandler:
    """Get or create a WebChatHandler for an organization.

    Args:
        org_id: The organization ID.

    Returns:
        WebChatHandler instance for the organization.
    """
    if org_id not in _handlers:
        config = _org_configs.get(org_id)
        if config is None:
            # Create default config for the org
            config = WebChatConfig(org_id=org_id)
            _org_configs[org_id] = config
        _handlers[org_id] = WebChatHandler(config)
    return _handlers[org_id]


def get_config(org_id: str) -> WebChatConfig | None:
    """Get configuration for an organization.

    Args:
        org_id: The organization ID.

    Returns:
        WebChatConfig if found, None otherwise.
    """
    return _org_configs.get(org_id)


def set_config(org_id: str, config: WebChatConfig) -> None:
    """Set configuration for an organization.

    Args:
        org_id: The organization ID.
        config: The WebChatConfig to set.
    """
    _org_configs[org_id] = config
    # Reset handler to pick up new config
    if org_id in _handlers:
        del _handlers[org_id]


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for creating a new chat session."""

    org_id: str
    user_identifier: str | None = None
    metadata: dict[str, Any] | None = None


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str
    org_id: str
    websocket_url: str
    created_at: str


class EndSessionResponse(BaseModel):
    """Response for session termination."""

    session_id: str
    ended: bool
    message: str


class MessageHistoryItem(BaseModel):
    """Single message in history."""

    id: str
    role: str  # "user" or "agent"
    content: str
    timestamp: str
    metadata: dict[str, Any] | None = None


class MessageHistoryResponse(BaseModel):
    """Response for message history request."""

    session_id: str
    messages: list[MessageHistoryItem]
    total_count: int


class WidgetConfigResponse(BaseModel):
    """Response for widget configuration."""

    org_id: str
    config: dict[str, Any]


class UpdateConfigRequest(BaseModel):
    """Request body for updating widget configuration."""

    widget_title: str | None = None
    widget_subtitle: str | None = None
    primary_color: str | None = None
    show_powered_by: bool | None = None
    allowed_origins: list[str] | None = None
    require_auth: bool | None = None
    session_timeout_minutes: int | None = None
    max_message_length: int | None = None
    rate_limit_messages_per_minute: int | None = None


# ---------------------------------------------------------------------------
# Configuration endpoints
# ---------------------------------------------------------------------------


@router.get("/config/{org_id}", response_model=WidgetConfigResponse)
async def get_widget_config(org_id: str) -> WidgetConfigResponse:
    """Get widget configuration for an organization.

    Args:
        org_id: The organization ID.

    Returns:
        Widget configuration including theme, position, and limits.

    Raises:
        HTTPException: If organization is not found.
    """
    # Validate org_id format
    try:
        uuid.UUID(org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID")

    config = get_config(org_id)
    if config is None:
        # Return default config for unknown org
        config = WebChatConfig(org_id=org_id)

    generator = WidgetConfigGenerator(
        base_url="",  # Will be populated by client
        org_id=org_id,
    )

    return WidgetConfigResponse(
        org_id=org_id,
        config=generator.generate_config_json(config),
    )


@router.put("/config/{org_id}")
async def update_widget_config(
    org_id: str,
    request: UpdateConfigRequest,
) -> WidgetConfigResponse:
    """Update widget configuration for an organization.

    Args:
        org_id: The organization ID.
        request: Configuration updates to apply.

    Returns:
        Updated widget configuration.
    """
    # Validate org_id format
    try:
        uuid.UUID(org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID")

    # Get existing config or create default
    existing = get_config(org_id)
    if existing is None:
        existing = WebChatConfig(org_id=org_id)

    # Apply updates
    updates = request.model_dump(exclude_none=True)
    config_dict = {
        "org_id": existing.org_id,
        "allowed_origins": updates.get("allowed_origins", existing.allowed_origins),
        "require_auth": updates.get("require_auth", existing.require_auth),
        "session_timeout_minutes": updates.get(
            "session_timeout_minutes", existing.session_timeout_minutes
        ),
        "max_message_length": updates.get(
            "max_message_length", existing.max_message_length
        ),
        "rate_limit_messages_per_minute": updates.get(
            "rate_limit_messages_per_minute", existing.rate_limit_messages_per_minute
        ),
        "widget_title": updates.get("widget_title", existing.widget_title),
        "widget_subtitle": updates.get("widget_subtitle", existing.widget_subtitle),
        "primary_color": updates.get("primary_color", existing.primary_color),
        "show_powered_by": updates.get("show_powered_by", existing.show_powered_by),
    }

    new_config = WebChatConfig(**config_dict)
    set_config(org_id, new_config)

    generator = WidgetConfigGenerator(base_url="", org_id=org_id)

    return WidgetConfigResponse(
        org_id=org_id,
        config=generator.generate_config_json(new_config),
    )


# ---------------------------------------------------------------------------
# Embed code endpoints
# ---------------------------------------------------------------------------


@router.get("/embed/{org_id}")
async def get_embed_code(
    org_id: str,
    format: str = Query("script", description="Embed format: script, iframe, or react"),
) -> Response:
    """Get embeddable code snippet for the chat widget.

    Args:
        org_id: The organization ID.
        format: Desired embed format - "script", "iframe", or "react".

    Returns:
        Code snippet in the requested format.

    Raises:
        HTTPException: If format is invalid or org not found.
    """
    # Validate org_id format
    try:
        uuid.UUID(org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID")

    valid_formats = {"script", "iframe", "react"}
    if format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}",
        )

    config = get_config(org_id)
    if config is None:
        config = WebChatConfig(org_id=org_id)

    # Use placeholder URL - client should replace with actual API URL
    generator = WidgetConfigGenerator(
        base_url="https://api.kintsugi.ai",
        org_id=org_id,
    )

    if format == "script":
        code = generator.generate_embed_code(config)
        content_type = "text/html"
    elif format == "iframe":
        code = generator.generate_iframe_embed_code(config)
        content_type = "text/html"
    else:  # react
        code = generator.generate_react_component_code(config)
        content_type = "text/plain"

    return Response(content=code, media_type=content_type)


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@router.post("/session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new chat session.

    Creates a new WebChat session for the specified organization,
    returning a session ID that can be used for WebSocket connections.

    Args:
        request: Session creation parameters.

    Returns:
        Session details including session_id and websocket_url.

    Raises:
        HTTPException: If org_id is invalid.
    """
    # Validate org_id format
    try:
        uuid.UUID(request.org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID")

    handler = get_or_create_handler(request.org_id)

    session = handler.create_session(
        org_id=request.org_id,
        user_identifier=request.user_identifier,
        metadata=request.metadata,
    )

    # Initialize empty message history
    _message_history[session.session_id] = []

    # Construct WebSocket URL
    ws_url = f"/ws/webchat/{request.org_id}?session_id={session.session_id}"

    return CreateSessionResponse(
        session_id=session.session_id,
        org_id=session.org_id,
        websocket_url=ws_url,
        created_at=session.connected_at.isoformat(),
    )


@router.delete("/session/{session_id}", response_model=EndSessionResponse)
async def end_session(session_id: str) -> EndSessionResponse:
    """End a chat session.

    Terminates an active chat session and cleans up associated resources.

    Args:
        session_id: The session ID to terminate.

    Returns:
        Confirmation of session termination.
    """
    # Find which handler has this session
    for org_id, handler in _handlers.items():
        session = handler.get_session(session_id)
        if session is not None:
            ended = handler.end_session(session_id)
            # Clean up message history
            _message_history.pop(session_id, None)
            return EndSessionResponse(
                session_id=session_id,
                ended=ended,
                message="Session ended successfully",
            )

    return EndSessionResponse(
        session_id=session_id,
        ended=False,
        message="Session not found or already ended",
    )


@router.get("/history/{session_id}", response_model=MessageHistoryResponse)
async def get_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
) -> MessageHistoryResponse:
    """Get message history for a session.

    Retrieves the conversation history for an active or recent session.

    Args:
        session_id: The session ID to get history for.
        limit: Maximum number of messages to return.
        offset: Number of messages to skip from the beginning.

    Returns:
        Message history with pagination info.

    Raises:
        HTTPException: If session is not found.
    """
    # Check if we have history for this session
    history = _message_history.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")

    total = len(history)
    messages = history[offset : offset + limit]

    return MessageHistoryResponse(
        session_id=session_id,
        messages=[MessageHistoryItem(**msg) for msg in messages],
        total_count=total,
    )


# ---------------------------------------------------------------------------
# Static asset endpoints
# ---------------------------------------------------------------------------


@router.get("/static/widget.css")
async def get_widget_stylesheet() -> Response:
    """Get the widget CSS stylesheet.

    Returns:
        CSS content for styling the chat widget.
    """
    css = get_widget_css()
    return Response(content=css, media_type="text/css")


@router.get("/static/loader.js")
async def get_widget_loader() -> Response:
    """Get the widget loader JavaScript.

    Returns:
        JavaScript for loading and initializing the widget.
    """
    js = get_widget_loader_js()
    return Response(content=js, media_type="application/javascript")


# ---------------------------------------------------------------------------
# Admin/stats endpoints
# ---------------------------------------------------------------------------


@router.get("/stats")
async def get_webchat_stats() -> dict:
    """Get WebChat statistics across all organizations.

    Returns:
        Statistics including session counts and message totals.
    """
    stats = {
        "total_orgs": len(_handlers),
        "total_sessions": 0,
        "orgs": {},
    }

    for org_id, handler in _handlers.items():
        org_stats = handler.get_session_stats()
        stats["orgs"][org_id] = org_stats
        stats["total_sessions"] += org_stats["total_sessions"]

    return stats


@router.post("/cleanup")
async def cleanup_expired_sessions() -> dict:
    """Trigger cleanup of expired sessions across all handlers.

    Returns:
        Summary of cleaned up sessions per organization.
    """
    results = {}
    total_cleaned = 0

    for org_id, handler in _handlers.items():
        cleaned = handler.cleanup_expired_sessions()
        results[org_id] = cleaned
        total_cleaned += cleaned

    return {
        "total_cleaned": total_cleaned,
        "by_org": results,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint (for direct integration)
# ---------------------------------------------------------------------------


@router.websocket("/ws/{org_id}")
async def webchat_websocket(
    websocket: WebSocket,
    org_id: str,
    session_id: str = Query(...),
) -> None:
    """WebSocket endpoint for WebChat connections.

    Handles real-time bidirectional communication for the chat widget.

    Args:
        websocket: The WebSocket connection.
        org_id: The organization ID.
        session_id: The session ID from the session creation endpoint.
    """
    handler = get_or_create_handler(org_id)
    session = handler.get_session(session_id)

    if session is None:
        await websocket.close(code=4001, reason="Invalid or expired session")
        return

    await websocket.accept()

    # Send connection confirmation
    await websocket.send_json({
        "type": WebChatMessageType.CONNECT.value,
        "session_id": session_id,
        "org_id": org_id,
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == WebChatMessageType.MESSAGE.value:
                content = data.get("content", "")
                result = await handler.handle_message(session_id, content)

                if result["type"] == WebChatMessageType.ERROR.value:
                    await websocket.send_json(result)
                else:
                    # Store message in history
                    msg_id = str(uuid.uuid4())
                    if session_id not in _message_history:
                        _message_history[session_id] = []

                    _message_history[session_id].append({
                        "id": msg_id,
                        "role": "user",
                        "content": content,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": None,
                    })

                    # Acknowledge message
                    await websocket.send_json({
                        "type": WebChatMessageType.MESSAGE.value,
                        "id": msg_id,
                        "status": "received",
                    })

                    # In a full implementation, this would trigger agent processing
                    # For now, send a placeholder response
                    await websocket.send_json({
                        "type": WebChatMessageType.AGENT_TYPING.value,
                    })

                    # Placeholder agent response
                    agent_msg_id = str(uuid.uuid4())
                    response_content = (
                        "Message received. Agent processing not yet implemented."
                    )

                    _message_history[session_id].append({
                        "id": agent_msg_id,
                        "role": "agent",
                        "content": response_content,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": None,
                    })

                    await websocket.send_json({
                        "type": WebChatMessageType.AGENT_RESPONSE.value,
                        "id": agent_msg_id,
                        "content": response_content,
                    })

            elif msg_type == WebChatMessageType.TYPING.value:
                # Typing indicator - could be forwarded to agents
                pass

            elif msg_type == WebChatMessageType.HISTORY.value:
                # Return message history
                history = _message_history.get(session_id, [])
                await websocket.send_json({
                    "type": WebChatMessageType.HISTORY.value,
                    "messages": history,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": WebChatMessageType.ERROR.value,
                    "error": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebChat session {session_id} disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        # Update last activity on disconnect
        session = handler.get_session(session_id)
        if session:
            session.update_activity()
