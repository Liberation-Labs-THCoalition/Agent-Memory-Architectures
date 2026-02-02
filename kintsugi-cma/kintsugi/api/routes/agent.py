"""Agent message endpoint — Phase 1 security + memory pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kintsugi.db import get_session
from kintsugi.models.base import Organization, TemporalMemory
from kintsugi.security.monitor import SecurityMonitor
from kintsugi.security.pii import PIIRedactor

router = APIRouter(prefix="/api/agent", tags=["agent"])

_monitor = SecurityMonitor()
_redactor = PIIRedactor()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    message: str
    org_id: str
    context: dict = {}


class AgentResponse(BaseModel):
    response: str
    org_id: str
    security_verdict: str
    redacted_input: str
    memory_context: list[dict] = []
    temporal_event_id: str | None = None


class TemporalEvent(BaseModel):
    id: str
    category: str
    message: str
    metadata: dict | None = None
    created_at: datetime


class TemporalListResponse(BaseModel):
    events: list[TemporalEvent]
    org_id: str
    count: int


# ---------------------------------------------------------------------------
# POST /api/agent/message
# ---------------------------------------------------------------------------

@router.post("/message", response_model=AgentResponse)
async def agent_message(
    req: AgentRequest,
    session: AsyncSession = Depends(get_session),
) -> AgentResponse:
    # 1. Validate org_id
    try:
        org_uuid = uuid.UUID(req.org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID.")

    result = await session.execute(
        select(Organization).where(Organization.id == org_uuid)
    )
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail=f"Organization {req.org_id} not found.")

    # 2. PII redaction
    redaction = _redactor.redact(req.message)
    redacted_text = redaction.redacted_text

    # 3. Security check on original message
    verdict = _monitor.check_text(req.message)
    verdict_str = verdict.verdict.value.lower()  # "allow" / "block" / "warn"

    # 4/5. Log to TemporalMemory and build response
    if verdict_str == "block":
        event = TemporalMemory(
            org_id=org_uuid,
            category="security",
            message=redacted_text,
            metadata_json={
                "security_verdict": verdict_str,
                "security_reason": verdict.reason,
                "matched_pattern": verdict.matched_pattern,
                "severity": verdict.severity.value if verdict.severity else None,
                "context": req.context,
                "pii_types_found": redaction.types_found,
            },
        )
        session.add(event)
        await session.flush()

        return AgentResponse(
            response=f"Message blocked by security monitor: {verdict.reason}",
            org_id=req.org_id,
            security_verdict=verdict_str,
            redacted_input=redacted_text,
            memory_context=[],
            temporal_event_id=str(event.id),
        )

    # ALLOW or WARN
    event = TemporalMemory(
        org_id=org_uuid,
        category="interaction",
        message=redacted_text,
        metadata_json={
            "security_verdict": verdict_str,
            "security_reason": verdict.reason,
            "context": req.context,
            "pii_types_found": redaction.types_found,
        },
    )
    session.add(event)
    await session.flush()

    warning_note = ""
    if verdict_str == "warn":
        warning_note = f" Warning: {verdict.reason}"

    return AgentResponse(
        response=(
            "Message processed through security pipeline. "
            f"LLM integration pending (Phase 2).{warning_note}"
        ),
        org_id=req.org_id,
        security_verdict=verdict_str,
        redacted_input=redacted_text,
        memory_context=[],
        temporal_event_id=str(event.id),
    )


# ---------------------------------------------------------------------------
# GET /api/agent/temporal
# ---------------------------------------------------------------------------

@router.get("/temporal", response_model=TemporalListResponse)
async def get_temporal_events(
    org_id: str = Query(..., description="Organization UUID"),
    limit: int = Query(20, ge=1, le=200),
    category: Optional[str] = Query(None, description="Filter by category"),
    session: AsyncSession = Depends(get_session),
) -> TemporalListResponse:
    try:
        org_uuid = uuid.UUID(org_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid org_id: not a valid UUID.")

    stmt = (
        select(TemporalMemory)
        .where(TemporalMemory.org_id == org_uuid)
        .order_by(TemporalMemory.created_at.desc())
        .limit(limit)
    )
    if category:
        stmt = stmt.where(TemporalMemory.category == category)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    events = [
        TemporalEvent(
            id=str(row.id),
            category=row.category,
            message=row.message,
            metadata=row.metadata_json,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return TemporalListResponse(events=events, org_id=org_id, count=len(events))
