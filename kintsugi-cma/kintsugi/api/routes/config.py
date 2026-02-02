"""Organization config / values endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kintsugi.config.values_loader import load_from_template, merge_with_defaults
from kintsugi.config.values_schema import OrganizationValues
from kintsugi.db import get_session
from kintsugi.models.base import Organization

router = APIRouter(prefix="/api/config", tags=["config"])

AVAILABLE_TEMPLATES = ["mutual_aid", "nonprofit_501c3", "cooperative", "advocacy"]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ValuesPayload(BaseModel):
    org_id: uuid.UUID
    values: dict


class InitPayload(BaseModel):
    org_name: str
    org_type: str
    overrides: dict = Field(default_factory=dict)


class InitResponse(BaseModel):
    org_id: uuid.UUID
    values: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/values")
async def get_values(
    org_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return the OrganizationValues document for an org."""
    result = await session.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found.")
    if org.values_json is None:
        return {
            "org_id": str(org_id),
            "values": {},
            "message": "No values configured yet. Use POST /api/config/init or PUT /api/config/values to set up.",
        }
    return {"org_id": str(org_id), "values": org.values_json}


@router.put("/values")
async def put_values(
    payload: ValuesPayload,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Validate and persist a full OrganizationValues document."""
    # Validate against schema
    try:
        validated = OrganizationValues.model_validate(payload.values)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    result = await session.execute(select(Organization).where(Organization.id == payload.org_id))
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found.")

    org.values_json = validated.model_dump()
    await session.commit()
    await session.refresh(org)

    return {"org_id": str(payload.org_id), "values": org.values_json}


@router.get("/templates")
async def list_templates() -> dict:
    """Return the list of available org-type templates."""
    return {"templates": AVAILABLE_TEMPLATES}


@router.get("/templates/{org_type}")
async def get_template(org_type: str) -> dict:
    """Load and return a specific template by org_type."""
    if org_type not in AVAILABLE_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Unknown template: {org_type}")
    values = load_from_template(org_type)
    return {"org_type": org_type, "values": values.model_dump()}


@router.post("/init")
async def init_org(
    payload: InitPayload,
    session: AsyncSession = Depends(get_session),
) -> InitResponse:
    """Create a new Organization and initialise its values from a template."""
    if payload.org_type not in AVAILABLE_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Unknown template: {payload.org_type}")

    # Build values: template + optional overrides
    if payload.overrides:
        values = merge_with_defaults(payload.overrides, payload.org_type)
    else:
        values = load_from_template(payload.org_type)

    org = Organization(
        name=payload.org_name,
        org_type=payload.org_type,
        values_json=values.model_dump(),
    )
    session.add(org)
    await session.commit()
    await session.refresh(org)

    return InitResponse(org_id=org.id, values=org.values_json)
