"""Agent studio: each user writes agents (instructions, model, tools), saves private versions and runs them.

The orchestrator decides which models and tools exist and enforces its own limits; the portal keeps the
versions, the quota and the shape of what it forwards."""

from __future__ import annotations

import json
import secrets
from typing import Annotated, Any

import httpx2
from fastapi import APIRouter, HTTPException, Path, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .context import Ctx, LoggedIn
from .db import StudioVersion
from .demos import in_db, relay_run, upstream_error
from .limits import hit

router = APIRouter(prefix="/api/studio")

ToolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)]


class AgentSpec(BaseModel):
    """What the visitor wrote. Bounds here are only the shape; the orchestrator checks the rest
    (which models and tools exist right now, its own caps on steps and instruction length)."""

    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1, max_length=4000)
    model: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=40)
    tools: list[ToolName] = Field(default_factory=list, max_length=30)
    temperature: float = Field(default=0.2, ge=0, le=1.5)
    max_steps: int = Field(default=4, ge=1, le=10)


class NewVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=60)
    note: str | None = Field(default=None, max_length=200)
    agent: AgentSpec


class StudioRun(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1)
    agent: AgentSpec


VersionId = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]


def public(v: StudioVersion) -> dict[str, Any]:
    return {
        "id": v.id,
        "name": v.name,
        "version": v.version,
        "note": v.note,
        "created_at": v.created_at.isoformat(timespec="seconds"),
        "agent": json.loads(v.spec),
    }


@router.get("/catalog")
async def catalog(current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    """Models, tools, starting templates and limits, straight from the orchestrator."""
    s = ctx.settings
    try:
        resp = await ctx.ahttp.get(
            f"{s.orchestrator_url}/v1/studio", headers={"X-API-Key": s.orchestrator_api_key}, timeout=10
        )
    except httpx2.HTTPError as exc:
        raise upstream_error() from exc
    if resp.status_code != 200:
        raise upstream_error()
    data: dict[str, Any] = resp.json()
    return {k: data.get(k) for k in ("models", "tools", "templates", "limits")}


@router.get("/versions")
async def list_versions(current: LoggedIn, ctx: Ctx) -> list[dict[str, Any]]:
    user_id = current.user.id

    def run(db: Session) -> list[dict[str, Any]]:
        rows = db.scalars(
            select(StudioVersion)
            .where(StudioVersion.user_id == user_id)
            .order_by(StudioVersion.name, StudioVersion.version.desc())
        )
        return [public(v) for v in rows]

    return await in_db(ctx, run)


@router.post("/versions", status_code=201)
async def save_version(body: NewVersion, current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    user_id = current.user.id
    s = ctx.settings

    def run(db: Session) -> dict[str, Any]:
        hit(db, ctx.window("studio_save", s.studio_saves_per_hour, 3600), user_id)
        total = db.scalar(select(func.count()).select_from(StudioVersion).where(StudioVersion.user_id == user_id))
        if (total or 0) >= s.studio_max_versions:
            raise HTTPException(409, "too_many_versions")
        for _ in range(3):  # two saves at once: the unique (user, name, version) makes one retry
            last = db.scalar(
                select(func.max(StudioVersion.version)).where(
                    StudioVersion.user_id == user_id, StudioVersion.name == body.name
                )
            )
            row = StudioVersion(
                id=secrets.token_hex(8),
                user_id=user_id,
                name=body.name,
                version=(last or 0) + 1,
                note=body.note or None,
                spec=body.agent.model_dump_json(),
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                continue
            return public(row)
        raise HTTPException(409, "version_conflict")

    return await in_db(ctx, run)


@router.delete("/versions/{version_id}", status_code=204)
async def delete_version(version_id: VersionId, current: LoggedIn, ctx: Ctx) -> Response:
    user_id = current.user.id

    def run(db: Session) -> bool:
        row = db.get(StudioVersion, version_id)
        if row is None or row.user_id != user_id:
            return False
        db.delete(row)
        db.commit()
        return True

    if not await in_db(ctx, run):
        raise HTTPException(404, "version_not_found")  # same answer for "not yours" and "doesn't exist"
    return Response(status_code=204)


@router.post("/run/stream")
async def run_stream(body: StudioRun, current: LoggedIn, ctx: Ctx) -> StreamingResponse:
    if len(body.question) > ctx.settings.question_max_chars:
        raise HTTPException(422, "question_too_long")
    payload = {"input": body.question, "agent": body.agent.model_dump()}
    return await relay_run(ctx, current.user.id, "studio", "/v1/studio/runs/stream", payload, invalid="invalid_agent")
