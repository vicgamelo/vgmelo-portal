"""Live demos behind login, with per-user daily quotas and daily token budgets (global and per user)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal, get_args

import httpx2
from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import credit_agent
from .context import AppContext, Ctx, LoggedIn
from .db import DemoRun, affected
from .limits import (
    TooMany,
    add_tokens,
    check_budget,
    check_user_tokens,
    crossed_alerts,
    hit,
    refund_quota,
    take_quota,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo")

SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def alert_budget(ctx: AppContext, db: Session, day: str) -> None:
    """Tells the owner when the agents demo is eating its daily token budget (each threshold once a day)."""
    s = ctx.settings
    budget = s.global_daily_tokens
    for pct, used in crossed_alerts(db, day, budget, s.budget_alert_percents):
        what = "acabou: a demo de agentes e o estúdio param até a meia-noite" if pct >= 100 else f"chegou a {pct}%"
        ctx.mailer.send(
            s.alert_to or s.contact_to,
            f"vgmelo.tech: teto de tokens de hoje {what}",
            f"Demos de agentes e estúdio, {day}: {used:,} de {budget:,} tokens ({used * 100 // budget}%).\n"
            "O teto zera à meia-noite (America/Sao_Paulo). Ele fica abaixo do plano grátis do Groq/OpenRouter;\n"
            "se os avisos virarem rotina, baixe GLOBAL_DAILY_TOKENS, USER_DAILY_TOKENS ou as cotas no .env da VPS.\n\n"
            f"Execuções e consumo por dia: {s.site_url}/admin\n",
        )


async def in_db[T](ctx: AppContext, fn: Callable[[Session], T]) -> T:
    def run() -> T:
        with ctx.sessions() as db:
            return fn(db)

    return await run_in_threadpool(run)


async def reserve(ctx: AppContext, user_id: str, demo: str) -> str:
    day = ctx.calendar.today()
    limit, reset = ctx.settings.quota_for(demo), ctx.calendar.next_reset()
    await in_db(ctx, lambda db: take_quota(db, user_id, day, demo, limit, reset))
    return day


async def refund(ctx: AppContext, user_id: str, day: str, demo: str) -> None:
    await in_db(ctx, lambda db: refund_quota(db, user_id, day, demo))


def upstream_error() -> HTTPException:
    return HTTPException(502, "upstream_unavailable")


# ---------------------------------------------------------------- agents (SSE)


# MCP servers the visitor can turn on and off in the agents demo (the orchestrator's names).
MCPServerName = Literal["brasil", "noticias", "normas", "agenda"]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1)
    # Omitted: every server. Repeated names are harmless; the orchestrator takes the set.
    servers: list[MCPServerName] | None = Field(default=None, max_length=4)


def _frame(event: str | None, data: list[str]) -> str:
    head = f"event: {event}\n" if event else ""
    return head + "".join(f"data: {line}\n" for line in data) + "\n"


async def _sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, list[str]]]:
    event: str | None = None
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data or event:
                yield event, data
            event, data = None, []
        elif line.startswith(":"):
            yield ":", [line[1:].strip()]
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data or event:
        yield event, data


@router.post("/agents/stream")
async def agents_stream(body: Question, current: LoggedIn, ctx: Ctx) -> StreamingResponse:
    if len(body.question) > ctx.settings.question_max_chars:
        raise HTTPException(422, "question_too_long")
    payload: dict[str, Any] = {"input": body.question}
    if body.servers is not None:
        payload["servers"] = sorted(set(body.servers))
    if ctx.settings.orchestrator_agent:
        payload["agent"] = ctx.settings.orchestrator_agent
    return await relay_run(ctx, current.user.id, "agents", "/v1/runs/stream", payload)


async def relay_run(
    ctx: AppContext, user_id: str, demo: str, path: str, payload: dict[str, Any], *, invalid: str | None = None
) -> StreamingResponse:
    """Runs on the orchestrator and relays its SSE, charging ``demo``'s quota and the token budget.

    ``invalid``: the reason to answer 422 with when the orchestrator refuses the payload (422); without
    it, a refusal is an upstream error like any other."""
    s = ctx.settings
    day, reset = ctx.calendar.today(), ctx.calendar.next_reset()
    # Check-and-add with no await in between: one event loop, so this is atomic.
    if user_id in ctx.active_runs or len(ctx.active_runs) >= s.max_concurrent_runs:
        raise TooMany("busy", datetime.now(UTC) + timedelta(seconds=30))
    ctx.active_runs.add(user_id)
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            ctx.active_runs.discard(user_id)

    try:

        def admit(db: Session) -> None:
            check_budget(db, day, s.global_daily_tokens, reset)
            check_user_tokens(db, user_id, day, s.user_daily_tokens, reset)

        await in_db(ctx, admit)
        await reserve(ctx, user_id, demo)
    except BaseException:
        release()
        raise

    request = ctx.ahttp.build_request(
        "POST",
        f"{s.orchestrator_url}{path}",
        json=payload,
        headers={"X-API-Key": s.orchestrator_api_key, "Accept": "text/event-stream"},
        timeout=httpx2.Timeout(10, read=s.orchestrator_timeout_s),
    )
    try:
        upstream = await ctx.ahttp.send(request, stream=True)
    except httpx2.HTTPError as exc:
        logger.warning("orchestrator unreachable: %s", type(exc).__name__)
        await refund(ctx, user_id, day, demo)
        release()
        raise upstream_error() from exc
    if upstream.status_code != 200:
        logger.warning("orchestrator answered %s", upstream.status_code)
        detail = ""
        if upstream.status_code == 422 and invalid:
            try:
                await upstream.aread()
                found = upstream.json().get("detail")
                detail = found if isinstance(found, str) else ""
            except (httpx2.HTTPError, ValueError, AttributeError):
                pass
        await upstream.aclose()
        await refund(ctx, user_id, day, demo)
        release()
        if upstream.status_code == 422 and invalid:
            raise HTTPException(422, {"reason": invalid, "message": detail[:300]})
        raise upstream_error()

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def record(done: dict[str, Any]) -> Callable[[Session], None]:
        tokens = int(done.get("input_tokens") or 0) + int(done.get("output_tokens") or 0)

        def run(db: Session) -> None:
            db.add(
                DemoRun(
                    run_id=str(done["run_id"])[:64],
                    trace_id=(str(done["trace_id"])[:64] if done.get("trace_id") else None),
                    user_id=user_id,
                    day=day,
                    tokens=tokens,
                )
            )
            db.commit()
            add_tokens(db, day, tokens, user_id)
            alert_budget(ctx, db, day)

        return run

    async def pump() -> None:
        """Reads the whole upstream run even if the visitor disconnects, so its tokens are always counted."""
        recorded = False
        try:
            async for event, data in _sse(upstream.aiter_lines()):
                if event == ":":
                    await queue.put(f": {data[0]}\n\n")
                    continue
                if event == "done":
                    try:
                        await in_db(ctx, record(json.loads("\n".join(data))))
                        recorded = True
                    except Exception:
                        logger.exception("could not record run")
                await queue.put(_frame(event, data))
        except httpx2.HTTPError as exc:
            logger.warning("orchestrator stream broke: %s", type(exc).__name__)
            await queue.put(_frame("error", [json.dumps({"message": "upstream_unavailable"})]))
        finally:
            await upstream.aclose()
            release()
            await queue.put(None)
            if not recorded:
                logger.warning("%s run ended without a done event user=%s", demo, user_id)

    task = asyncio.create_task(pump())
    ctx.background.add(task)
    task.add_done_callback(ctx.background.discard)

    async def relay() -> AsyncIterator[str]:
        while (item := await queue.get()) is not None:
            yield item

    return StreamingResponse(relay(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/agents/servers")
async def agents_servers(current: LoggedIn, ctx: Ctx) -> list[dict[str, Any]]:
    """Which MCP servers are up and which agents each one feeds, for the demo's on/off switches."""
    s = ctx.settings
    try:
        resp = await ctx.ahttp.get(
            f"{s.orchestrator_url}/v1/servers", headers={"X-API-Key": s.orchestrator_api_key}, timeout=10
        )
    except httpx2.HTTPError as exc:
        raise upstream_error() from exc
    if resp.status_code != 200:
        raise upstream_error()
    allowed = set(get_args(MCPServerName))
    return [
        {"name": x["name"], "up": bool(x.get("up")), "agents": list(x.get("agents") or [])}
        for x in resp.json()
        if isinstance(x, dict) and x.get("name") in allowed
    ]


# ---------------------------------------------------------------- traces (genai-lens)

TraceId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: TraceId, current: LoggedIn, ctx: Ctx) -> list[dict[str, Any]]:
    user_id = current.user.id
    run = await in_db(
        ctx, lambda db: db.scalar(select(DemoRun).where(DemoRun.trace_id == trace_id, DemoRun.user_id == user_id))
    )
    if run is None:
        raise HTTPException(404, "trace_not_found")  # same answer for "not yours" and "doesn't exist"
    day = await reserve(ctx, user_id, "traces")
    try:
        resp = await ctx.ahttp.get(f"{ctx.settings.lens_url}/api/traces/{trace_id}", timeout=10)
    except httpx2.HTTPError as exc:
        await refund(ctx, user_id, day, "traces")
        raise upstream_error() from exc
    if resp.status_code == 404:
        await refund(ctx, user_id, day, "traces")
        raise HTTPException(404, "trace_not_found")  # not exported yet: the site retries
    if resp.status_code != 200:
        await refund(ctx, user_id, day, "traces")
        raise upstream_error()
    spans: list[dict[str, Any]] = resp.json()
    # The lens has the per-call numbers; if they add up to more than the run reported, charge the difference.
    lens_tokens = sum(int(sp.get("input_tokens") or 0) + int(sp.get("output_tokens") or 0) for sp in spans
                      if sp.get("kind") == "llm")  # fmt: skip
    if lens_tokens > run.tokens:
        extra = lens_tokens - run.tokens

        def reconcile(db: Session) -> None:
            res = db.execute(
                update(DemoRun)
                .where(DemoRun.run_id == run.run_id, DemoRun.tokens == run.tokens)
                .values(tokens=lens_tokens)
            )
            db.commit()
            if affected(res):
                add_tokens(db, run.day, extra, run.user_id)
                alert_budget(ctx, db, run.day)

        await in_db(ctx, reconcile)
    return spans


# ---------------------------------------------------------------- credit scoring (mlops-platform)

Short = Annotated[str, Field(min_length=1, max_length=64)]


class Applicant(BaseModel):
    """Same fields and bounds as mlops-platform's /predict; unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")
    checking_status: Literal["<0", "0<=X<1940", ">=1940", "no checking"]
    duration: Annotated[int, Field(ge=1, le=120)]
    credit_history: Short
    purpose: Short
    credit_amount: Annotated[float, Field(gt=0, le=10_000_000)]  # reais
    savings_status: Short
    employment: Short
    installment_commitment: Annotated[int, Field(ge=1, le=4)]
    other_parties: Short
    residence_since: Annotated[int, Field(ge=1, le=4)]
    property_magnitude: Short
    age: Annotated[int, Field(ge=18, le=100)]
    other_payment_plans: Short
    housing: Short
    existing_credits: Annotated[int, Field(ge=0, le=10)]
    job: Short
    num_dependents: Annotated[int, Field(ge=0, le=10)]
    own_telephone: Literal["none", "yes"]


@router.post("/score")
async def score(applicant: Applicant, current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    user_id = current.user.id
    day = await reserve(ctx, user_id, "score")
    try:
        resp = await ctx.ahttp.post(
            f"{ctx.settings.scoring_url}/predict", json={"applicants": [applicant.model_dump()]}, timeout=15
        )
        if resp.status_code == 422:  # e.g. a category outside credit-g: the visitor's input, not an outage
            await refund(ctx, user_id, day, "score")
            raise HTTPException(422, "invalid_applicant")
        resp.raise_for_status()
        data = resp.json()
        return {"model_version": data["model_version"], "prediction": data["predictions"][0]}
    except (httpx2.HTTPError, KeyError, IndexError, ValueError) as exc:
        logger.warning("scoring failed: %s", type(exc).__name__)
        await refund(ctx, user_id, day, "score")
        raise upstream_error() from exc


# Live pipeline: values stay loose (text, number or null = "não sei") because showing the casts and the
# imputation is the point of the demo. The scoring service validates the domain and answers 422 before
# streaming; the quota is refunded then.
Loose = StrictInt | StrictFloat | Annotated[str, Field(max_length=64)] | None
Lang = Literal["pt", "en"]


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checking_status: Loose
    duration: Loose
    credit_history: Loose
    purpose: Loose
    credit_amount: Loose
    savings_status: Loose
    employment: Loose
    installment_commitment: Loose
    other_parties: Loose
    residence_since: Loose
    property_magnitude: Loose
    age: Loose
    other_payment_plans: Loose
    housing: Loose
    existing_credits: Loose
    job: Loose
    num_dependents: Loose
    own_telephone: Loose


class PipelineBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicant: Draft
    lang: Lang = "pt"


def _problems(resp: httpx2.Response) -> list[dict[str, Any]]:
    """Field problems from the scoring service's 422, reduced to known keys."""
    try:
        detail = resp.json().get("detail", {})
        problems = detail.get("problems", []) if isinstance(detail, dict) else []
    except (ValueError, AttributeError):
        return []
    keys = ("column", "code", "message", "value", "severity")
    return [{k: p.get(k) for k in keys} for p in problems[:40] if isinstance(p, dict)]


@router.post("/score/pipeline")
async def score_pipeline(body: PipelineBody, current: LoggedIn, ctx: Ctx) -> StreamingResponse:
    """One use of the score quota: streams every step of the champion's prediction (SSE)."""
    user_id = current.user.id
    day = await reserve(ctx, user_id, "score")
    request = ctx.ahttp.build_request(
        "POST",
        f"{ctx.settings.scoring_url}/pipeline",
        params={"lang": body.lang},
        json={"applicant": body.applicant.model_dump()},
        headers={"Accept": "text/event-stream"},
        timeout=httpx2.Timeout(10, read=30),
    )
    try:
        upstream = await ctx.ahttp.send(request, stream=True)
    except httpx2.HTTPError as exc:
        logger.warning("scoring unreachable: %s", type(exc).__name__)
        await refund(ctx, user_id, day, "score")
        raise upstream_error() from exc
    if upstream.status_code != 200:
        await upstream.aread()
        await upstream.aclose()
        await refund(ctx, user_id, day, "score")
        if upstream.status_code == 422:
            raise HTTPException(422, {"reason": "invalid_applicant", "problems": _problems(upstream)})
        if upstream.status_code == 429:
            raise TooMany("busy", datetime.now(UTC) + timedelta(seconds=5))
        logger.warning("scoring pipeline answered %s", upstream.status_code)
        raise upstream_error()

    async def relay() -> AsyncIterator[str]:
        # Re-framed with \n line endings (the site's reader splits on blank lines), comments dropped.
        try:
            async for event, data in _sse(upstream.aiter_lines()):
                if event != ":":
                    yield _frame(event, data)
        except httpx2.HTTPError as exc:
            logger.warning("scoring stream broke: %s", type(exc).__name__)
            yield _frame("error", [json.dumps({"message": "upstream_unavailable"})])
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), media_type="text/event-stream", headers=SSE_HEADERS)


CSV_TYPES = frozenset({"text/csv", "text/plain"})


async def _csv_body(request: Request) -> bytes:
    """The file goes as the raw body (text/csv): no multipart parser, no JSON escaping inflating it.
    The Guard already capped its size at BATCH_MAX_BYTES."""
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype not in CSV_TYPES:
        raise HTTPException(415, "csv_expected")
    body = await request.body()
    if not body.strip():
        raise HTTPException(422, "empty_file")
    return body


async def _to_scoring(ctx: AppContext, path: str, body: bytes, lang: Lang) -> httpx2.Response:
    return await ctx.ahttp.post(
        f"{ctx.settings.scoring_url}{path}",
        params={"lang": lang},
        content=body,
        headers={"content-type": "text/csv"},
        timeout=httpx2.Timeout(10, read=30),
    )


@router.post("/score/batch/validate")
async def score_batch_validate(
    request: Request, current: LoggedIn, ctx: Ctx, lang: Annotated[Lang, Query()] = "pt"
) -> dict[str, Any]:
    """Checks the file (what is wrong, by line and column). Free, but rate limited per user."""
    body = await _csv_body(request)
    user_id = current.user.id
    window = ctx.window("score_validate", ctx.settings.batch_validate_per_hour, 3600)
    await in_db(ctx, lambda db: hit(db, window, user_id))
    try:
        resp = await _to_scoring(ctx, "/batch/validate", body, lang)
    except httpx2.HTTPError as exc:
        raise upstream_error() from exc
    if resp.status_code == 413:
        raise HTTPException(413, "body_too_large")
    if resp.status_code != 200:
        logger.warning("scoring batch/validate answered %s", resp.status_code)
        raise upstream_error()
    report: dict[str, Any] = resp.json()
    return report


@router.post("/score/batch")
async def score_batch(
    request: Request, current: LoggedIn, ctx: Ctx, lang: Annotated[Lang, Query()] = "pt"
) -> dict[str, Any]:
    """Scores the valid rows of the file. The whole file is one use of the score quota, refunded when
    nothing in it could be scored."""
    body = await _csv_body(request)
    user_id = current.user.id
    day = await reserve(ctx, user_id, "score")
    try:
        resp = await _to_scoring(ctx, "/batch/score", body, lang)
    except httpx2.HTTPError as exc:
        await refund(ctx, user_id, day, "score")
        raise upstream_error() from exc
    if resp.status_code == 200:
        try:
            result: dict[str, Any] = resp.json()
            return result
        except ValueError:
            pass
    await refund(ctx, user_id, day, "score")
    if resp.status_code == 413:
        raise HTTPException(413, "body_too_large")
    if resp.status_code == 422:
        try:
            report = resp.json()["detail"]["report"]
        except (ValueError, KeyError, TypeError):
            report = None
        raise HTTPException(422, {"reason": "no_valid_rows", "report": report})
    logger.warning("scoring batch/score answered %s", resp.status_code)
    raise upstream_error()


class AgentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: credit_agent.FieldName
    answer: Annotated[str, Field(min_length=1, max_length=120)]


@router.post("/score/agent")
async def score_agent(body: AgentBody, current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    """Microagente: interpreta a resposta livre de um campo. Não gasta a cota de score; tem limite por
    hora e usa o teto global de tokens. Qualquer falha aqui faz o site cair nas regras."""
    s = ctx.settings
    if not s.groq_api_key:
        raise HTTPException(503, "agent_off")
    user_id = current.user.id
    day, reset = ctx.calendar.today(), ctx.calendar.next_reset()
    window = ctx.window("score_agent", s.credit_agent_per_hour, 3600)

    def admit(db: Session) -> None:
        check_budget(db, day, s.global_daily_tokens, reset)
        check_user_tokens(db, user_id, day, s.user_daily_tokens, reset)
        hit(db, window, user_id)

    await in_db(ctx, admit)
    payload: dict[str, Any] = {
        "model": s.credit_agent_model,
        "messages": credit_agent.messages(body.field, body.answer),
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    if s.credit_agent_model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"  # o raciocínio conta no max_tokens
    try:
        resp = await ctx.ahttp.post(
            s.groq_url, json=payload, headers={"Authorization": f"Bearer {s.groq_api_key}"}, timeout=10
        )
    except httpx2.HTTPError as exc:
        logger.warning("groq unreachable: %s", type(exc).__name__)
        raise upstream_error() from exc
    if resp.status_code != 200:
        logger.warning("groq answered %s", resp.status_code)
        raise upstream_error()
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise upstream_error() from exc

    def spend(db: Session) -> None:
        add_tokens(db, day, tokens, user_id)
        alert_budget(ctx, db, day)

    await in_db(ctx, spend)
    return {**credit_agent.read(body.field, content), "model": s.credit_agent_model}


# ---------------------------------------------------------------- MCP tools (mcp-server-brasil)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CepArgs(_Args):
    cep: str = Field(min_length=8, max_length=9, pattern=r"^\d{5}-?\d{3}$")


class CnpjArgs(_Args):
    cnpj: str = Field(min_length=14, max_length=18, pattern=r"^[0-9A-Za-z./-]+$")


class DocumentoArgs(_Args):
    documento: str = Field(min_length=11, max_length=18, pattern=r"^[0-9A-Za-z./-]+$")


class FeriadosArgs(_Args):
    ano: int = Field(ge=1900, le=2199)


class DiasUteisArgs(_Args):
    inicio: date
    fim: date


class IndicadorArgs(_Args):
    indicador: Literal["selic_meta", "selic", "cdi", "ipca", "igpm", "dolar", "euro"]
    ultimos: int = Field(default=12, ge=1, le=120)


class MunicipiosArgs(_Args):
    uf: str = Field(pattern=r"^[A-Z]{2}$")


class DddArgs(_Args):
    ddd: str = Field(pattern=r"^\d{2}$")


MCP_TOOLS: dict[str, type[_Args]] = {
    "consultar_cep": CepArgs,
    "consultar_cnpj": CnpjArgs,
    "validar_documento": DocumentoArgs,
    "feriados_nacionais": FeriadosArgs,
    "dias_uteis": DiasUteisArgs,
    "indicador_economico": IndicadorArgs,
    "listar_municipios": MunicipiosArgs,
    "consultar_ddd": DddArgs,
}


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/mcp/{tool}")
async def call_mcp(tool: str, body: ToolCall, current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    model = MCP_TOOLS.get(tool)
    if model is None:
        raise HTTPException(404, "unknown_tool")
    try:
        args = model.model_validate(body.args).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(422, [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]) from exc
    user_id = current.user.id
    day = await reserve(ctx, user_id, "mcp")
    try:
        async with (
            httpx2.AsyncClient(transport=ctx.transport, timeout=httpx2.Timeout(15, read=30)) as http,
            Client(streamable_http_client(ctx.settings.mcp_url, http_client=http)) as client,
        ):
            result = await client.call_tool(tool, args)
    except Exception as exc:
        logger.warning("mcp call %s failed: %s", tool, type(exc).__name__)
        await refund(ctx, user_id, day, "mcp")
        raise upstream_error() from exc
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)[:500]
        raise HTTPException(422, {"reason": "tool_error", "message": text})
    content: Any = result.structured_content
    if isinstance(content, dict) and set(content) == {"result"}:
        content = content["result"]
    if content is None:
        texts = [getattr(c, "text", "") for c in result.content]
        try:
            content = json.loads(texts[0]) if len(texts) == 1 else texts
        except ValueError:
            content = texts
    return {"tool": tool, "result": content}
