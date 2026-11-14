"""vgmelo-portal: auth, quotas and the demo gateway for vgmelo.tech."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .admin import router as admin_router
from .auth import describe
from .auth import router as auth_router
from .config import Settings
from .contact import router as contact_router
from .context import DB, AppContext, Ctx, MaybeCurrent
from .db import make_engine, make_sessionmaker
from .demos import router as demo_router
from .limits import DEMOS, Calendar, TooMany, tokens_used, used_today, user_tokens_used
from .mailer import Mailer
from .ml import router as ml_router
from .ml_assistant import router as ml_assistant_router
from .security import Keyring
from .studio import router as studio_router

logger = logging.getLogger("vgmelo_portal")

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Routes that take a file instead of a small JSON body (their own, larger, cap).
UPLOAD_PATHS = frozenset({"/api/demo/score/batch", "/api/demo/score/batch/validate"})
ML_UPLOAD_PATHS = frozenset({"/api/ml/runs"})
ASSISTANT_PATHS = frozenset({"/api/ml/assistant"})


def ml_cors_path(path: str) -> bool:
    """Rotas que o Giz (outro subdomínio) chama com o cookie de sessão."""
    return path == "/api/me" or path.startswith("/api/ml/")


class Guard:
    """Outermost middleware: Origin check on unsafe methods (CSRF), body size cap, security headers,
    and an access log that never includes query strings (OAuth codes travel there)."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app, self.settings = app, settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        method, path = scope["method"], scope["path"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        status_holder = {"status": 0}

        async def reply(status: int, detail: str) -> None:
            await JSONResponse({"detail": detail}, status)(scope, receive, send_wrapped)

        origin = headers.get("origin", "").rstrip("/")
        cors = bool(self.settings.ml_origin) and origin == self.settings.ml_origin and ml_cors_path(path)

        async def send_wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                extra = [*(self._cors_headers(origin) if cors else [])]
                extra += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"x-frame-options", b"DENY"),
                ]
                existing = {k.lower() for k, _ in message.get("headers", [])}
                if path.startswith("/api/") and b"cache-control" not in existing:
                    extra.append((b"cache-control", b"no-store"))
                message["headers"] = [*message.get("headers", []), *(h for h in extra if h[0] not in existing)]
            await send(message)

        try:
            if cors and method == "OPTIONS":  # preflight do Giz
                await send_wrapped(
                    {
                        "type": "http.response.start",
                        "status": 204,
                        "headers": [
                            (b"access-control-allow-methods", b"GET, POST"),
                            (b"access-control-allow-headers", b"content-type"),
                            (b"access-control-max-age", b"600"),
                        ],
                    }
                )
                await send_wrapped({"type": "http.response.body", "body": b""})
                return
            if method not in SAFE_METHODS and origin not in self.settings.origins and not cors:
                await reply(403, "bad_origin")
                return
            declared = headers.get("content-length")
            if path in ML_UPLOAD_PATHS:
                limit = self.settings.ml_max_bytes
            elif path in ASSISTANT_PATHS:
                limit = self.settings.assistant_max_bytes
            elif path in UPLOAD_PATHS:
                limit = self.settings.batch_max_bytes
            else:
                limit = self.settings.max_body_bytes
            if declared is not None and (not declared.isdigit() or int(declared) > limit):
                await reply(413, "body_too_large")
                return
            # Bodies here are tiny: read it all up front (chunked too), then replay it to the app.
            body = b""
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body += message.get("body", b"")
                if len(body) > limit:
                    await reply(413, "body_too_large")
                    return
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()  # later calls: disconnect notifications for streaming responses

            await self.app(scope, replay, send_wrapped)
        finally:
            logger.info(
                "%s %s %s %.0fms", method, path, status_holder["status"], (time.perf_counter() - started) * 1000
            )

    @staticmethod
    def _cors_headers(origin: str) -> list[tuple[bytes, bytes]]:
        return [
            (b"access-control-allow-origin", origin.encode("latin-1")),
            (b"access-control-allow-credentials", b"true"),
            (b"vary", b"Origin"),
        ]


def quiet_third_party_loggers() -> None:
    """httpx2 logs every request URL at INFO, query string included (OAuth codes, states)."""
    for noisy in ("httpx2", "httpcore", "mcp", "authlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(settings: Settings | None = None, transport: Any = None) -> FastAPI:
    quiet_third_party_loggers()
    cfg = settings or Settings()  # type: ignore[call-arg]
    engine = make_engine(cfg.database_url)
    ctx = AppContext(
        settings=cfg,
        sessions=make_sessionmaker(engine),
        keyring=Keyring(cfg.secret_key),
        mailer=Mailer(cfg.resend_api_key, cfg.email_from, httpx2.Client(transport=transport, timeout=10)),
        calendar=Calendar(cfg.quota_tz),
        http=httpx2.Client(transport=transport, timeout=10),
        ahttp=httpx2.AsyncClient(transport=transport, timeout=httpx2.Timeout(10, read=30)),
        transport=transport,
    )

    app = FastAPI(title="vgmelo-portal", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ctx = ctx
    app.state.engine = engine

    @app.exception_handler(TooMany)
    async def too_many(_: Request, exc: TooMany) -> JSONResponse:
        return JSONResponse(exc.detail, 429, headers=exc.headers)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/me")
    def me(current: MaybeCurrent, ctx: Ctx, db: DB) -> dict[str, Any]:
        user, pending = describe(current)
        day = ctx.calendar.today()
        used = used_today(db, current.user.id, day) if user and current else {}
        spent = user_tokens_used(db, current.user.id, day) if user and current else 0
        quotas = {}
        for demo in DEMOS:
            limit = cfg.quota_for(demo)
            quotas[demo] = {"limit": limit, "used": used.get(demo, 0), "remaining": max(0, limit - used.get(demo, 0))}
        return {
            "user": user,
            "pending": pending,
            "providers": {"google": cfg.google_enabled, "github": cfg.github_enabled, "email": cfg.email_enabled},
            "quotas": quotas,
            "resets_at": ctx.calendar.next_reset().isoformat(timespec="seconds"),
            "budget_exhausted": tokens_used(db, day) >= cfg.global_daily_tokens,
            "tokens": {
                "limit": cfg.user_daily_tokens,
                "used": spent,
                "remaining": max(0, cfg.user_daily_tokens - spent),
            },
        }

    app.include_router(auth_router)
    app.include_router(demo_router)
    app.include_router(studio_router)
    app.include_router(ml_router)
    app.include_router(ml_assistant_router)
    app.include_router(contact_router)
    app.include_router(admin_router)
    app.add_middleware(Guard, settings=cfg)
    return app


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        "vgmelo_portal.app:create_app",
        factory=True,
        host=os.getenv("HOST", "0.0.0.0"),  # noqa: S104 - inside the container, behind Caddy
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # concurrency limits for agents runs are in-process
        proxy_headers=False,  # X-Forwarded-For is read by the app, only from TRUSTED_PROXIES
        access_log=False,  # uvicorn's access log includes query strings (OAuth codes); Guard logs paths only
        server_header=False,
    )
