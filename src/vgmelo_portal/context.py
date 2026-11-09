"""Per-app state shared by the routers, plus the common FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any

import httpx2
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .db import AuthSession, User, utcnow
from .limits import Calendar, Window
from .mailer import Mailer
from .security import Keyring, client_ip, sha256

SESSION_COOKIE = "__Host-vgm_session"


@dataclass
class AppContext:
    settings: Settings
    sessions: sessionmaker[Any]
    keyring: Keyring
    mailer: Mailer
    calendar: Calendar
    http: httpx2.Client  # outbound, sync (Resend)
    ahttp: httpx2.AsyncClient  # outbound, async (internal services)
    # Test seam: every outbound request (OAuth, Resend, internal services, MCP) goes through it.
    transport: Any = None
    # Agents runs in flight (single process): user ids and total.
    active_runs: set[str] = field(default_factory=set)
    background: set[Any] = field(default_factory=set)

    def window(self, bucket: str, limit: int, seconds: int | None = None) -> Window:
        return Window(bucket, seconds or self.settings.lockout_window_minutes * 60, limit)

    @property
    def lock_account(self) -> Window:
        return self.window("fail_acct", self.settings.lockout_max_per_account)

    @property
    def lock_ip(self) -> Window:
        return self.window("fail_ip", self.settings.lockout_max_per_ip)

    @property
    def session_ttl(self) -> timedelta:
        return timedelta(hours=self.settings.session_ttl_hours)

    @property
    def pending_ttl(self) -> timedelta:
        return timedelta(minutes=self.settings.pending_mfa_ttl_minutes)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx  # type: ignore[no-any-return]


Ctx = Annotated[AppContext, Depends(get_ctx)]


def get_db(ctx: Ctx) -> Iterator[Session]:
    with ctx.sessions() as session:
        yield session


DB = Annotated[Session, Depends(get_db)]


def get_ip(request: Request, ctx: Ctx) -> str:
    peer = request.client.host if request.client else None
    return client_ip(peer, request.headers.get("x-forwarded-for"), ctx.settings.proxies)


IP = Annotated[str, Depends(get_ip)]


@dataclass
class Current:
    user: User
    session: AuthSession


def get_current(request: Request, db: DB) -> Current | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token or len(token) > 100:
        return None
    row = db.get(AuthSession, sha256(token))
    if row is None or row.expires_at <= utcnow():
        return None
    user = db.scalar(select(User).where(User.id == row.user_id))
    return Current(user, row) if user else None


MaybeCurrent = Annotated[Current | None, Depends(get_current)]


def require_any_session(current: MaybeCurrent) -> Current:
    if current is None:
        raise HTTPException(401, "login_required")
    return current


def require_user(current: MaybeCurrent) -> Current:
    """A fully authenticated session (OAuth, or email + TOTP)."""
    if current is None or not current.session.mfa_ok:
        raise HTTPException(401, "login_required")
    return current


AnySession = Annotated[Current, Depends(require_any_session)]
LoggedIn = Annotated[Current, Depends(require_user)]
