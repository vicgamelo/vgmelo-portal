from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware datetimes on both Postgres and SQLite (which drops the offset)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime")
        return value.astimezone(UTC) if value is not None else None

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # google | github | email. Accounts are never linked by email across providers (pre-hijacking).
    provider: Mapped[str] = mapped_column(String(10))
    subject: Mapped[str] = mapped_column(String(320))
    email: Mapped[str | None] = mapped_column(String(320))
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    name: Mapped[str | None] = mapped_column(String(200))
    password_hash: Mapped[str | None] = mapped_column(String(200))
    totp_secret: Mapped[str | None] = mapped_column(String(200))  # encrypted
    totp_pending: Mapped[str | None] = mapped_column(String(200))  # encrypted, during setup
    totp_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    totp_last_step: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    __table_args__ = (UniqueConstraint("provider", "subject", name="uq_users_provider_subject"),)


class RecoveryCode(Base):
    __tablename__ = "recovery_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class PendingRegistration(Base):
    """One row per register call. The code only works from the browser that registered (reg cookie),
    so a code someone else triggered for your address can't set their password on your account."""

    __tablename__ = "pending_registrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    browser_hash: Mapped[str] = mapped_column(String(64), index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    code_hash: Mapped[str] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)


class AuthSession(Base):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # False while an email account still has to finish TOTP setup.
    mfa_ok: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)


class OAuthState(Base):
    __tablename__ = "oauth_states"

    state_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(10))
    code_verifier: Mapped[str] = mapped_column(String(128))
    next_path: Mapped[str] = mapped_column(String(500))
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)


class RateCounter(Base):
    """Fixed-window counters for rate limits and lockouts. ``key`` is already hashed."""

    __tablename__ = "rate_counters"

    bucket: Mapped[str] = mapped_column(String(40), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    window_start: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class Usage(Base):
    __tablename__ = "usage"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    demo: Mapped[str] = mapped_column(String(20), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class TokenBudget(Base):
    __tablename__ = "token_budget"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    tokens: Mapped[int] = mapped_column(BigInteger, default=0)


class UserTokens(Base):
    """Tokens each user spent on the LLM demos that day (agents, studio, credit micro-agent)."""

    __tablename__ = "user_tokens"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    tokens: Mapped[int] = mapped_column(BigInteger, default=0)


class BudgetAlert(Base):
    """One row per (day, threshold) already e-mailed: the primary key makes the alert go out once."""

    __tablename__ = "budget_alerts"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    percent: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class DemoRun(Base):
    __tablename__ = "demo_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[str] = mapped_column(String(10))
    tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class StudioVersion(Base):
    """An agent saved in the studio. Versions are never edited: saving again makes the next number."""

    __tablename__ = "studio_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(60))
    version: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(String(200))
    spec: Mapped[str] = mapped_column(Text)  # JSON: instructions, model, tools, temperature, max_steps
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "name", "version", name="uq_studio_user_name_version"),)


class ContactMessage(Base):
    __tablename__ = "contact_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(320))
    kind: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    notified: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_contact_created", "created_at"),)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    if url in {"sqlite://", "sqlite:///:memory:"}:
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_sessionmaker(engine: Engine) -> sessionmaker[Any]:
    return sessionmaker(engine, expire_on_commit=False)


def affected(result: Any) -> int:
    """Rows touched by an UPDATE/DELETE (``Result`` is typed without ``rowcount``)."""
    return int(result.rowcount or 0)
