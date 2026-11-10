"""Settings, all from the environment (see .env.example)."""

from __future__ import annotations

import ipaddress
from functools import cached_property

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    database_url: str = "sqlite:///./portal.db"
    # Used to HMAC short codes and to encrypt TOTP secrets at rest. Rotating it invalidates both.
    secret_key: str = Field(min_length=32)

    # Public origin of the site; the API lives under the same origin at /api.
    site_url: str = "https://vgmelo.tech"
    # Origins accepted on non-GET requests (CSRF). Defaults to site_url.
    allowed_origins: str = ""
    # CIDRs of the reverse proxy. X-Forwarded-For is only read when the peer is in this list.
    trusted_proxies: str = ""

    session_ttl_hours: int = Field(default=24 * 7, ge=1)
    pending_mfa_ttl_minutes: int = Field(default=15, ge=1)

    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    resend_api_key: str = ""
    email_from: str = "vgmelo.tech <no-reply@vgmelo.tech>"
    contact_to: str = ""  # who gets contact messages and budget alerts; empty: only stored
    admin_emails: str = ""

    email_code_ttl_minutes: int = Field(default=15, ge=1)
    totp_issuer: str = "vgmelo.tech"

    # Lockout after repeated password / TOTP / code failures (fixed windows).
    lockout_window_minutes: int = Field(default=15, ge=1)
    lockout_max_per_account: int = Field(default=5, ge=1)
    lockout_max_per_ip: int = Field(default=20, ge=1)
    register_max_per_ip_hour: int = Field(default=5, ge=1)
    register_max_per_email_hour: int = Field(default=3, ge=1)
    login_max_per_ip_window: int = Field(default=30, ge=1)
    oauth_start_max_per_ip_window: int = Field(default=30, ge=1)
    contact_max_per_ip_hour: int = Field(default=5, ge=1)
    contact_min_fill_seconds: int = Field(default=3, ge=0)

    # Per user, per day (QUOTA_TZ), per demo.
    quota_tz: str = "America/Sao_Paulo"
    quota_agents: int = Field(default=5, ge=0)
    quota_score: int = Field(default=30, ge=0)
    quota_mcp: int = Field(default=30, ge=0)
    quota_traces: int = Field(default=200, ge=0)
    # Sum of input+output tokens across every agents run, per day. 0 disables the demo.
    global_daily_tokens: int = Field(default=300_000, ge=0)
    max_concurrent_runs: int = Field(default=2, ge=1)
    # E-mail to the owner when the day's tokens cross these percentages of the budget (empty disables).
    alert_to: str = ""  # defaults to contact_to
    alert_budget_percents: str = Field(default="80,100", pattern=r"^\s*(\d{1,3}\s*(,\s*\d{1,3}\s*)*)?$")

    max_body_bytes: int = Field(default=16 * 1024, ge=1024)
    # CSV of the credit batch demo (raw text/csv body). Caddy caps /api/* at 64 KB, so keep it below that.
    batch_max_bytes: int = Field(default=50 * 1024, ge=1024, le=63 * 1024)
    batch_validate_per_hour: int = Field(default=30, ge=1)  # validating is free, but not unlimited
    question_max_chars: int = Field(default=500, ge=10)

    # Microagente da demo de crédito: modelo pequeno do Groq, só plano grátis. Os tokens entram no mesmo
    # GLOBAL_DAILY_TOKENS da demo de agentes. Sem chave, o site usa só as regras dele.
    groq_api_key: str = ""
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    credit_agent_model: str = "llama-3.1-8b-instant"
    credit_agent_per_hour: int = Field(default=90, ge=1)  # por usuário; um perfil tem 18 perguntas

    orchestrator_url: str = "http://agent-orchestrator:8000"
    orchestrator_api_key: str = ""
    orchestrator_agent: str = ""
    orchestrator_timeout_s: float = 180.0
    lens_url: str = "http://genai-lens:8080"
    scoring_url: str = "http://scoring:8000"
    mcp_url: str = "http://mcp-brasil:8000/mcp"

    @field_validator("site_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @cached_property
    def origins(self) -> frozenset[str]:
        return frozenset(o.rstrip("/") for o in (_csv(self.allowed_origins) or [self.site_url]))

    @cached_property
    def proxies(self) -> tuple[Network, ...]:
        return tuple(ipaddress.ip_network(c, strict=False) for c in _csv(self.trusted_proxies))

    @cached_property
    def admins(self) -> frozenset[str]:
        return frozenset(e.lower() for e in _csv(self.admin_emails))

    @property
    def email_enabled(self) -> bool:
        return bool(self.resend_api_key)

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def github_enabled(self) -> bool:
        return bool(self.github_client_id and self.github_client_secret)

    def quota_for(self, demo: str) -> int:
        return int(getattr(self, f"quota_{demo}"))

    @property
    def budget_alert_percents(self) -> list[int]:
        return sorted({int(p) for p in self.alert_budget_percents.split(",") if p.strip()})
