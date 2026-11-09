"""Shared fixtures. Every outbound call (Google, GitHub, Resend, orchestrator, lens, scoring, MCP) goes
through ``FakeNet``, an httpx2 transport: the app code runs unchanged down to the HTTP request."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2
import pyotp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import create_engine

from vgmelo_portal.app import create_app
from vgmelo_portal.config import Settings
from vgmelo_portal.db import Base

SITE = "https://vgmelo.tech"
ORIGIN = {"Origin": SITE}
PASSWORD = "correct horse battery"
TRACE_ID = "ab" * 16

Handler = Callable[[httpx2.Request], httpx2.Response]


def sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    return "".join(f"event: {e}\r\ndata: {json.dumps(d)}\r\n\r\n" for e, d in events).encode()


def done_event(run_id: str = "run_1", trace_id: str | None = TRACE_ID, tokens: tuple[int, int] = (100, 20)) -> bytes:
    return sse(
        ("agent_start", {"type": "agent_start", "agent": "supervisor", "depth": 0, "data": {"task": "q"}}),
        ("llm", {"type": "llm", "agent": "supervisor", "depth": 0, "data": {"text": "hi"}}),
        (
            "done",
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "status": "completed",
                "output": "answer",
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
            },
        ),
    )


def build_fake_mcp() -> MCPServer:
    mcp = MCPServer(name="fake-brasil")

    @mcp.tool()
    def validar_documento(documento: str) -> dict[str, Any]:
        return {"tipo": "CPF", "valido": documento.startswith("529"), "formatado": documento}

    @mcp.tool()
    async def consultar_cep(cep: str) -> dict[str, Any]:
        if cep.startswith("99"):
            raise ToolError("Não encontrado: CEP 99999999")
        return {"cep": cep.replace("-", ""), "cidade": "São Paulo", "uf": "SP"}

    return mcp


class FakeNet(httpx2.BaseTransport, httpx2.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.routes: dict[tuple[str, str], Handler] = {}
        self.mcp_running = False
        self.fresh_mcp()
        self.orchestrator_body = done_event()
        self.google_user: dict[str, Any] = {
            "sub": "g-1",
            "email": "ana@gmail.com",
            "email_verified": True,
            "name": "Ana",
        }
        self.github_user: dict[str, Any] = {"id": 42, "login": "octo", "name": "Octo Cat"}
        self.github_emails: list[dict[str, Any]] = [
            {"email": "octo@users.noreply.github.com", "primary": False, "verified": True},
            {"email": "Octo@Example.com", "primary": True, "verified": True},
        ]
        self.traces: dict[str, list[dict[str, Any]]] = {}
        self._defaults()

    def fresh_mcp(self) -> None:
        """A Streamable HTTP session manager runs once; each TestClient lifespan gets a new one."""
        # Same mode as the real server; its DNS-rebinding guard must accept the internal Host.
        security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["mcp-brasil:8000"])
        self.mcp_app = build_fake_mcp().streamable_http_app(
            stateless_http=True, json_response=True, transport_security=security
        )
        self._mcp = httpx2.ASGITransport(app=self.mcp_app)

    # -- routes ---------------------------------------------------------------------------------

    def _defaults(self) -> None:
        r = self.routes
        r[("oauth2.googleapis.com", "/token")] = lambda req: httpx2.Response(
            200, json={"access_token": "g-access", "token_type": "Bearer", "expires_in": 3600}
        )
        r[("openidconnect.googleapis.com", "/v1/userinfo")] = lambda req: httpx2.Response(200, json=self.google_user)
        r[("github.com", "/login/oauth/access_token")] = lambda req: httpx2.Response(
            200, json={"access_token": "gh-access", "token_type": "bearer", "scope": "read:user,user:email"}
        )
        r[("api.github.com", "/user")] = lambda req: httpx2.Response(200, json=self.github_user)
        r[("api.github.com", "/user/emails")] = lambda req: httpx2.Response(200, json=self.github_emails)
        r[("api.resend.com", "/emails")] = lambda req: httpx2.Response(200, json={"id": "email_1"})
        r[("agent-orchestrator", "/v1/runs/stream")] = lambda req: httpx2.Response(
            200, content=self.orchestrator_body, headers={"content-type": "text/event-stream"}
        )
        r[("scoring", "/predict")] = lambda req: httpx2.Response(
            200,
            json={
                "model_version": "3",
                "predictions": [{"default_probability": 0.71, "decision": "review"}],
            },
        )

    def _route(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "genai-lens" and path.startswith("/api/traces/"):
            spans = self.traces.get(path.rsplit("/", 1)[1])
            return httpx2.Response(200, json=spans) if spans is not None else httpx2.Response(404, json={})
        handler = self.routes.get((host, path))
        if handler is None:
            return httpx2.Response(599, json={"error": f"no fake for {host}{path}"})
        return handler(request)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        return self._route(request)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "mcp-brasil":
            self.requests.append(request)
            return await self._mcp.handle_async_request(request)
        await request.aread()
        return self._route(request)

    # -- inspection -----------------------------------------------------------------------------

    def to(self, host: str, path: str | None = None) -> list[httpx2.Request]:
        return [r for r in self.requests if r.url.host == host and (path is None or r.url.path == path)]

    def emails(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.to("api.resend.com")]

    def last_code(self, to: str) -> str:
        for mail in reversed(self.emails()):
            if mail["to"] == [to] and (m := re.search(r"Seu código: (\d{6})", mail["text"])):
                return m.group(1)
        raise AssertionError(f"no code sent to {to}")


@pytest.fixture
def db_url() -> Iterator[str]:
    """SQLite in memory by default; set PORTAL_TEST_DATABASE_URL to run against Postgres."""
    url = os.getenv("PORTAL_TEST_DATABASE_URL", "sqlite://")
    if not url.startswith("sqlite"):
        engine = create_engine(url)
        Base.metadata.drop_all(engine)
        engine.dispose()
    yield url
    if not url.startswith("sqlite"):
        engine = create_engine(url)
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def net() -> FakeNet:
    return FakeNet()


@pytest.fixture
def overrides() -> dict[str, Any]:
    """Tests tweak settings by mutating this before ``client`` is built."""
    return {}


@pytest.fixture
def settings(db_url: str, overrides: dict[str, Any]) -> Settings:
    base: dict[str, Any] = {
        "database_url": db_url,
        "secret_key": "test-secret-key-that-is-long-enough-0123456789",
        "site_url": SITE,
        "google_client_id": "gid",
        "google_client_secret": "gsecret",
        "github_client_id": "ghid",
        "github_client_secret": "ghsecret",
        "resend_api_key": "re_test_key",
        "admin_emails": "ana@gmail.com",
        "orchestrator_api_key": "internal-orch-key",
        "contact_min_fill_seconds": 3,
        "contact_to": "dono@example.com",
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def app(settings: Settings, net: FakeNet) -> FastAPI:
    app = create_app(settings, transport=net)
    portal_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(a: Any) -> AsyncIterator[Any]:
        # The fake MCP server's session manager runs in the same event loop as the portal.
        async with self_lifespan(net), portal_lifespan(a) as state:
            yield state

    app.router.lifespan_context = lifespan
    return app


@asynccontextmanager
async def self_lifespan(net: FakeNet) -> AsyncIterator[None]:
    if net.mcp_running:  # a second TestClient on the same app (another browser)
        yield
        return
    net.mcp_running = True
    net.fresh_mcp()
    try:
        async with net.mcp_app.router.lifespan_context(net.mcp_app):
            yield
    finally:
        net.mcp_running = False


def make_client(app: FastAPI, ip: str = "203.0.113.7") -> TestClient:
    return TestClient(app, base_url="https://testserver", headers=ORIGIN, client=(ip, 50000), follow_redirects=False)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with make_client(app) as c:
        yield c


# ------------------------------------------------------------------------------------- flows


def oauth_login(client: TestClient, provider: str = "google", next_path: str = "/demos") -> httpx2.Response:
    start = client.get(f"/api/auth/{provider}/start", params={"next": next_path})
    assert start.status_code == 302, start.text
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    return client.get(f"/api/auth/{provider}/callback", params={"code": "auth-code", "state": state})


def pkce_ok(verifier: str, challenge: str) -> bool:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == challenge


def register(client: TestClient, net: FakeNet, email: str, password: str = PASSWORD) -> None:
    r = client.post("/api/auth/email/register", json={"email": email, "password": password})
    assert r.status_code == 202, r.text
    r = client.post("/api/auth/email/verify", json={"email": email, "code": net.last_code(email)})
    assert r.json() == {"status": "totp_setup_required"}, r.text


def setup_totp(client: TestClient) -> tuple[str, list[str]]:
    r = client.post("/api/auth/totp/setup")
    assert r.status_code == 200, r.text
    secret = parse_qs(urlsplit(r.json()["otpauth_uri"]).query)["secret"][0]
    r = client.post("/api/auth/totp/confirm", json={"code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200, r.text
    return secret, r.json()["recovery_codes"]


def email_user(client: TestClient, net: FakeNet, email: str = "bia@example.com") -> tuple[str, list[str]]:
    register(client, net, email)
    return setup_totp(client)


def code_at(secret: str, offset_steps: int = 0) -> str:
    return pyotp.TOTP(secret).at(time.time() + 30 * offset_steps)
