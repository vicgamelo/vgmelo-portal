"""A visitor who closes the tab mid-run must still be charged for the tokens the run used."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import httpx2
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message

from vgmelo_portal.context import SESSION_COOKIE
from vgmelo_portal.db import DemoRun, TokenBudget

from .conftest import SITE, FakeNet, oauth_login, sse


def test_run_is_recorded_after_client_disconnects(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    cookie = client.cookies[SESSION_COOKIE]
    ctx = app.state.ctx

    async def main() -> None:
        gate, gone = anyio.Event(), anyio.Event()

        async def slow_run() -> AsyncIterator[bytes]:
            yield sse(("agent_start", {"type": "agent_start"}))
            await gate.wait()  # the orchestrator keeps working after the visitor left
            yield sse(("done", {"run_id": "run_late", "trace_id": None, "input_tokens": 70, "output_tokens": 30}))

        net.routes[("agent-orchestrator", "/v1/runs/stream")] = lambda req: httpx2.Response(
            200, content=slow_run(), headers={"content-type": "text/event-stream"}
        )
        body = json.dumps({"question": "oi"}).encode()
        sent: list[Message] = []
        first = True

        async def receive() -> Message:
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                gone.set()  # got the first event, close the tab

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/api/demo/agents/stream",
            "raw_path": b"/api/demo/agents/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"origin", SITE.encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"cookie", f"{SESSION_COOKIE}={cookie}".encode()),
            ],
            "client": ("203.0.113.7", 50000),
            "server": ("testserver", 443),
        }
        with anyio.fail_after(5):
            await app(scope, receive, send)
            assert sent[0]["status"] == 200
            assert ctx.active_runs, "run still in flight after the visitor left"
            gate.set()
            await asyncio.gather(*ctx.background)

    anyio.run(main)
    with ctx.sessions() as db:
        run = db.get(DemoRun, "run_late")
        assert run is not None and run.tokens == 100
        assert db.get(TokenBudget, ctx.calendar.today()).tokens == 100  # type: ignore[union-attr]
    assert ctx.active_runs == set()
