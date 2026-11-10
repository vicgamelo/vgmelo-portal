from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vgmelo_portal.db import DemoRun

from .conftest import TRACE_ID, FakeNet, done_event, make_client, oauth_login

AGENT: dict[str, Any] = {
    "instructions": "Você é um economista. Responda em português.",
    "model": "gpt-oss-20b",
    "tools": ["brasil__indicador_economico", "calculator"],
    "temperature": 0.3,
    "max_steps": 4,
}

CATALOG = {
    "models": [{"id": "gpt-oss-20b", "label": "GPT-OSS 20B · Groq", "provider": "groq", "available": True}],
    "tools": [{"name": "calculator", "server": None, "description": "math"}],
    "templates": [{"name": "economist", "instructions": "e", "model": "gpt-oss-20b", "tools": ["calculator"]}],
    "limits": {"max_steps": 6, "max_tokens": 2048, "instructions_max_chars": 4000},
    "internal": "not for visitors",
}


@pytest.fixture
def studio_net(net: FakeNet) -> FakeNet:
    net.routes[("agent-orchestrator", "/v1/studio")] = lambda req: httpx2.Response(200, json=CATALOG)
    net.routes[("agent-orchestrator", "/v1/studio/runs/stream")] = lambda req: httpx2.Response(
        200, content=net.orchestrator_body, headers={"content-type": "text/event-stream"}
    )
    return net


def save(client: TestClient, name: str = "Economista", note: str | None = None, **agent: Any) -> httpx2.Response:
    return client.post("/api/studio/versions", json={"name": name, "note": note, "agent": AGENT | agent})


def run(client: TestClient, question: str = "Qual a Selic?", **agent: Any) -> httpx2.Response:
    return client.post("/api/studio/run/stream", json={"question": question, "agent": AGENT | agent})


def studio_used(client: TestClient) -> int:
    return int(client.get("/api/me").json()["quotas"]["studio"]["used"])


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/studio/catalog", None),
        ("GET", "/api/studio/versions", None),
        ("POST", "/api/studio/versions", {"name": "x", "agent": AGENT}),
        ("DELETE", "/api/studio/versions/0123456789abcdef", None),
        ("POST", "/api/studio/run/stream", {"question": "oi", "agent": AGENT}),
    ],
)
def test_studio_needs_login(client: TestClient, studio_net: FakeNet, method: str, path: str, body: Any) -> None:
    r = client.request(method, path, json=body)
    assert r.status_code == 401 and r.json() == {"detail": "login_required"}
    assert not studio_net.to("agent-orchestrator")


def test_catalog_comes_from_the_orchestrator(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    r = client.get("/api/studio/catalog")
    assert r.status_code == 200
    assert r.json() == {k: CATALOG[k] for k in ("models", "tools", "templates", "limits")}
    assert studio_net.to("agent-orchestrator", "/v1/studio")[0].headers["x-api-key"] == "internal-orch-key"
    studio_net.routes[("agent-orchestrator", "/v1/studio")] = lambda req: httpx2.Response(404, json={})
    assert client.get("/api/studio/catalog").status_code == 502


def test_versions_are_numbered_per_agent_name(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    first = save(client, note="primeira")
    assert first.status_code == 201
    v1 = first.json()
    assert v1["version"] == 1 and v1["name"] == "Economista" and v1["note"] == "primeira"
    assert v1["agent"] == AGENT
    assert save(client, temperature=0.9).json()["version"] == 2
    assert save(client, name="Repórter", tools=[]).json()["version"] == 1
    # spaces around the name don't make another agent
    assert save(client, name="  Economista ").json()["version"] == 3

    listed = client.get("/api/studio/versions").json()
    assert [(v["name"], v["version"]) for v in listed] == [
        ("Economista", 3),
        ("Economista", 2),
        ("Economista", 1),
        ("Repórter", 1),
    ]
    assert listed[1]["agent"]["temperature"] == 0.9
    assert listed[2]["note"] == "primeira" and listed[0]["note"] is None
    # saving is free: no quota, no orchestrator
    assert studio_used(client) == 0
    assert not studio_net.to("agent-orchestrator")


def test_defaults_are_filled_in(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    r = client.post("/api/studio/versions", json={"name": "Mínimo", "agent": {"instructions": "oi", "model": "m"}})
    assert r.status_code == 201
    assert r.json()["agent"] == {"instructions": "oi", "model": "m", "tools": [], "temperature": 0.2, "max_steps": 4}


def test_versions_are_private(client: TestClient, studio_net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    mine = save(client).json()
    with make_client(app, ip="198.51.100.41") as other:
        oauth_login(other, "github")
        assert other.get("/api/studio/versions").json() == []
        assert other.delete(f"/api/studio/versions/{mine['id']}").status_code == 404
        theirs = save(other).json()
        assert theirs["version"] == 1  # numbering is per user too
    assert [v["id"] for v in client.get("/api/studio/versions").json()] == [mine["id"]]
    assert client.delete(f"/api/studio/versions/{theirs['id']}").status_code == 404
    assert client.delete(f"/api/studio/versions/{mine['id']}").status_code == 204
    assert client.get("/api/studio/versions").json() == []
    assert client.delete(f"/api/studio/versions/{mine['id']}").status_code == 404
    assert client.delete("/api/studio/versions/../../users").status_code == 404


@pytest.mark.parametrize(
    "agent",
    [
        {"extra": "field"},
        {"instructions": ""},
        {"instructions": "x" * 4001},
        {"model": "openai/gpt-oss-120b"},  # an id from the catalog, not a provider path
        {"model": "Big Model"},
        {"tools": ["http://evil.example/mcp"]},
        {"tools": ["Calculator"]},
        {"tools": ["calculator"] * 31},
        {"tools": "calculator"},
        {"temperature": 2},
        {"max_steps": 0},
        {"max_steps": 11},
    ],
)
def test_bad_agents_are_refused(client: TestClient, studio_net: FakeNet, agent: dict[str, Any]) -> None:
    oauth_login(client)
    assert save(client, **agent).status_code == 422
    assert run(client, **agent).status_code == 422
    assert client.get("/api/studio/versions").json() == []
    assert not studio_net.to("agent-orchestrator")
    assert studio_used(client) == 0


@pytest.mark.parametrize("body", [{"name": "", "agent": AGENT}, {"name": "x" * 61, "agent": AGENT},
                                  {"name": "x", "note": "n" * 201, "agent": AGENT}, {"name": "x"}])  # fmt: skip
def test_bad_version_fields(client: TestClient, studio_net: FakeNet, body: dict[str, Any]) -> None:
    oauth_login(client)
    assert client.post("/api/studio/versions", json=body).status_code == 422


@pytest.fixture
def two_versions(overrides: dict[str, Any]) -> None:
    overrides.update(studio_max_versions=2)


@pytest.fixture
def three_saves(overrides: dict[str, Any]) -> None:
    overrides.update(studio_saves_per_hour=3)


@pytest.fixture
def small_budget(overrides: dict[str, Any]) -> None:
    overrides.update(global_daily_tokens=200)


def test_version_cap_per_user(two_versions: None, client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    assert save(client).status_code == 201
    assert save(client, name="Outro").status_code == 201
    r = save(client)
    assert r.status_code == 409 and r.json() == {"detail": "too_many_versions"}
    # deleting one frees a slot
    first = client.get("/api/studio/versions").json()[0]["id"]
    assert client.delete(f"/api/studio/versions/{first}").status_code == 204
    assert save(client).status_code == 201


def test_saves_are_rate_limited(three_saves: None, client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    for _ in range(3):
        assert save(client).status_code == 201
    r = save(client)
    assert r.status_code == 429 and r.json()["reason"] == "rate_limited"


def test_run_relays_the_stream_and_spends_the_studio_quota(
    client: TestClient, studio_net: FakeNet, app: FastAPI
) -> None:
    oauth_login(client)
    r = run(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in r.text

    sent = studio_net.to("agent-orchestrator", "/v1/studio/runs/stream")[0]
    assert sent.headers["x-api-key"] == "internal-orch-key"
    assert json.loads(sent.content) == {"input": "Qual a Selic?", "agent": AGENT}

    quotas = client.get("/api/me").json()["quotas"]
    assert quotas["studio"] == {"limit": 5, "used": 1, "remaining": 4}
    assert quotas["agents"]["used"] == 0  # separate from the team demo
    with app.state.ctx.sessions() as db:
        assert db.get(DemoRun, "run_1") is not None
    # the run is the visitor's, so its trace is too
    studio_net.traces[TRACE_ID] = [{"name": "run.studio", "kind": "chain"}]
    assert client.get(f"/api/demo/traces/{TRACE_ID}").status_code == 200


def test_studio_quota_per_day(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    for i in range(5):
        studio_net.orchestrator_body = done_event(run_id=f"run_{i}", trace_id=None)
        assert run(client).status_code == 200
    r = run(client)
    assert r.status_code == 429 and r.json()["reason"] == "daily_quota"
    assert len(studio_net.to("agent-orchestrator", "/v1/studio/runs/stream")) == 5


def test_agent_the_orchestrator_refuses(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    studio_net.routes[("agent-orchestrator", "/v1/studio/runs/stream")] = lambda req: httpx2.Response(
        422, json={"detail": "tools not available: ['normas__buscar_normas']"}
    )
    r = run(client, tools=["normas__buscar_normas"])
    assert r.status_code == 422
    assert r.json() == {
        "detail": {"reason": "invalid_agent", "message": "tools not available: ['normas__buscar_normas']"}
    }
    assert studio_used(client) == 0  # refunded
    # a 422 with a body that isn't the usual one still answers invalid_agent
    studio_net.routes[("agent-orchestrator", "/v1/studio/runs/stream")] = lambda req: httpx2.Response(
        422, json={"detail": [{"loc": ["body"], "msg": "bad"}]}
    )
    assert run(client).json() == {"detail": {"reason": "invalid_agent", "message": ""}}
    studio_net.routes[("agent-orchestrator", "/v1/studio/runs/stream")] = lambda req: httpx2.Response(500, json={})
    assert run(client).status_code == 502
    assert studio_used(client) == 0


def test_team_demo_still_treats_an_orchestrator_422_as_upstream_error(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    net.routes[("agent-orchestrator", "/v1/runs/stream")] = lambda req: httpx2.Response(422, json={"detail": "x"})
    r = client.post("/api/demo/agents/stream", json={"question": "oi"})
    assert r.status_code == 502 and r.json() == {"detail": "upstream_unavailable"}


def test_run_question_limits(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    assert run(client, question="x" * 501).status_code == 422
    assert run(client, question="  ").status_code == 422
    assert not studio_net.to("agent-orchestrator")


def test_studio_and_team_share_one_run_at_a_time(client: TestClient, studio_net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    me = client.get("/api/me").json()["user"]["id"]
    app.state.ctx.active_runs.add(me)
    r = run(client)
    assert r.status_code == 429 and r.json()["reason"] == "busy"
    assert studio_used(client) == 0


def test_studio_counts_toward_the_global_budget(small_budget: None, client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    studio_net.orchestrator_body = done_event(run_id="big", trace_id=None, tokens=(150, 60))
    assert run(client).status_code == 200
    r = run(client)
    assert r.status_code == 429 and r.json()["reason"] == "global_budget"
    # and the team demo stops too: same free LLM key
    assert client.post("/api/demo/agents/stream", json={"question": "oi"}).json()["reason"] == "global_budget"
