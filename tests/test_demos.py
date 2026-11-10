from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from vgmelo_portal.db import DemoRun, TokenBudget

from .conftest import TRACE_ID, FakeNet, done_event, email_user, make_client, oauth_login

APPLICANT: dict[str, Any] = {
    "checking_status": "<0", "duration": 48, "credit_history": "existing paid", "purpose": "new car",
    "credit_amount": 87300, "savings_status": "<970", "employment": "<1", "installment_commitment": 4,
    "other_parties": "none", "residence_since": 1, "property_magnitude": "no known property", "age": 22,
    "other_payment_plans": "none", "housing": "rent", "existing_credits": 1, "job": "skilled",
    "num_dependents": 1, "own_telephone": "none",
}  # fmt: skip


def events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        out.append((name, json.loads(data)))
    return out


def ask(client: TestClient, question: str = "Qual a Selic hoje?") -> httpx2.Response:
    return client.post("/api/demo/agents/stream", json={"question": question})


def midnight_sp() -> str:
    tz = ZoneInfo("America/Sao_Paulo")
    tomorrow = datetime.now(tz).date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), tz).isoformat(timespec="seconds")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/demo/agents/stream", {"question": "oi"}),
        ("GET", f"/api/demo/traces/{TRACE_ID}", None),
        ("POST", "/api/demo/score", APPLICANT),
        ("POST", "/api/demo/mcp/validar_documento", {"args": {"documento": "52998224725"}}),
    ],
)
def test_demos_need_login(client: TestClient, net: FakeNet, method: str, path: str, body: Any) -> None:
    r = client.request(method, path, json=body)
    assert r.status_code == 401 and r.json() == {"detail": "login_required"}
    assert not [r for r in net.requests if r.url.host in {"agent-orchestrator", "genai-lens", "scoring", "mcp-brasil"}]


def test_agents_stream_relays_and_records_the_run(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    r = ask(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    got = events(r.text)
    assert [e for e, _ in got] == ["agent_start", "llm", "done"]
    assert got[-1][1]["run_id"] == "run_1" and got[-1][1]["trace_id"] == TRACE_ID

    upstream = net.to("agent-orchestrator", "/v1/runs/stream")[0]
    assert upstream.headers["x-api-key"] == "internal-orch-key"
    assert json.loads(upstream.content) == {"input": "Qual a Selic hoje?"}

    with app.state.ctx.sessions() as db:
        run = db.get(DemoRun, "run_1")
        assert run is not None and run.trace_id == TRACE_ID and run.tokens == 120
        assert db.scalar(select(TokenBudget.tokens)) == 120
    assert client.get("/api/me").json()["quotas"]["agents"] == {"limit": 5, "used": 1, "remaining": 4}


def test_agents_quota_per_day(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    for i in range(5):
        net.orchestrator_body = done_event(run_id=f"run_{i}", trace_id=None)
        assert ask(client).status_code == 200
    r = ask(client)
    assert r.status_code == 429
    assert r.json() == {"reason": "daily_quota", "retry_at": midnight_sp()}
    assert len(net.to("agent-orchestrator")) == 5


def test_quota_is_per_user(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    for i in range(5):
        net.orchestrator_body = done_event(run_id=f"run_{i}", trace_id=None)
        ask(client)
    assert ask(client).status_code == 429
    with make_client(app, ip="198.51.100.40") as other:
        oauth_login(other, "github")
        net.orchestrator_body = done_event(run_id="run_other", trace_id=None)
        assert ask(other).status_code == 200


@pytest.fixture
def quota_two(overrides: dict[str, Any]) -> None:
    overrides.update(quota_agents=2)


def test_quota_is_configurable(quota_two: None, client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    for i in range(2):
        net.orchestrator_body = done_event(run_id=f"r{i}", trace_id=None)
        assert ask(client).status_code == 200
    assert ask(client).status_code == 429


@pytest.fixture
def small_budget(overrides: dict[str, Any]) -> None:
    overrides.update(global_daily_tokens=200)


def test_global_token_budget(small_budget: None, client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="big", trace_id=None, tokens=(150, 60))
    assert ask(client).status_code == 200
    # Another visitor, quota untouched, still gets "come back tomorrow": the budget is global.
    with make_client(app, ip="198.51.100.41") as other:
        oauth_login(other, "github")
        r = ask(other)
        assert r.status_code == 429 and r.json() == {"reason": "global_budget", "retry_at": midnight_sp()}
        assert other.get("/api/me").json()["budget_exhausted"] is True
    assert len(net.to("agent-orchestrator")) == 1


def test_question_size_limit(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    assert ask(client, "x" * 501).status_code == 422
    assert ask(client, "   ").status_code == 422
    assert net.to("agent-orchestrator") == []
    assert client.get("/api/me").json()["quotas"]["agents"]["used"] == 0


def test_upstream_failure_refunds_quota(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    net.routes[("agent-orchestrator", "/v1/runs/stream")] = lambda req: httpx2.Response(500, json={})
    r = ask(client)
    assert r.status_code == 502 and r.json() == {"detail": "upstream_unavailable"}
    assert client.get("/api/me").json()["quotas"]["agents"]["used"] == 0


def test_upstream_unreachable_refunds_quota(client: TestClient, net: FakeNet) -> None:
    def down(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=req)

    oauth_login(client)
    net.routes[("agent-orchestrator", "/v1/runs/stream")] = down
    assert ask(client).status_code == 502
    assert client.get("/api/me").json()["quotas"]["agents"]["used"] == 0
    # The slot is released: the next try reaches the orchestrator again.
    net.routes[("agent-orchestrator", "/v1/runs/stream")] = lambda req: httpx2.Response(
        200, content=done_event(), headers={"content-type": "text/event-stream"}
    )
    assert ask(client).status_code == 200


def test_one_run_at_a_time_per_user(client: TestClient, app: FastAPI) -> None:
    oauth_login(client)
    me = client.get("/api/me").json()["user"]["id"]
    app.state.ctx.active_runs.add(me)
    r = ask(client)
    assert r.status_code == 429 and r.json()["reason"] == "busy"
    assert client.get("/api/me").json()["quotas"]["agents"]["used"] == 0


def test_stream_without_done_still_releases(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    net.orchestrator_body = b'event: llm\r\ndata: {"text": "partial"}\r\n\r\n'
    r = ask(client)
    assert r.status_code == 200 and [e for e, _ in events(r.text)] == ["llm"]
    assert app.state.ctx.active_runs == set()


def test_trace_belongs_to_the_user(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    net.traces[TRACE_ID] = [{"span_id": "s1", "kind": "llm", "input_tokens": 100, "output_tokens": 20}]
    oauth_login(client)
    ask(client)
    r = client.get(f"/api/demo/traces/{TRACE_ID}")
    assert r.status_code == 200 and r.json()[0]["span_id"] == "s1"

    with make_client(app, ip="198.51.100.42") as other:
        oauth_login(other, "github")
        r = other.get(f"/api/demo/traces/{TRACE_ID}")
        assert r.status_code == 404 and r.json() == {"detail": "trace_not_found"}
    # Only the owner's request reached the lens.
    assert len(net.to("genai-lens")) == 1


def test_unknown_trace_is_404_without_calling_lens(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    assert client.get(f"/api/demo/traces/{'cd' * 16}").status_code == 404
    assert net.to("genai-lens") == []


@pytest.mark.parametrize("bad", ["..%2F..%2Fadmin", "ABCDEF" * 6, "ab" * 15, "zz" * 16])
def test_trace_id_format(client: TestClient, bad: str) -> None:
    oauth_login(client)
    assert client.get(f"/api/demo/traces/{bad}").status_code in {404, 422}


def test_trace_tokens_reconcile_with_lens(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    net.orchestrator_body = done_event(tokens=(100, 20))
    ask(client)
    net.traces[TRACE_ID] = [
        {"span_id": "a", "kind": "chain", "input_tokens": None},
        {"span_id": "b", "kind": "llm", "input_tokens": 150, "output_tokens": 30},
        {"span_id": "c", "kind": "llm", "input_tokens": 10, "output_tokens": 10},
    ]
    client.get(f"/api/demo/traces/{TRACE_ID}")
    client.get(f"/api/demo/traces/{TRACE_ID}")  # second read charges nothing more
    with app.state.ctx.sessions() as db:
        assert db.scalar(select(TokenBudget.tokens)) == 200
        run = db.get(DemoRun, "run_1")
        assert run is not None and run.tokens == 200


def test_score_forwards_one_applicant(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    r = client.post("/api/demo/score", json=APPLICANT)
    assert r.status_code == 200
    assert r.json() == {"model_version": "3", "prediction": {"default_probability": 0.71, "decision": "review"}}
    assert json.loads(net.to("scoring", "/predict")[0].content) == {"applicants": [APPLICANT]}


@pytest.mark.parametrize(
    "patch",
    [{"age": 12}, {"checking_status": "rich"}, {"purpose": "x" * 65}, {"extra": 1}, {"duration": "48; DROP"}],
)
def test_score_validates_like_the_model(client: TestClient, net: FakeNet, patch: dict[str, Any]) -> None:
    oauth_login(client)
    assert client.post("/api/demo/score", json={**APPLICANT, **patch}).status_code == 422
    assert net.to("scoring") == []


def test_score_quota(client: TestClient) -> None:
    oauth_login(client)
    for _ in range(30):
        assert client.post("/api/demo/score", json=APPLICANT).status_code == 200
    r = client.post("/api/demo/score", json=APPLICANT)
    assert r.status_code == 429 and r.json()["reason"] == "daily_quota"


def test_email_user_without_full_mfa_cannot_use_demos(client: TestClient, net: FakeNet) -> None:
    from .conftest import register

    register(client, net, "half@example.com")
    assert client.post("/api/demo/score", json=APPLICANT).status_code == 401
    email_user(client, net, "full@example.com")
    assert client.post("/api/demo/score", json=APPLICANT).status_code == 200


def test_mcp_allowed_tool(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    r = client.post("/api/demo/mcp/validar_documento", json={"args": {"documento": "52998224725"}})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "tool": "validar_documento",
        "result": {"tipo": "CPF", "valido": True, "formatado": "52998224725"},
    }
    assert net.to("mcp-brasil")


def test_mcp_tool_error_is_422(client: TestClient) -> None:
    oauth_login(client)
    r = client.post("/api/demo/mcp/consultar_cep", json={"args": {"cep": "99999-999"}})
    assert r.status_code == 422 and r.json()["detail"]["reason"] == "tool_error"


@pytest.mark.parametrize(
    ("tool", "args", "status"),
    [
        ("delete_everything", {}, 404),
        ("validar_documento", {"documento": "52998224725", "x": 1}, 422),
        ("validar_documento", {}, 422),
        ("consultar_cep", {"cep": "../../etc"}, 422),
        ("indicador_economico", {"indicador": "bitcoin"}, 422),
        ("indicador_economico", {"indicador": "ipca", "ultimos": 10_000}, 422),
    ],
)
def test_mcp_closed_list(client: TestClient, net: FakeNet, tool: str, args: dict[str, Any], status: int) -> None:
    oauth_login(client)
    assert client.post(f"/api/demo/mcp/{tool}", json={"args": args}).status_code == status
    assert net.to("mcp-brasil") == []
    assert client.get("/api/me").json()["quotas"]["mcp"]["used"] == 0


def test_mcp_down_refunds(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)

    async def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    net._mcp = httpx2.MockTransport(down)  # type: ignore[assignment]
    r = client.post("/api/demo/mcp/validar_documento", json={"args": {"documento": "52998224725"}})
    assert r.status_code == 502
    assert client.get("/api/me").json()["quotas"]["mcp"]["used"] == 0


def budget_mails(net: FakeNet) -> list[dict[str, Any]]:
    return [m for m in net.emails() if "teto de tokens" in m["subject"]]


@pytest.fixture
def alert_budget(overrides: dict[str, Any]) -> None:
    overrides.update(global_daily_tokens=1000, quota_agents=10, contact_to="dono@example.com")


def test_budget_alert_at_80_and_100_once_each(alert_budget: None, client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="a", trace_id=None, tokens=(500, 200))  # 70%: nada
    assert ask(client).status_code == 200
    assert budget_mails(net) == []

    net.orchestrator_body = done_event(run_id="b", trace_id=None, tokens=(100, 50))  # 85%
    assert ask(client).status_code == 200
    net.orchestrator_body = done_event(run_id="c", trace_id=None, tokens=(30, 10))  # 89%: já avisou dos 80
    assert ask(client).status_code == 200
    mails = budget_mails(net)
    assert len(mails) == 1 and mails[0]["to"] == ["dono@example.com"]
    assert "80%" in mails[0]["subject"] and "890" not in mails[0]["text"] and "850 de 1,000" in mails[0]["text"]

    net.orchestrator_body = done_event(run_id="d", trace_id=None, tokens=(100, 50))  # 104%
    assert ask(client).status_code == 200
    mails = budget_mails(net)
    assert len(mails) == 2 and "acabou" in mails[1]["subject"]
    assert ask(client).status_code == 429  # e não manda de novo
    assert len(budget_mails(net)) == 2


def test_budget_alert_jumps_straight_to_100(alert_budget: None, client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="big", trace_id=None, tokens=(900, 300))
    assert ask(client).status_code == 200
    subjects = [m["subject"] for m in budget_mails(net)]
    assert len(subjects) == 2 and "80%" in subjects[0] and "acabou" in subjects[1]


def test_budget_alert_after_lens_reconcile(alert_budget: None, client: TestClient, net: FakeNet) -> None:
    # O run disse 100 tokens; o lens mostra 850: a diferença também conta para o aviso.
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="r", tokens=(80, 20))
    assert ask(client).status_code == 200
    net.traces[TRACE_ID] = [{"kind": "llm", "input_tokens": 700, "output_tokens": 150}]
    assert client.get(f"/api/demo/traces/{TRACE_ID}").status_code == 200
    assert len(budget_mails(net)) == 1


@pytest.mark.parametrize("percents, mails", [("", 0), ("100", 1)])
def test_budget_alert_thresholds_are_configurable(
    overrides: dict[str, Any], percents: str, mails: int, request: pytest.FixtureRequest
) -> None:
    overrides.update(global_daily_tokens=100, alert_budget_percents=percents)
    client, net = request.getfixturevalue("client"), request.getfixturevalue("net")
    oauth_login(client)
    assert ask(client).status_code == 200
    assert len(budget_mails(net)) == mails
