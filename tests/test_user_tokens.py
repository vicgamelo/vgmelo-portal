"""Teto diário de tokens por usuário: agentes, estúdio e o microagente de crédito somam no mesmo número."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .conftest import TRACE_ID, FakeNet, done_event, make_client, oauth_login
from .test_demos import ask, midnight_sp
from .test_studio import run

GROQ = ("api.groq.com", "/openai/v1/chat/completions")


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {"user_daily_tokens": 200, "groq_api_key": "gsk-test"}


@pytest.fixture
def studio_net(net: FakeNet) -> FakeNet:
    net.routes[("agent-orchestrator", "/v1/studio/runs/stream")] = lambda req: httpx2.Response(
        200, content=net.orchestrator_body, headers={"content-type": "text/event-stream"}
    )
    return net


def tokens(client: TestClient) -> dict[str, int]:
    return dict(client.get("/api/me").json()["tokens"])


def credit(client: TestClient, net: FakeNet, spent: int) -> httpx2.Response:
    body = {"choices": [{"message": {"content": json.dumps({"value": "new car"})}}], "usage": {"total_tokens": spent}}
    net.routes[GROQ] = lambda req: httpx2.Response(200, json=body)
    return client.post("/api/demo/score/agent", json={"field": "purpose", "answer": "um carro"})


def test_me_shows_the_users_tokens(client: TestClient, net: FakeNet) -> None:
    assert client.get("/api/me").json()["tokens"] == {"limit": 200, "used": 0, "remaining": 200}
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="r1", trace_id=None, tokens=(90, 30))
    assert ask(client).status_code == 200
    assert tokens(client) == {"limit": 200, "used": 120, "remaining": 80}


def test_agents_stop_when_the_users_tokens_run_out(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="big", trace_id=None, tokens=(150, 60))
    assert ask(client).status_code == 200  # começou com saldo: termina, mesmo passando do teto
    r = ask(client)
    assert r.status_code == 429 and r.json() == {"reason": "user_tokens", "retry_at": midnight_sp()}
    assert client.get("/api/me").json()["quotas"]["agents"]["used"] == 1  # a recusa não gasta a cota
    assert tokens(client)["remaining"] == 0
    # o teto é de cada um: outra pessoa continua rodando
    with make_client(app, ip="198.51.100.41") as other:
        oauth_login(other, "github")
        net.orchestrator_body = done_event(run_id="other", trace_id=None)
        assert ask(other).status_code == 200
        assert tokens(other)["used"] == 120
    assert len(net.to("agent-orchestrator")) == 2


def test_studio_and_agents_share_the_users_tokens(client: TestClient, studio_net: FakeNet) -> None:
    oauth_login(client)
    studio_net.orchestrator_body = done_event(run_id="s1", trace_id=None, tokens=(150, 60))
    assert run(client).status_code == 200
    assert run(client).json()["reason"] == "user_tokens"
    assert ask(client).json()["reason"] == "user_tokens"
    assert client.get("/api/me").json()["quotas"]["studio"]["used"] == 1


def test_credit_agent_counts_and_stops_too(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    assert credit(client, net, 150).status_code == 200
    assert tokens(client)["used"] == 150
    assert credit(client, net, 100).status_code == 200
    r = credit(client, net, 100)
    assert r.status_code == 429 and r.json()["reason"] == "user_tokens"
    assert len(net.to(*GROQ)) == 2


def test_lens_reconcile_charges_the_user(client: TestClient, net: FakeNet) -> None:
    # O run disse 100 tokens; o lens mostra 250: a diferença também sai do saldo de quem rodou.
    oauth_login(client)
    net.orchestrator_body = done_event(run_id="r", tokens=(80, 20))
    assert ask(client).status_code == 200
    net.traces[TRACE_ID] = [{"kind": "llm", "input_tokens": 200, "output_tokens": 50}]
    assert client.get(f"/api/demo/traces/{TRACE_ID}").status_code == 200
    assert tokens(client)["used"] == 250
    assert ask(client).json()["reason"] == "user_tokens"


@pytest.mark.parametrize("overrides", [{"user_daily_tokens": 0}])
def test_zero_turns_the_llm_demos_off(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    assert ask(client).json()["reason"] == "user_tokens"
    assert not net.to("agent-orchestrator")
