"""Microagente da demo de crédito: o modelo pequeno do Groq lê a resposta e o servidor valida o que volta."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vgmelo_portal import credit_agent
from vgmelo_portal.limits import tokens_used

from .conftest import FakeNet, oauth_login

GROQ = ("api.groq.com", "/openai/v1/chat/completions")


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {"groq_api_key": "gsk-test", "quota_score": 5}


def groq_says(net: FakeNet, content: Any, tokens: int = 250, status: int = 200) -> None:
    body = {
        "choices": [{"message": {"content": content if isinstance(content, str) else json.dumps(content)}}],
        "usage": {"total_tokens": tokens},
    }
    net.routes[GROQ] = lambda req: httpx2.Response(status, json=body)


def agent(client: TestClient, field: str = "purpose", answer: str = "um carro", **extra: Any) -> httpx2.Response:
    return client.post("/api/demo/score/agent", json={"field": field, "answer": answer, **extra})


def sent(net: FakeNet) -> dict[str, Any]:
    reqs = net.to(*GROQ)
    assert reqs, "o Groq não foi chamado"
    body: dict[str, Any] = json.loads(reqs[-1].content)
    return body


def test_needs_login(client: TestClient, net: FakeNet) -> None:
    groq_says(net, {"value": "new car"})
    assert agent(client).status_code == 401
    assert net.to(*GROQ) == []


def test_picks_one_option_with_a_small_free_model(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    oauth_login(client)
    groq_says(net, {"value": "new car"}, tokens=310)
    r = agent(client, answer="um carro zero")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "value": "new car", "model": "openai/gpt-oss-20b"}
    body = sent(net)
    assert body["model"] == "openai/gpt-oss-20b" and body["temperature"] == 0 and body["max_tokens"] == 300
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}
    assert net.to(*GROQ)[-1].headers["authorization"] == "Bearer gsk-test"
    # A resposta vai como string JSON, separada das instruções.
    assert body["messages"][1] == {"role": "user", "content": '"um carro zero"'}
    # Tokens entram no teto global; a cota de score fica intacta.
    with app.state.ctx.sessions() as db:
        assert tokens_used(db, app.state.ctx.calendar.today()) == 310
    assert client.get("/api/me").json()["quotas"]["score"]["used"] == 0


def test_ambiguous_answer_asks_back_with_only_the_valid_candidates(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    groq_says(net, {"candidates": ["used car", "new car", "spaceship", "used car"]})
    assert agent(client).json()["status"] == "ask"
    assert agent(client).json()["options"] == ["new car", "used car"]  # ordem do domínio, sem lixo


@pytest.mark.parametrize(
    ("field", "content", "expected"),
    [
        ("purpose", {"value": "carro"}, {"status": "unknown"}),  # fora do domínio
        # O gpt-oss-20b respondeu a descrição em vez do valor para "trabalho lá faz 5 anos".
        ("employment", {"value": "4 to 7 years"}, {"status": "ok", "value": "4<=X<7"}),
        ("checking_status", {"value": " Has no checking account"}, {"status": "ok", "value": "no checking"}),
        ("purpose", {"value": None}, {"status": "unknown"}),
        ("purpose", {"candidates": ["used car", "nope"]}, {"status": "ok", "value": "used car"}),
        ("purpose", {"candidates": ["nope"]}, {"status": "unknown"}),
        ("housing", {"candidates": ["own", "rent", "for free"]}, {"status": "unknown"}),  # todas = não decidiu nada
        ("purpose", {"dont_know": True}, {"status": "dont_know"}),
        ("purpose", "isto não é JSON", {"status": "unknown"}),
        ("purpose", "[1, 2]", {"status": "unknown"}),
        ("duration", {"value": 24}, {"status": "ok", "value": 24}),
        ("duration", {"value": 24.0}, {"status": "ok", "value": 24}),
        ("duration", {"value": "36"}, {"status": "ok", "value": 36}),
        ("duration", {"value": 24.5}, {"status": "unknown"}),  # inteiro no modelo
        ("duration", {"value": 0}, {"status": "unknown"}),
        ("duration", {"value": True}, {"status": "unknown"}),
        ("age", {"value": "trinta"}, {"status": "unknown"}),
        ("age", {"value": 101}, {"status": "unknown"}),
        ("credit_amount", {"value": 90000}, {"status": "ok", "value": 90000}),
        ("credit_amount", {"value": 0}, {"status": "unknown"}),  # mínimo exclusivo
        ("credit_amount", {"value": 10_000_001}, {"status": "unknown"}),  # teto do scoring, em reais
        ("credit_amount", {"value": "NaN"}, {"status": "unknown"}),
    ],
)
def test_only_values_the_scoring_accepts_come_out(field: str, content: Any, expected: dict[str, Any]) -> None:
    raw = content if isinstance(content, str) else json.dumps(content)
    assert credit_agent.read(field, raw) == expected


def test_prompt_lists_the_brackets_in_reais() -> None:
    system = credit_agent.messages("savings_status", "uns 3 mil")[0]["content"]
    assert "- <970: less than R$ 970" in system
    assert "- >=9700: R$ 9,700 or more" in system
    assert "from R$ 0 to R$ 1,940" in credit_agent.messages("checking_status", "x")[0]["content"]
    amount = credit_agent.messages("credit_amount", "x")[0]["content"]
    assert "Brazilian reais" in amount and "from 0 to 10,000,000" in amount
    assert "DM" not in "".join(m["content"] for f in credit_agent.FIELDS for m in credit_agent.messages(f, "x"))


def test_contract_matches_the_scoring_domain() -> None:
    """Mesmas categorias de mlops-platform/src/mlops_platform/schema.py (faixas de dinheiro em reais)."""
    assert list(credit_agent.CATS["checking_status"].options) == ["<0", "0<=X<1940", ">=1940", "no checking"]
    assert list(credit_agent.CATS["savings_status"].options) == [
        "<970",
        "970<=X<4850",
        "4850<=X<9700",
        ">=9700",
        "no known savings",
    ]


def test_every_field_of_the_credit_contract_is_covered() -> None:
    assert sorted(credit_agent.FIELDS) == sorted(credit_agent.FieldName.__args__)
    assert len(credit_agent.FIELDS) == 18


def test_unknown_field_and_long_answers_are_refused(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    groq_says(net, {"value": "new car"})
    assert agent(client, field="personal_status").status_code == 422  # protegido, fora do modelo
    assert agent(client, answer="x" * 121).status_code == 422
    assert agent(client, brl_per_dm=10).status_code == 422  # a taxa saiu: tudo já é reais
    assert net.to(*GROQ) == []


@pytest.mark.parametrize("overrides", [{"groq_api_key": ""}])
def test_agent_off_without_key(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    r = agent(client)
    assert r.status_code == 503 and r.json()["detail"] == "agent_off"
    assert net.to(*GROQ) == []


def test_groq_failure_is_a_502(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    groq_says(net, {"error": "rate limit"}, status=429)
    assert agent(client).status_code == 502
    net.routes[GROQ] = lambda req: httpx2.Response(200, json={"choices": []})
    assert agent(client).status_code == 502


@pytest.mark.parametrize("overrides", [{"groq_api_key": "gsk-test", "global_daily_tokens": 300}])
def test_respects_the_global_token_budget(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    groq_says(net, {"value": "new car"}, tokens=300)
    assert agent(client).status_code == 200
    r = agent(client)
    assert r.status_code == 429 and r.json()["reason"] == "global_budget"
    assert len(net.to(*GROQ)) == 1


@pytest.mark.parametrize("overrides", [{"groq_api_key": "gsk-test", "credit_agent_per_hour": 2}])
def test_hourly_limit_per_user(client: TestClient, net: FakeNet) -> None:
    oauth_login(client)
    groq_says(net, {"value": "new car"})
    assert [agent(client).status_code for _ in range(3)] == [200, 200, 429]
    assert len(net.to(*GROQ)) == 2
