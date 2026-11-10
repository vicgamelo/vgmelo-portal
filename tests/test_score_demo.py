"""Credit demo: live pipeline (SSE relay) and batch CSV, behind login and the daily score quota."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from .conftest import FakeNet, oauth_login, sse
from .test_demos import APPLICANT, events

CSV = {"content-type": "text/csv"}
FILE = b"id,age\n1,30\n"
DRAFT = {**APPLICANT, "duration": "48", "age": None}
PIPELINE_EVENTS = sse(
    ("stage", {"id": "raw", "ms": 0.1}),
    ("stage", {"id": "transform", "ms": 0.2}),
    ("result", {"default_probability": 0.71, "decision": "review", "model_version": "3"}),
    ("done", {"ok": True}),
)
REPORT = {"ok": False, "rows": 1, "valid_rows": 0, "file_errors": [{"code": "missing_columns", "message": "faltam"}]}


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {"quota_score": 5}


@pytest.fixture
def scoring(net: FakeNet) -> FakeNet:
    net.routes[("scoring", "/pipeline")] = lambda req: httpx2.Response(
        200, content=PIPELINE_EVENTS, headers={"content-type": "text/event-stream"}
    )
    net.routes[("scoring", "/batch/validate")] = lambda req: httpx2.Response(200, json={"ok": True, "rows": 1})
    net.routes[("scoring", "/batch/score")] = lambda req: httpx2.Response(
        200, json={"model_version": "3", "report": {"ok": True}, "results": [], "csv": "line\n"}
    )
    return net


def used(client: TestClient) -> int:
    return int(client.get("/api/me").json()["quotas"]["score"]["used"])


def pipeline(client: TestClient, body: dict[str, Any] | None = None) -> httpx2.Response:
    return client.post("/api/demo/score/pipeline", json=body or {"applicant": DRAFT, "lang": "en"})


def batch(client: TestClient, path: str = "/api/demo/score/batch", body: bytes = FILE) -> httpx2.Response:
    return client.post(f"{path}?lang=en", content=body, headers=CSV)


@pytest.mark.parametrize(
    ("path", "kwargs"),
    [
        ("/api/demo/score/pipeline", {"json": {"applicant": DRAFT}}),
        ("/api/demo/score/batch/validate", {"content": FILE, "headers": CSV}),
        ("/api/demo/score/batch", {"content": FILE, "headers": CSV}),
    ],
)
def test_need_login(client: TestClient, scoring: FakeNet, path: str, kwargs: dict[str, Any]) -> None:
    r = client.post(path, **kwargs)
    assert r.status_code == 401
    assert scoring.to("scoring") == []


def test_pipeline_relays_the_stream_and_costs_one_use(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    r = pipeline(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    assert [d.get("id", e) for e, d in events(r.text)] == ["raw", "transform", "result", "done"]
    sent = scoring.to("scoring", "/pipeline")[0]
    assert sent.url.params["lang"] == "en"
    assert json.loads(sent.content) == {"applicant": DRAFT}  # loose values go through untouched
    assert used(client) == 1


@pytest.mark.parametrize(
    "patch",
    [{"age": "x" * 65}, {"age": [1]}, {"age": {"a": 1}}, {"salary": 1}, {"age": True}],
)
def test_pipeline_refuses_shapes_before_charging(client: TestClient, scoring: FakeNet, patch: dict[str, Any]) -> None:
    oauth_login(client)
    assert pipeline(client, {"applicant": {**DRAFT, **patch}}).status_code == 422
    assert scoring.to("scoring") == [] and used(client) == 0


def test_pipeline_needs_every_field_even_if_null(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    draft = dict(DRAFT)
    del draft["job"]
    assert pipeline(client, {"applicant": draft}).status_code == 422


def test_pipeline_invalid_profile_is_422_with_problems_and_refunded(client: TestClient, scoring: FakeNet) -> None:
    problems = [{"column": "age", "code": "out_of_range", "message": "fora", "value": "7", "severity": "error"}]
    scoring.routes[("scoring", "/pipeline")] = lambda req: httpx2.Response(
        422, json={"detail": {"reason": "invalid_applicant", "problems": [{**problems[0], "extra": "x"}]}}
    )
    oauth_login(client)
    r = pipeline(client)
    assert r.status_code == 422
    assert r.json()["detail"] == {"reason": "invalid_applicant", "problems": problems}
    assert used(client) == 0


@pytest.mark.parametrize(("status", "expected"), [(503, 502), (500, 502), (429, 429)])
def test_pipeline_upstream_errors_refund(client: TestClient, scoring: FakeNet, status: int, expected: int) -> None:
    scoring.routes[("scoring", "/pipeline")] = lambda req: httpx2.Response(status, json={"detail": "x"})
    oauth_login(client)
    assert pipeline(client).status_code == expected
    assert used(client) == 0


def test_pipeline_scoring_down_refunds(client: TestClient, scoring: FakeNet) -> None:
    def down(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    scoring.routes[("scoring", "/pipeline")] = down
    oauth_login(client)
    assert pipeline(client).status_code == 502 and used(client) == 0


def test_validate_is_free_and_forwards_the_raw_csv(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    r = batch(client, "/api/demo/score/batch/validate")
    assert r.status_code == 200 and r.json() == {"ok": True, "rows": 1}
    sent = scoring.to("scoring", "/batch/validate")[0]
    assert sent.content == FILE and sent.headers["content-type"] == "text/csv" and sent.url.params["lang"] == "en"
    assert used(client) == 0


@pytest.fixture
def tight(overrides: dict[str, Any]) -> None:
    overrides["batch_validate_per_hour"] = 2


def test_validate_rate_limit(tight: None, client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    assert batch(client, "/api/demo/score/batch/validate").status_code == 200
    assert batch(client, "/api/demo/score/batch/validate").status_code == 200
    r = batch(client, "/api/demo/score/batch/validate")
    assert r.status_code == 429 and r.json()["reason"] == "rate_limited"


def test_batch_is_one_use_per_file(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    r = batch(client)
    assert r.status_code == 200 and r.json()["csv"] == "line\n"
    assert used(client) == 1


def test_batch_with_nothing_to_score_is_refunded(client: TestClient, scoring: FakeNet) -> None:
    scoring.routes[("scoring", "/batch/score")] = lambda req: httpx2.Response(
        422, json={"detail": {"reason": "no_valid_rows", "report": REPORT}}
    )
    oauth_login(client)
    r = batch(client)
    assert r.status_code == 422 and r.json()["detail"] == {"reason": "no_valid_rows", "report": REPORT}
    assert used(client) == 0


@pytest.mark.parametrize("status", [500, 503])
def test_batch_upstream_error_refunds(client: TestClient, scoring: FakeNet, status: int) -> None:
    scoring.routes[("scoring", "/batch/score")] = lambda req: httpx2.Response(status, json={})
    oauth_login(client)
    assert batch(client).status_code == 502 and used(client) == 0


def test_batch_needs_a_csv_body(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    assert client.post("/api/demo/score/batch", json={"csv": "a,b"}).status_code == 415
    assert batch(client, body=b"  \n").status_code == 422
    assert scoring.to("scoring") == [] and used(client) == 0


def test_upload_routes_take_bigger_bodies_than_the_rest(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    forty_kb = b"id,age\n" + b"1,30\n" * 8000
    assert 16 * 1024 < len(forty_kb) < 50 * 1024
    assert batch(client, "/api/demo/score/batch/validate", forty_kb).status_code == 200
    assert batch(client, body=b"x" * (50 * 1024 + 1)).status_code == 413
    # The JSON routes keep the small cap.
    big = {"applicant": {**DRAFT, "purpose": "x"}, "pad": "x" * (17 * 1024)}
    assert client.post("/api/demo/score/pipeline", json=big).status_code == 413
    assert used(client) == 0  # validate is free and the 413s never reach the quota


def test_quota_is_shared_by_pipeline_batch_and_single_score(client: TestClient, scoring: FakeNet) -> None:
    oauth_login(client)
    assert pipeline(client).status_code == 200
    assert batch(client).status_code == 200
    assert client.post("/api/demo/score", json=APPLICANT).status_code == 200
    assert pipeline(client).status_code == 200
    assert batch(client).status_code == 200
    assert used(client) == 5
    for r in (pipeline(client), batch(client), client.post("/api/demo/score", json=APPLICANT)):
        assert r.status_code == 429 and r.json()["reason"] == "daily_quota"
    assert len(scoring.to("scoring")) == 5
    # Validating still works when the quota is gone: it doesn't score anything.
    assert batch(client, "/api/demo/score/batch/validate").status_code == 200


def test_single_score_out_of_domain_is_422_and_refunded(client: TestClient, scoring: FakeNet) -> None:
    """Bug: /predict now refuses categories outside credit-g with 422; the portal answered 502 and kept the use."""
    scoring.routes[("scoring", "/predict")] = lambda req: httpx2.Response(422, json={"detail": []})
    oauth_login(client)
    r = client.post("/api/demo/score", json={**APPLICANT, "purpose": "banana"})
    assert r.status_code == 422 and r.json() == {"detail": "invalid_applicant"}
    assert used(client) == 0
