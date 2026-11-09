from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from vgmelo_portal.db import ContactMessage

from .conftest import FakeNet, email_user, make_client, oauth_login


def form(**kw: Any) -> dict[str, Any]:
    return {
        "name": "Recrutadora",
        "email": "rh@empresa.com",
        "kind": "job",
        "message": "Olá Victor, temos uma vaga de engenheiro de ML.",
        "website": "",
        "started_at": int((time.time() - 20) * 1000),
        **kw,
    }


def stored(app: FastAPI) -> list[ContactMessage]:
    with app.state.ctx.sessions() as db:
        return list(db.scalars(select(ContactMessage)))


def test_contact_is_stored_and_notified(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    r = client.post("/api/contact", json=form())
    assert r.status_code == 200 and r.json() == {"status": "sent"}
    rows = stored(app)
    assert len(rows) == 1 and rows[0].kind == "job" and rows[0].notified
    mail = net.emails()[-1]
    assert mail["to"] == ["dono@example.com"]
    assert mail["reply_to"] == "rh@empresa.com"
    assert "vaga de engenheiro" in mail["text"] and "html" not in mail


@pytest.fixture
def no_recipient(overrides: dict[str, Any]) -> None:
    overrides.update(contact_to="")


def test_contact_without_recipient_is_only_stored(
    no_recipient: None, client: TestClient, net: FakeNet, app: FastAPI
) -> None:
    r = client.post("/api/contact", json=form())
    assert r.status_code == 200 and r.json() == {"status": "sent"}
    assert len(stored(app)) == 1 and not stored(app)[0].notified
    assert net.to("api.resend.com") == []


def test_contact_subject_has_no_header_injection(client: TestClient, net: FakeNet) -> None:
    client.post("/api/contact", json=form(name="A\r\nBcc: x@evil.example"))
    assert "\n" not in net.emails()[-1]["subject"] and "\r" not in net.emails()[-1]["subject"]


def test_honeypot_is_silently_dropped(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    r = client.post("/api/contact", json=form(website="http://spam.example"))
    assert r.status_code == 200 and r.json() == {"status": "sent"}
    assert stored(app) == [] and net.emails() == []


@pytest.mark.parametrize("offset_s", [-1, 60, -2 * 86400])
def test_too_fast_or_bogus_timestamp_is_dropped(client: TestClient, net: FakeNet, app: FastAPI, offset_s: int) -> None:
    r = client.post("/api/contact", json=form(started_at=int((time.time() + offset_s) * 1000)))
    assert r.status_code == 200
    assert stored(app) == [] and net.emails() == []


def test_contact_rate_limit_per_ip(client: TestClient, app: FastAPI) -> None:
    for _ in range(5):
        assert client.post("/api/contact", json=form()).status_code == 200
    r = client.post("/api/contact", json=form())
    assert r.status_code == 429 and r.json()["reason"] == "rate_limited"
    with make_client(app, ip="198.51.100.60") as other:
        assert other.post("/api/contact", json=form()).status_code == 200


@pytest.mark.parametrize(
    "patch",
    [{"kind": "spam"}, {"email": "nope"}, {"message": "x" * 5001}, {"name": ""}, {"extra": 1}, {"message": "hi"}],
)
def test_contact_validation(client: TestClient, patch: dict[str, Any]) -> None:
    assert client.post("/api/contact", json=form(**patch)).status_code == 422


@pytest.fixture
def no_resend(overrides: dict[str, Any]) -> None:
    overrides.update(resend_api_key="")


def test_contact_without_resend_still_stores(no_resend: None, client: TestClient, app: FastAPI, net: FakeNet) -> None:
    assert client.post("/api/contact", json=form()).status_code == 200
    rows = stored(app)
    assert len(rows) == 1 and rows[0].notified is False
    assert net.emails() == []


def test_contact_survives_resend_outage(client: TestClient, app: FastAPI, net: FakeNet) -> None:
    import httpx2

    net.routes[("api.resend.com", "/emails")] = lambda req: httpx2.Response(500, json={})
    assert client.post("/api/contact", json=form()).status_code == 200
    assert stored(app)[0].notified is False


def test_admin_hidden_from_anonymous_and_non_admins(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    assert client.get("/admin").status_code == 404
    oauth_login(client, "github")  # octo@example.com is not in ADMIN_EMAILS
    assert client.get("/admin").status_code == 404
    with make_client(app, ip="198.51.100.61") as other:
        email_user(other, net, "someone@example.com")
        assert other.get("/admin").status_code == 404


def test_admin_requires_verified_email(client: TestClient, net: FakeNet) -> None:
    net.google_user = {**net.google_user, "email_verified": False}
    oauth_login(client)
    assert client.get("/admin").status_code == 404


def test_admin_shows_contacts_escaped(client: TestClient, net: FakeNet) -> None:
    client.post("/api/contact", json=form(name="<script>alert(1)</script>", message="<img src=x onerror=alert(1)>!"))
    oauth_login(client)  # ana@gmail.com, verified, listed in ADMIN_EMAILS
    r = client.get("/admin")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<script>alert" not in r.text and "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text
    assert "<img" not in r.text and "&lt;img src=x onerror=alert(1)&gt;" in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
