from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

from vgmelo_portal.context import SESSION_COOKIE
from vgmelo_portal.db import AuthSession, utcnow

from .conftest import SITE, FakeNet, make_client, oauth_login, pkce_ok


def test_google_login_uses_state_and_pkce(client: TestClient, net: FakeNet) -> None:
    start = client.get("/api/auth/google/start", params={"next": "/en/demos"})
    assert start.status_code == 302
    url = urlsplit(start.headers["location"])
    q = parse_qs(url.query)
    assert url.netloc == "accounts.google.com"
    assert q["redirect_uri"] == [f"{SITE}/api/auth/google/callback"]
    assert q["code_challenge_method"] == ["S256"] and q["scope"] == ["openid email profile"]
    assert len(q["state"][0]) >= 32

    done = client.get("/api/auth/google/callback", params={"code": "c0de", "state": q["state"][0]})
    assert done.status_code == 302
    assert done.headers["location"] == f"{SITE}/en/demos"

    token_req = net.to("oauth2.googleapis.com", "/token")[0]
    form = parse_qs(token_req.content.decode())
    assert form["code"] == ["c0de"]
    assert pkce_ok(form["code_verifier"][0], q["code_challenge"][0])

    me = client.get("/api/me").json()
    assert me["user"]["provider"] == "google" and me["user"]["email"] == "ana@gmail.com"
    assert me["user"]["name"] == "Ana" and me["user"]["mfa"] is False
    assert me["quotas"]["agents"] == {"limit": 3, "used": 0, "remaining": 3}


def test_github_login_takes_primary_verified_email(client: TestClient, net: FakeNet) -> None:
    r = oauth_login(client, "github")
    assert r.status_code == 302 and "auth_error" not in r.headers["location"]
    me = client.get("/api/me").json()
    assert me["user"]["provider"] == "github"
    assert me["user"]["email"] == "octo@example.com"
    token_req = net.to("github.com", "/login/oauth/access_token")[0]
    assert token_req.headers["accept"] == "application/json"
    assert "code_verifier" in parse_qs(token_req.content.decode())


def test_same_provider_subject_is_one_account(client: TestClient) -> None:
    oauth_login(client)
    first = client.get("/api/me").json()["user"]["id"]
    oauth_login(client)
    assert client.get("/api/me").json()["user"]["id"] == first


def test_callback_without_the_browser_cookie_is_refused(app: FastAPI, client: TestClient) -> None:
    """Login CSRF: an attacker sends the victim a callback URL with the attacker's own state."""
    start = client.get("/api/auth/google/start")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    with make_client(app) as victim:
        r = victim.get("/api/auth/google/callback", params={"code": "x", "state": state})
        assert r.headers["location"] == f"{SITE}/?auth_error=oauth"
        assert victim.get("/api/me").json()["user"] is None


def test_state_is_single_use(client: TestClient, net: FakeNet) -> None:
    start = client.get("/api/auth/google/start")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    assert (
        "auth_error"
        not in client.get("/api/auth/google/callback", params={"code": "a", "state": state}).headers["location"]
    )
    client.post("/api/auth/logout")
    # The browser still has no state cookie for it, and the row is gone either way.
    client.cookies.set("__Host-vgm_oauth", state, domain="testserver.local")
    again = client.get("/api/auth/google/callback", params={"code": "a", "state": state})
    assert "auth_error=oauth" in again.headers["location"]
    assert client.get("/api/me").json()["user"] is None


def test_state_for_another_provider_is_refused(client: TestClient) -> None:
    start = client.get("/api/auth/google/start")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    r = client.get("/api/auth/github/callback", params={"code": "a", "state": state})
    assert "auth_error=oauth" in r.headers["location"]


def test_provider_error_redirects_without_session(client: TestClient, net: FakeNet) -> None:
    import httpx2

    net.routes[("oauth2.googleapis.com", "/token")] = lambda req: httpx2.Response(400, json={"error": "invalid_grant"})
    r = oauth_login(client, next_path="/en/")
    assert r.headers["location"] == f"{SITE}/en/?auth_error=oauth"
    assert client.get("/api/me").json()["user"] is None


@pytest.mark.parametrize(
    "evil",
    [
        "https://evil.example/phish",
        "//evil.example/x",
        "/\\evil.example",
        "\\\\evil.example",
        "javascript:alert(1)",
        "/%0d%0aSet-Cookie:x=1",
        "evil.example",
        "@evil.example",
        ".evil.example/x",
        "/ok#frag'><script>",
    ],
)
def test_next_never_leaves_the_site(client: TestClient, evil: str) -> None:
    r = oauth_login(client, next_path=evil)
    location = r.headers["location"]
    assert location.startswith(SITE + "/"), location
    assert urlsplit(location).netloc == "vgmelo.tech"
    assert "evil" not in urlsplit(location).netloc


def test_relative_next_keeps_query(client: TestClient) -> None:
    r = oauth_login(client, next_path="/en/demos?tab=agents")
    assert r.headers["location"] == f"{SITE}/en/demos?tab=agents"


@pytest.fixture
def no_github(overrides: dict[str, Any]) -> None:
    overrides.update(github_client_id="", github_client_secret="")


def test_provider_without_credentials_is_off(no_github: None, client: TestClient) -> None:
    assert client.get("/api/auth/github/start").status_code == 404
    assert client.get("/api/me").json()["providers"] == {"google": True, "github": False, "email": True}


def test_unknown_provider_is_422(client: TestClient) -> None:
    assert client.get("/api/auth/facebook/start").status_code == 422


def test_session_cookie_flags(client: TestClient) -> None:
    r = oauth_login(client)
    cookie = next(v for k, v in r.headers.multi_items() if k == "set-cookie" and v.startswith(SESSION_COOKIE))
    parts = {p.strip().lower() for p in cookie.split(";")}
    assert {"httponly", "secure", "samesite=lax", "path=/"} <= parts
    assert not any(p.startswith("domain=") for p in parts)  # __Host- prefix: host-only
    token = cookie.split(";", 1)[0].split("=", 1)[1]
    assert len(token) >= 40


def test_session_is_stored_hashed(app: FastAPI, client: TestClient) -> None:
    oauth_login(client)
    token = client.cookies[SESSION_COOKIE]
    with app.state.ctx.sessions() as db:
        hashes = [s.token_hash for s in db.query(AuthSession).all()]
    assert token not in hashes and len(hashes) == 1


def test_login_rotates_the_session(client: TestClient) -> None:
    oauth_login(client)
    old = client.cookies[SESSION_COOKIE]
    oauth_login(client)
    new = client.cookies[SESSION_COOKIE]
    assert new != old
    client.cookies.set(SESSION_COOKIE, old, domain="testserver.local")
    assert client.get("/api/me").json()["user"] is None


def test_logout_invalidates_on_the_server(client: TestClient) -> None:
    oauth_login(client)
    token = client.cookies[SESSION_COOKIE]
    r = client.post("/api/auth/logout")
    assert r.status_code == 204
    assert "max-age=0" in r.headers["set-cookie"].lower() or "expires=" in r.headers["set-cookie"].lower()
    # Replaying the stolen cookie after logout gets nothing.
    client.cookies.set(SESSION_COOKIE, token, domain="testserver.local")
    assert client.get("/api/me").json()["user"] is None
    assert client.post("/api/demo/score", json={}).status_code == 401


def test_expired_session_is_ignored(app: FastAPI, client: TestClient) -> None:
    oauth_login(client)
    with app.state.ctx.sessions() as db:
        db.execute(update(AuthSession).values(expires_at=utcnow() - timedelta(seconds=1)))
        db.commit()
    assert client.get("/api/me").json()["user"] is None
