from __future__ import annotations

import ipaddress
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vgmelo_portal.security import Keyring, client_ip, match_totp, safe_next

from .conftest import PASSWORD, make_client


@pytest.mark.parametrize("origin", [None, "https://evil.example", "null", "https://vgmelo.tech.evil.example"])
def test_unsafe_methods_need_our_origin(client: TestClient, origin: str | None) -> None:
    headers = {"Origin": origin} if origin else {}
    if origin is None:
        client.headers.pop("Origin")
    for path in ["/api/auth/logout", "/api/auth/email/register", "/api/contact", "/api/demo/score"]:
        r = client.post(path, json={}, headers=headers)
        assert r.status_code == 403 and r.json() == {"detail": "bad_origin"}, path


def test_get_without_origin_is_fine(client: TestClient) -> None:
    client.headers.pop("Origin")
    assert client.get("/api/me").status_code == 200


def test_allowed_origin_list(overrides: dict[str, Any], client: TestClient) -> None:
    assert client.post("/api/auth/logout").status_code == 204


@pytest.fixture
def two_origins(overrides: dict[str, Any]) -> None:
    overrides.update(allowed_origins="https://vgmelo.tech,https://www.vgmelo.tech")


def test_extra_origin(two_origins: None, client: TestClient) -> None:
    assert client.post("/api/auth/logout", headers={"Origin": "https://www.vgmelo.tech"}).status_code == 204


def test_body_size_cap(client: TestClient) -> None:
    big = {"email": "a@example.com", "password": PASSWORD, "pad": "x" * 20_000}
    assert client.post("/api/auth/email/register", json=big).status_code == 413


def test_body_size_cap_without_content_length(client: TestClient) -> None:
    def chunks() -> Any:
        for _ in range(40):
            yield b"x" * 1024

    r = client.post("/api/contact", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_api_responses_are_not_cached(client: TestClient) -> None:
    r = client.get("/api/me")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_no_docs_exposed(client: TestClient) -> None:
    for path in ["/docs", "/openapi.json", "/redoc"]:
        assert client.get(path).status_code == 404


NETS = (ipaddress.ip_network("172.18.0.0/16"),)


@pytest.mark.parametrize(
    ("peer", "xff", "expected"),
    [
        ("203.0.113.5", "1.2.3.4", "203.0.113.5"),  # untrusted peer: header ignored
        ("172.18.0.2", "1.2.3.4", "1.2.3.4"),  # Caddy appends the real client
        ("172.18.0.2", "6.6.6.6, 1.2.3.4", "1.2.3.4"),  # spoofed left-most hop ignored
        ("172.18.0.2", "1.2.3.4, 172.18.0.3", "1.2.3.4"),  # chain of trusted proxies
        ("172.18.0.2", None, "172.18.0.2"),
        ("172.18.0.2", "garbage", "172.18.0.2"),
    ],
)
def test_client_ip(peer: str, xff: str | None, expected: str) -> None:
    assert client_ip(peer, xff, NETS) == expected


def test_client_ip_without_trusted_proxies_ignores_header() -> None:
    assert client_ip("172.18.0.2", "1.2.3.4", ()) == "172.18.0.2"


@pytest.fixture
def behind_proxy(overrides: dict[str, Any]) -> None:
    overrides.update(trusted_proxies="172.18.0.0/16", register_max_per_ip_hour=2)


def test_rate_limit_uses_forwarded_ip_only_from_proxy(behind_proxy: None, app: FastAPI) -> None:
    def register(c: TestClient, i: int, xff: str) -> int:
        body = {"email": f"p{i}@example.com", "password": PASSWORD}
        return c.post("/api/auth/email/register", json=body, headers={"X-Forwarded-For": xff}).status_code

    with make_client(app, ip="172.18.0.2") as proxy:
        assert [register(proxy, i, "1.1.1.1") for i in range(3)] == [202, 202, 429]
        assert register(proxy, 9, "2.2.2.2") == 202  # another visitor behind the same proxy
    with make_client(app, ip="203.0.113.50") as direct:
        # Spoofing the header from outside doesn't buy a fresh bucket.
        assert [register(direct, 10 + i, f"9.9.9.{i}") for i in range(3)] == [202, 202, 429]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/en/demos", "/en/demos"),
        ("/a?b=c", "/a?b=c"),
        (None, "/"),
        ("", "/"),
        ("//evil.example", "/"),
        ("/\\evil.example", "/"),
        ("https://evil.example", "/"),
        ("javascript:alert(1)", "/"),
        ("@evil.example", "/"),
        ("/ok#x", "/"),
        ("/x\r\nLocation: //evil", "/"),
        ("/" + "a" * 600, "/"),
    ],
)
def test_safe_next(value: str | None, expected: str) -> None:
    assert safe_next(value) == expected


def test_match_totp_refuses_replay_and_far_codes() -> None:
    import pyotp

    secret = pyotp.random_base32()
    now = time.time()
    step = int(now // 30)
    code = pyotp.TOTP(secret).at(now)
    assert match_totp(secret, code, 0, now) == step
    assert match_totp(secret, code, step, now) is None
    assert match_totp(secret, pyotp.TOTP(secret).at(now + 120), 0, now) is None
    assert match_totp(secret, "12345a", 0, now) is None


def test_keyring_encrypts_with_fresh_nonce() -> None:
    k = Keyring("k" * 40)
    a, b = k.encrypt("SECRET"), k.encrypt("SECRET")
    assert a != b and k.decrypt(a) == k.decrypt(b) == "SECRET"
    assert Keyring("k" * 40).mac("x") == k.mac("x") != Keyring("j" * 40).mac("x")
