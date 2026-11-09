from __future__ import annotations

from datetime import timedelta
from typing import Any

import pyotp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from vgmelo_portal.db import PendingRegistration, RecoveryCode, User, utcnow

from .conftest import PASSWORD, FakeNet, code_at, email_user, make_client, oauth_login, register, setup_totp

EMAIL = "bia@example.com"


def login(client: TestClient, email: str = EMAIL, password: str = PASSWORD, **factor: str) -> Any:
    return client.post("/api/auth/email/login", json={"email": email, "password": password, **factor})


def test_full_signup_flow(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    r = client.post("/api/auth/email/register", json={"email": "Bia@Example.com", "password": PASSWORD})
    assert r.status_code == 202 and r.json() == {"status": "verification_sent"}
    mail = net.emails()[-1]
    assert mail["to"] == [EMAIL] and mail["from"] == "vgmelo.tech <no-reply@vgmelo.tech>"
    assert net.to("api.resend.com")[-1].headers["authorization"] == "Bearer re_test_key"

    r = client.post("/api/auth/email/verify", json={"email": EMAIL, "code": net.last_code(EMAIL)})
    assert r.json() == {"status": "totp_setup_required"}
    # Verified but no TOTP yet: not logged in for anything but the setup.
    me = client.get("/api/me").json()
    assert me["user"] is None and me["pending"] == "totp_setup"
    assert client.post("/api/demo/score", json={}).status_code == 401

    setup = client.post("/api/auth/totp/setup").json()
    assert setup["otpauth_uri"].startswith("otpauth://totp/vgmelo.tech:bia%40example.com?secret=")
    assert setup["qr_svg"].startswith("<svg")
    # Vai num <img src="data:image/svg+xml,...">: sem o namespace o navegador não desenha nada.
    assert 'xmlns="http://www.w3.org/2000/svg"' in setup["qr_svg"]
    secret, codes = setup_totp(client)
    assert len(codes) == 10 and len(set(codes)) == 10
    me = client.get("/api/me").json()
    assert me["user"]["provider"] == "email" and me["user"]["mfa"] is True and me["pending"] is None

    with app.state.ctx.sessions() as db:
        user = db.scalar(select(User).where(User.subject == EMAIL))
        assert user is not None and user.password_hash and user.password_hash.startswith("$argon2id$")
        assert user.totp_secret and secret not in user.totp_secret  # encrypted at rest
        stored = [rc.code_hash for rc in db.scalars(select(RecoveryCode))]
        assert len(stored) == 10 and not set(codes) & set(stored)

    client.post("/api/auth/logout")
    r = login(client, totp=code_at(secret, 1))
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert client.get("/api/me").json()["user"]["email"] == EMAIL


def test_register_does_not_reveal_existing_accounts(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    email_user(client, net)
    client.post("/api/auth/logout")
    sent_before = len(net.emails())
    fresh = client.post("/api/auth/email/register", json={"email": "new@example.com", "password": PASSWORD})
    taken = client.post("/api/auth/email/register", json={"email": EMAIL, "password": "another password 1"})
    assert fresh.status_code == taken.status_code == 202 and fresh.json() == taken.json()
    assert len(net.emails()) == sent_before + 2
    notice = net.emails()[-1]
    assert notice["to"] == [EMAIL] and "Seu código" not in notice["text"]
    with app.state.ctx.sessions() as db:
        assert db.scalar(select(PendingRegistration).where(PendingRegistration.email == EMAIL)) is None


def test_login_errors_are_identical_for_unknown_and_wrong_password(client: TestClient, net: FakeNet) -> None:
    secret, _ = email_user(client, net)
    client.post("/api/auth/logout")
    unknown = login(client, email="nobody@example.com", totp="123456")
    wrong_pw = login(client, password="wrong password!", totp=code_at(secret, 1))
    wrong_totp = login(client, totp="000000")
    assert unknown.status_code == wrong_pw.status_code == wrong_totp.status_code == 401
    assert unknown.json() == wrong_pw.json() == wrong_totp.json() == {"detail": "invalid_credentials"}


def test_verify_code_only_works_in_the_registering_browser(app: FastAPI, client: TestClient, net: FakeNet) -> None:
    """Someone registers *your* address with *their* password: the code lands in your inbox, but typing it
    in your browser must not activate their password."""
    with make_client(app, ip="198.51.100.9") as attacker:
        attacker.post("/api/auth/email/register", json={"email": EMAIL, "password": "attacker password"})
    code = net.last_code(EMAIL)
    r = client.post("/api/auth/email/verify", json={"email": EMAIL, "code": code})
    assert r.status_code == 400 and r.json() == {"detail": "invalid_code"}
    # The victim's own registration still works.
    register(client, net, EMAIL)


def test_verify_code_expires(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    client.post("/api/auth/email/register", json={"email": EMAIL, "password": PASSWORD})
    with app.state.ctx.sessions() as db:
        db.execute(update(PendingRegistration).values(expires_at=utcnow() - timedelta(seconds=1)))
        db.commit()
    r = client.post("/api/auth/email/verify", json={"email": EMAIL, "code": net.last_code(EMAIL)})
    assert r.status_code == 400


def test_verify_code_dies_after_five_wrong_tries(client: TestClient, net: FakeNet) -> None:
    client.post("/api/auth/email/register", json={"email": EMAIL, "password": PASSWORD})
    code = net.last_code(EMAIL)
    wrong = f"{(int(code) + 1) % 1_000_000:06d}"
    for _ in range(5):
        assert client.post("/api/auth/email/verify", json={"email": EMAIL, "code": wrong}).status_code == 400
    assert client.post("/api/auth/email/verify", json={"email": EMAIL, "code": code}).status_code == 400


def test_totp_code_cannot_be_reused(client: TestClient, net: FakeNet) -> None:
    secret, _ = email_user(client, net)
    client.post("/api/auth/logout")
    code = code_at(secret, 1)
    assert login(client, totp=code).status_code == 200
    client.post("/api/auth/logout")
    assert login(client, totp=code).status_code == 401
    # The code used at setup (current step) is also spent.
    assert login(client, totp=code_at(secret, 0)).status_code == 401


def test_totp_window_is_short(client: TestClient, net: FakeNet) -> None:
    secret, _ = email_user(client, net)
    client.post("/api/auth/logout")
    assert login(client, totp=code_at(secret, 3)).status_code == 401  # 90s in the future
    assert login(client, totp=code_at(secret, -3)).status_code == 401  # 90s in the past


def test_wrong_totp_locks_the_account(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    secret, _ = email_user(client, net)
    client.post("/api/auth/logout")
    for _ in range(5):
        assert login(client, totp="000000").status_code == 401
    locked = login(client, totp=code_at(secret, 1))
    assert locked.status_code == 429
    assert locked.json()["reason"] == "locked" and "retry_at" in locked.json()
    assert int(locked.headers["retry-after"]) > 0
    # Per account: another IP is locked out of this account too.
    with make_client(app, ip="198.51.100.20") as other:
        assert login(other, totp=code_at(secret, 1)).status_code == 429


def test_lockout_also_covers_unknown_accounts(client: TestClient) -> None:
    for _ in range(5):
        assert login(client, email="ghost@example.com", totp="000000").status_code == 401
    assert login(client, email="ghost@example.com", totp="000000").status_code == 429


@pytest.fixture
def ip_lock_small(overrides: dict[str, Any]) -> None:
    overrides.update(lockout_max_per_ip=3)


def test_lockout_per_ip_across_accounts(ip_lock_small: None, client: TestClient, app: FastAPI) -> None:
    for i in range(3):
        assert login(client, email=f"user{i}@example.com", totp="000000").status_code == 401
    r = login(client, email="someone-else@example.com", totp="000000")
    assert r.status_code == 429 and r.json()["reason"] == "locked"
    with make_client(app, ip="198.51.100.30") as other:
        assert login(other, email="someone-else@example.com", totp="000000").status_code == 401


def test_recovery_code_works_once(client: TestClient, net: FakeNet) -> None:
    _, codes = email_user(client, net)
    client.post("/api/auth/logout")
    assert login(client, recovery_code=codes[0].upper()).status_code == 200
    client.post("/api/auth/logout")
    assert login(client, recovery_code=codes[0]).status_code == 401
    assert login(client, recovery_code=codes[1]).status_code == 200


def test_password_alone_is_not_enough(client: TestClient, net: FakeNet) -> None:
    email_user(client, net)
    client.post("/api/auth/logout")
    assert login(client).status_code == 401
    assert client.get("/api/me").json()["user"] is None


def test_abandoned_setup_resumes_with_password(client: TestClient, net: FakeNet) -> None:
    register(client, net, EMAIL)
    client.post("/api/auth/logout")
    r = login(client)
    assert r.json() == {"status": "totp_setup_required"}
    setup_totp(client)
    assert client.get("/api/me").json()["user"]["mfa"] is True


def test_wrong_totp_at_setup_counts_towards_lockout(client: TestClient, net: FakeNet) -> None:
    register(client, net, EMAIL)
    client.post("/api/auth/totp/setup")
    for _ in range(5):
        assert client.post("/api/auth/totp/confirm", json={"code": "000000"}).status_code == 400
    assert client.post("/api/auth/totp/confirm", json={"code": "000000"}).status_code == 429


def test_totp_cannot_be_replaced_once_enabled(client: TestClient, net: FakeNet) -> None:
    email_user(client, net)
    assert client.post("/api/auth/totp/setup").status_code == 409


def test_totp_setup_needs_a_session_and_an_email_account(client: TestClient) -> None:
    assert client.post("/api/auth/totp/setup").status_code == 401
    oauth_login(client)
    assert client.post("/api/auth/totp/setup").status_code == 404


def test_pending_session_expires_quickly(client: TestClient, net: FakeNet, app: FastAPI) -> None:
    register(client, net, EMAIL)
    with app.state.ctx.sessions() as db:
        from vgmelo_portal.db import AuthSession

        row = db.scalar(select(AuthSession))
        assert row is not None and row.mfa_ok is False
        assert row.expires_at - utcnow() <= timedelta(minutes=15)


def test_register_rate_limit_per_ip(client: TestClient) -> None:
    for i in range(5):
        r = client.post("/api/auth/email/register", json={"email": f"u{i}@example.com", "password": PASSWORD})
        assert r.status_code == 202
    r = client.post("/api/auth/email/register", json={"email": "u9@example.com", "password": PASSWORD})
    assert r.status_code == 429 and r.json()["reason"] == "rate_limited"


def test_password_rules(client: TestClient) -> None:
    short = client.post("/api/auth/email/register", json={"email": EMAIL, "password": "short"})
    huge = client.post("/api/auth/email/register", json={"email": EMAIL, "password": "x" * 129})
    bad_email = client.post("/api/auth/email/register", json={"email": "not-an-email", "password": PASSWORD})
    assert short.status_code == huge.status_code == bad_email.status_code == 422


@pytest.fixture
def no_resend(overrides: dict[str, Any]) -> None:
    overrides.update(resend_api_key="")


def test_without_resend_email_login_is_off_but_oauth_works(no_resend: None, client: TestClient, net: FakeNet) -> None:
    assert client.post("/api/auth/email/register", json={"email": EMAIL, "password": PASSWORD}).status_code == 404
    assert login(client, totp="123456").status_code == 404
    assert client.get("/api/me").json()["providers"]["email"] is False
    oauth_login(client)
    assert client.get("/api/me").json()["user"]["provider"] == "google"
    assert net.to("api.resend.com") == []


def test_secrets_never_reach_the_logs(client: TestClient, net: FakeNet, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    secret, codes = email_user(client, net)
    client.post("/api/auth/logout")
    totp = code_at(secret, 1)
    login(client, totp=totp)
    oauth_login(client)
    logs = caplog.text
    email_code = net.last_code(EMAIL)
    for needle in [PASSWORD, secret, codes[0], totp, email_code, "auth-code", "g-access", "re_test_key"]:
        assert needle not in logs, needle
    assert client.cookies["__Host-vgm_session"] not in logs
    assert "login ok provider=email" in logs


def test_totp_secret_matches_pyotp(client: TestClient, net: FakeNet) -> None:
    secret, _ = email_user(client, net)
    assert len(pyotp.TOTP(secret).now()) == 6
