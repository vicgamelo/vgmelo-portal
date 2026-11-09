"""Login: Google / GitHub OAuth (state + PKCE) and email + password + mandatory TOTP.

Sessions live in the database; the cookie only carries a random token whose SHA-256 is the row key.
"""

from __future__ import annotations

import hmac
import io
import logging
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Literal
from urllib.parse import urlencode

import segno
from authlib.integrations.httpx_client import OAuth2Client
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .context import DB, IP, SESSION_COOKIE, AnySession, AppContext, Ctx, MaybeCurrent
from .db import AuthSession, OAuthState, PendingRegistration, RecoveryCode, User, affected, utcnow
from .limits import check_locked, hit, purge_old_counters, record_failure
from .security import (
    hash_password,
    match_totp,
    new_token,
    new_totp_secret,
    normalize_recovery_code,
    password_needs_rehash,
    recovery_codes,
    safe_next,
    sha256,
    six_digit_code,
    totp_uri,
    verify_password,
)

logger = logging.getLogger(__name__)

OAUTH_COOKIE = "__Host-vgm_oauth"
REG_COOKIE = "__Host-vgm_reg"
OAUTH_TTL_S = 600
MAX_CODE_ATTEMPTS = 5

router = APIRouter(prefix="/api/auth")


# ---------------------------------------------------------------- sessions


def set_cookie(response: Response, name: str, value: str, max_age: int) -> None:
    response.set_cookie(name, value, max_age=max_age, path="/", secure=True, httponly=True, samesite="lax")


def clear_cookie(response: Response, name: str) -> None:
    response.delete_cookie(name, path="/", secure=True, httponly=True, samesite="lax")


def cleanup(db: Session) -> None:
    now = utcnow()
    db.execute(delete(AuthSession).where(AuthSession.expires_at <= now))
    db.execute(delete(OAuthState).where(OAuthState.expires_at <= now))
    db.execute(delete(PendingRegistration).where(PendingRegistration.expires_at <= now))
    purge_old_counters(db)


def start_session(ctx: AppContext, db: Session, request: Request, response: Response, user: User, mfa_ok: bool) -> None:
    """Always a fresh token (rotation): the session presented with the request, if any, is destroyed."""
    old = request.cookies.get(SESSION_COOKIE)
    if old:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == sha256(old)))
    cleanup(db)
    ttl = ctx.session_ttl if mfa_ok else ctx.pending_ttl
    token = new_token()
    db.add(AuthSession(token_hash=sha256(token), user_id=user.id, mfa_ok=mfa_ok, expires_at=utcnow() + ttl))
    if mfa_ok:
        user.last_login_at = utcnow()
    db.commit()
    set_cookie(response, SESSION_COOKIE, token, int(ttl.total_seconds()))


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: DB) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == sha256(token)))
        db.commit()
    clear_cookie(response, SESSION_COOKIE)


# ---------------------------------------------------------------- OAuth


@dataclass(frozen=True)
class Provider:
    name: str
    authorize_url: str
    token_url: str
    scope: str


PROVIDERS = {
    "google": Provider(
        "google",
        "https://accounts.google.com/o/oauth2/v2/auth",
        "https://oauth2.googleapis.com/token",
        "openid email profile",
    ),
    "github": Provider(
        "github",
        "https://github.com/login/oauth/authorize",
        "https://github.com/login/oauth/access_token",
        "read:user user:email",
    ),
}
GOOGLE_USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"
GITHUB_USER = "https://api.github.com/user"
GITHUB_EMAILS = "https://api.github.com/user/emails"

ProviderName = Literal["google", "github"]


def _oauth_client(ctx: AppContext, provider: str) -> OAuth2Client:
    s = ctx.settings
    if provider == "google":
        client_id, secret = s.google_client_id, s.google_client_secret
    else:
        client_id, secret = s.github_client_id, s.github_client_secret
    if not (client_id and secret):
        raise HTTPException(404, "provider_disabled")
    return OAuth2Client(
        client_id=client_id,
        client_secret=secret,
        scope=PROVIDERS[provider].scope,
        redirect_uri=f"{s.site_url}/api/auth/{provider}/callback",
        code_challenge_method="S256",
        transport=ctx.transport,
        timeout=10,
    )


@router.get("/{provider}/start")
def oauth_start(
    provider: ProviderName, ctx: Ctx, db: DB, ip: IP, next: Annotated[str | None, Query(max_length=500)] = None
) -> RedirectResponse:
    hit(db, ctx.window("oauth_start_ip", ctx.settings.oauth_start_max_per_ip_window), ctx.keyring.mac(ip))
    with _oauth_client(ctx, provider) as client:
        verifier = secrets.token_urlsafe(64)
        state = new_token()
        url, _ = client.create_authorization_url(PROVIDERS[provider].authorize_url, state=state, code_verifier=verifier)
    db.add(
        OAuthState(
            state_hash=sha256(state),
            provider=provider,
            code_verifier=verifier,
            next_path=safe_next(next),
            expires_at=utcnow() + timedelta(seconds=OAUTH_TTL_S),
        )
    )
    db.commit()
    resp = RedirectResponse(url, status_code=302)
    # Binds the flow to this browser: a callback URL planted by someone else (login CSRF) has no cookie.
    set_cookie(resp, OAUTH_COOKIE, state, OAUTH_TTL_S)
    return resp


@router.get("/{provider}/callback")
def oauth_callback(
    provider: ProviderName,
    request: Request,
    ctx: Ctx,
    db: DB,
    code: Annotated[str | None, Query(max_length=2000)] = None,
    state: Annotated[str | None, Query(max_length=200)] = None,
) -> RedirectResponse:
    cookie = request.cookies.get(OAUTH_COOKIE) or ""
    row = db.get(OAuthState, sha256(state)) if state else None
    if row is not None:
        db.delete(row)  # single use, whatever happens next
        db.commit()
    ok = (
        row is not None
        and state is not None
        and hmac.compare_digest(cookie, state)
        and row.provider == provider
        and row.expires_at > utcnow()
        and code
    )
    if not ok or row is None or code is None:
        logger.info("oauth %s callback refused (state)", provider)
        return _auth_redirect(ctx, "/", error="oauth")
    try:
        profile = _fetch_profile(ctx, provider, code, row.code_verifier)
    except Exception as exc:  # any provider failure ends the same way for the visitor
        logger.warning("oauth %s exchange failed: %s", provider, type(exc).__name__)
        return _auth_redirect(ctx, row.next_path, error="oauth")
    user = db.scalar(select(User).where(User.provider == provider, User.subject == profile["subject"]))
    if user is None:
        user = User(id=secrets.token_hex(16), provider=provider, subject=profile["subject"])
        db.add(user)
    user.email, user.email_verified, user.name = profile["email"], profile["email_verified"], profile["name"]
    db.flush()
    resp = _auth_redirect(ctx, row.next_path)
    start_session(ctx, db, request, resp, user, mfa_ok=True)
    clear_cookie(resp, OAUTH_COOKIE)
    logger.info("login ok provider=%s user=%s", provider, user.id)
    return resp


def _fetch_profile(ctx: AppContext, provider: str, code: str, verifier: str) -> dict[str, Any]:
    with _oauth_client(ctx, provider) as client:
        client.fetch_token(PROVIDERS[provider].token_url, code=code, code_verifier=verifier)
        if provider == "google":
            info = client.get(GOOGLE_USERINFO).raise_for_status().json()
            return {
                "subject": str(info["sub"]),
                "email": (info.get("email") or "").lower() or None,
                "email_verified": bool(info.get("email_verified")),
                "name": (info.get("name") or "")[:200] or None,
            }
        headers = {"Accept": "application/vnd.github+json"}
        info = client.get(GITHUB_USER, headers=headers).raise_for_status().json()
        emails = client.get(GITHUB_EMAILS, headers=headers).raise_for_status().json()
        primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
        return {
            "subject": str(info["id"]),
            "email": primary["email"].lower() if primary else None,
            "email_verified": primary is not None,
            "name": (info.get("name") or info.get("login") or "")[:200] or None,
        }


def _auth_redirect(ctx: AppContext, path: str, error: str | None = None) -> RedirectResponse:
    target = ctx.settings.site_url + safe_next(path)
    if error:
        target += ("&" if "?" in target else "?") + urlencode({"auth_error": error})
    return RedirectResponse(target, status_code=302)


# ---------------------------------------------------------------- email + password + TOTP


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegisterIn(Strict):
    email: EmailStr = Field(max_length=254)
    password: str = Field(min_length=10, max_length=128)


class VerifyIn(Strict):
    email: EmailStr = Field(max_length=254)
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class LoginIn(Strict):
    email: EmailStr = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)
    totp: str | None = Field(default=None, max_length=6)
    recovery_code: str | None = Field(default=None, max_length=20)


class CodeIn(Strict):
    code: str = Field(min_length=6, max_length=6)


def _require_email(ctx: AppContext) -> None:
    if not ctx.settings.email_enabled:
        raise HTTPException(404, "email_login_disabled")


def _email_user(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.provider == "email", User.subject == email))


@router.post("/email/register", status_code=202)
def email_register(body: RegisterIn, request: Request, response: Response, ctx: Ctx, db: DB, ip: IP) -> dict[str, str]:
    _require_email(ctx)
    email = body.email.lower()
    hit(db, ctx.window("register_ip", ctx.settings.register_max_per_ip_hour, 3600), ctx.keyring.mac(ip))
    hit(db, ctx.window("register_email", ctx.settings.register_max_per_email_hour, 3600), ctx.keyring.mac(email))
    password_hash = hash_password(body.password)  # always, so both branches cost the same
    browser = request.cookies.get(REG_COOKIE) or new_token()
    ttl = ctx.settings.email_code_ttl_minutes
    if _email_user(db, email) is not None:
        ctx.mailer.send(
            email,
            "Sua conta no vgmelo.tech / Your vgmelo.tech account",
            "Alguém tentou criar uma conta com este e-mail, mas ela já existe. Se foi você, é só entrar.\n\n"
            "Someone tried to sign up with this address, but the account already exists. If it was you, just log in.",
        )
    else:
        code = six_digit_code()
        db.add(
            PendingRegistration(
                email=email,
                browser_hash=sha256(browser),
                password_hash=password_hash,
                code_hash=ctx.keyring.mac(code),
                expires_at=utcnow() + timedelta(minutes=ttl),
            )
        )
        db.commit()
        ctx.mailer.send(
            email,
            "Código de verificação vgmelo.tech / verification code",
            f"Seu código: {code}\nYour code: {code}\n\nVale por {ttl} minutos / valid for {ttl} minutes.\n"
            "Se não foi você, ignore este e-mail. / If this wasn't you, ignore this email.",
        )
    set_cookie(response, REG_COOKIE, browser, ttl * 60)
    return {"status": "verification_sent"}


@router.post("/email/verify")
def email_verify(body: VerifyIn, request: Request, response: Response, ctx: Ctx, db: DB, ip: IP) -> dict[str, str]:
    _require_email(ctx)
    email = body.email.lower()
    ip_key = ctx.keyring.mac(ip)
    check_locked(db, ctx.lock_ip, ip_key)
    browser = request.cookies.get(REG_COOKIE) or ""
    row = db.scalar(
        select(PendingRegistration)
        .where(
            PendingRegistration.email == email,
            PendingRegistration.browser_hash == sha256(browser),
            PendingRegistration.expires_at > utcnow(),
        )
        .order_by(PendingRegistration.id.desc())
        .limit(1)
    )
    if (
        row is None
        or row.attempts >= MAX_CODE_ATTEMPTS
        or not hmac.compare_digest(row.code_hash, ctx.keyring.mac(body.code))
    ):
        if row is not None:
            row.attempts += 1
            db.commit()
        record_failure(db, ctx.lock_ip, ip_key)
        raise HTTPException(400, "invalid_code")
    if _email_user(db, email) is not None:
        db.execute(delete(PendingRegistration).where(PendingRegistration.email == email))
        db.commit()
        raise HTTPException(400, "invalid_code")
    user = User(
        id=secrets.token_hex(16), provider="email", subject=email, email=email, email_verified=True,
        password_hash=row.password_hash,
    )  # fmt: skip
    db.add(user)
    db.execute(delete(PendingRegistration).where(PendingRegistration.email == email))
    db.flush()
    start_session(ctx, db, request, response, user, mfa_ok=False)
    clear_cookie(response, REG_COOKIE)
    logger.info("email verified user=%s", user.id)
    return {"status": "totp_setup_required"}


@router.post("/email/login")
def email_login(body: LoginIn, request: Request, response: Response, ctx: Ctx, db: DB, ip: IP) -> dict[str, str]:
    _require_email(ctx)
    email = body.email.lower()
    acct_key, ip_key = ctx.keyring.mac(email), ctx.keyring.mac(ip)
    hit(db, ctx.window("login_ip", ctx.settings.login_max_per_ip_window), ip_key)
    check_locked(db, ctx.lock_account, acct_key)
    check_locked(db, ctx.lock_ip, ip_key)

    def fail() -> HTTPException:
        record_failure(db, ctx.lock_account, acct_key)
        record_failure(db, ctx.lock_ip, ip_key)
        return HTTPException(401, "invalid_credentials")

    user = _email_user(db, email)
    if not verify_password(user.password_hash if user else None, body.password) or user is None:
        raise fail()
    if not user.totp_enabled:
        # Verified address but TOTP never finished: only the setup endpoints accept this session.
        start_session(ctx, db, request, response, user, mfa_ok=False)
        return {"status": "totp_setup_required"}
    if body.totp:
        step = match_totp(ctx.keyring.decrypt(user.totp_secret or ""), body.totp, user.totp_last_step)
        # Conditional update: two requests racing with the same code can't both win.
        if step is None or not affected(
            db.execute(update(User).where(User.id == user.id, User.totp_last_step < step).values(totp_last_step=step))
        ):
            db.rollback()
            raise fail()
    elif body.recovery_code:
        code_hash = ctx.keyring.mac(normalize_recovery_code(body.recovery_code))
        rc = db.scalar(
            select(RecoveryCode).where(
                RecoveryCode.user_id == user.id, RecoveryCode.code_hash == code_hash, RecoveryCode.used_at.is_(None)
            )
        )
        if rc is None or not affected(
            db.execute(
                update(RecoveryCode)
                .where(RecoveryCode.id == rc.id, RecoveryCode.used_at.is_(None))
                .values(used_at=utcnow())
            )
        ):
            db.rollback()
            raise fail()
        logger.info("recovery code used user=%s", user.id)
    else:
        raise fail()
    if user.password_hash and password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    start_session(ctx, db, request, response, user, mfa_ok=True)
    logger.info("login ok provider=email user=%s", user.id)
    return {"status": "ok"}


def _setup_user(current: AnySession) -> User:
    user = current.user
    if user.provider != "email":
        raise HTTPException(404, "not_found")
    if user.totp_enabled:
        raise HTTPException(409, "totp_already_enabled")
    return user


@router.post("/totp/setup")
def totp_setup(current: AnySession, ctx: Ctx, db: DB) -> dict[str, str]:
    _require_email(ctx)
    user = _setup_user(current)
    hit(db, ctx.window("totp_setup", 10, 3600), user.id)
    secret = new_totp_secret()
    db.execute(update(User).where(User.id == user.id).values(totp_pending=ctx.keyring.encrypt(secret)))
    db.commit()
    uri = totp_uri(secret, user.email or user.subject, ctx.settings.totp_issuer)
    # SVG completo (com xmlns): o site mostra num <img> via data: URI, e sem o namespace o navegador recusa.
    buf = io.BytesIO()
    segno.make(uri, error="m").save(buf, kind="svg", xmldecl=False, scale=4, dark="#111", light="#fff")
    svg = buf.getvalue().decode()
    return {"otpauth_uri": uri, "qr_svg": svg}


@router.post("/totp/confirm")
def totp_confirm(
    body: CodeIn, request: Request, response: Response, current: AnySession, ctx: Ctx, db: DB, ip: IP
) -> dict[str, Any]:
    _require_email(ctx)
    user = _setup_user(current)
    acct_key, ip_key = ctx.keyring.mac(user.subject), ctx.keyring.mac(ip)
    check_locked(db, ctx.lock_account, acct_key)
    check_locked(db, ctx.lock_ip, ip_key)
    pending = db.scalar(select(User.totp_pending).where(User.id == user.id))
    step = match_totp(ctx.keyring.decrypt(pending), body.code, 0) if pending else None
    if step is None:
        record_failure(db, ctx.lock_account, acct_key)
        record_failure(db, ctx.lock_ip, ip_key)
        raise HTTPException(400, "invalid_code")
    codes = recovery_codes()
    user.totp_secret, user.totp_pending, user.totp_enabled, user.totp_last_step = pending, None, True, step
    db.execute(delete(RecoveryCode).where(RecoveryCode.user_id == user.id))
    db.add_all(RecoveryCode(user_id=user.id, code_hash=ctx.keyring.mac(normalize_recovery_code(c))) for c in codes)
    db.flush()
    start_session(ctx, db, request, response, user, mfa_ok=True)
    logger.info("totp enabled user=%s", user.id)
    return {"status": "ok", "recovery_codes": codes}


# ---------------------------------------------------------------- /api/me helpers


def describe(current: MaybeCurrent) -> tuple[dict[str, Any] | None, str | None]:
    if current is None:
        return None, None
    if not current.session.mfa_ok:
        return None, "totp_setup"
    u = current.user
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "provider": u.provider,
        "mfa": u.totp_enabled,
    }, None
