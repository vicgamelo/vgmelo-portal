"""Read-only admin page: contact messages and demo usage. Server-rendered, every value escaped."""

from __future__ import annotations

from html import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from .context import DB, Ctx, MaybeCurrent
from .db import ContactMessage, DemoRun, TokenBudget, Usage, User

router = APIRouter()

ADMIN_CSP = "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; form-action 'none'; base-uri 'none'"

STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#111;background:#fff}
table{border-collapse:collapse;margin:8px 0 28px;width:100%}
td,th{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f4f4f4} pre{white-space:pre-wrap;margin:0;font:inherit}
"""


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td><pre>{escape(str(c) if c is not None else '')}</pre></td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body or '<tr><td>-</td></tr>'}</table>"


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin(current: MaybeCurrent, ctx: Ctx, db: DB) -> HTMLResponse:
    user = current.user if current and current.session.mfa_ok else None
    # 404 rather than 403: nothing here tells a visitor the page exists.
    if user is None or not user.email_verified or (user.email or "").lower() not in ctx.settings.admins:
        raise HTTPException(404, "Not Found")

    contacts = db.execute(
        select(ContactMessage.created_at, ContactMessage.kind, ContactMessage.name, ContactMessage.email,
               ContactMessage.message, ContactMessage.notified)
        .order_by(ContactMessage.created_at.desc()).limit(200)
    ).all()  # fmt: skip
    usage = db.execute(
        select(Usage.day, Usage.demo, func.sum(Usage.count), func.count(Usage.user_id))
        .group_by(Usage.day, Usage.demo).order_by(Usage.day.desc(), Usage.demo).limit(120)
    ).all()  # fmt: skip
    budget = db.execute(select(TokenBudget.day, TokenBudget.tokens).order_by(TokenBudget.day.desc()).limit(30)).all()
    runs = db.execute(
        select(DemoRun.created_at, User.provider, User.email, DemoRun.run_id, DemoRun.tokens)
        .join(User, User.id == DemoRun.user_id).order_by(DemoRun.created_at.desc()).limit(50)
    ).all()  # fmt: skip
    users = db.execute(select(User.provider, func.count()).group_by(User.provider)).all()

    html = (
        "<!doctype html><html lang=en><meta charset=utf-8><title>vgmelo-portal admin</title>"
        f"<style>{STYLE}</style><h1>Admin</h1>"
        f"<p>Token budget per day: {ctx.settings.global_daily_tokens}</p>"
        "<h2>Contacts</h2>"
        + _table(["when", "kind", "name", "email", "message", "notified"], [list(r) for r in contacts])
        + "<h2>Usage per day</h2>"
        + _table(["day", "demo", "calls", "users"], [list(r) for r in usage])
        + "<h2>Tokens per day</h2>"
        + _table(["day", "tokens"], [list(r) for r in budget])
        + "<h2>Recent agent runs</h2>"
        + _table(["when", "provider", "email", "run", "tokens"], [list(r) for r in runs])
        + "<h2>Users</h2>"
        + _table(["provider", "count"], [list(r) for r in users])
    )
    return HTMLResponse(html, headers={"Content-Security-Policy": ADMIN_CSP, "Cache-Control": "no-store"})
