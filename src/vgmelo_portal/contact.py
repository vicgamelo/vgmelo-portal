"""Contact form: stored in the database, then a plain-text notice to CONTACT_TO."""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .context import DB, IP, Ctx
from .db import ContactMessage
from .limits import hit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

KINDS = {"job": "Vaga / job", "consulting": "Consultoria / consulting", "other": "Outro / other"}


class ContactIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr = Field(max_length=254)
    kind: Literal["job", "consulting", "other"]
    message: str = Field(min_length=10, max_length=5000)
    website: str = Field(default="", max_length=200)  # honeypot: humans never see it
    started_at: int = Field(description="epoch milliseconds when the form was rendered")


@router.post("/contact")
def contact(body: ContactIn, ctx: Ctx, db: DB, ip: IP) -> dict[str, str]:
    hit(db, ctx.window("contact_ip", ctx.settings.contact_max_per_ip_hour, 3600), ctx.keyring.mac(ip))
    elapsed = time.time() - body.started_at / 1000
    # Bots get the same answer as people, so they don't learn which check caught them.
    if body.website or not ctx.settings.contact_min_fill_seconds <= elapsed <= 86400:
        logger.info("contact dropped (honeypot or timing)")
        return {"status": "sent"}
    row = ContactMessage(name=body.name, email=body.email.lower(), kind=body.kind, message=body.message)
    db.add(row)
    db.commit()
    text = f"Nome / name: {body.name}\nE-mail: {body.email}\nAssunto / kind: {KINDS[body.kind]}\n\n{body.message}\n"
    row.notified = ctx.mailer.send(
        ctx.settings.contact_to, f"[vgmelo.tech] {KINDS[body.kind]}: {body.name}", text, reply_to=body.email
    )
    db.commit()
    logger.info("contact stored id=%s notified=%s", row.id, row.notified)
    return {"status": "sent"}
