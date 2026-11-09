"""Transactional email through the Resend HTTP API (plain text only)."""

from __future__ import annotations

import logging

import httpx2

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


class Mailer:
    def __init__(self, api_key: str, sender: str, http: httpx2.Client) -> None:
        self._key, self._from, self._http = api_key, sender, http

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    def send(self, to: str, subject: str, text: str, reply_to: str | None = None) -> bool:
        if not self._key or not to:
            return False
        body: dict[str, object] = {
            "from": self._from,
            "to": [to],
            "subject": " ".join(subject.split())[:200],
            "text": text,
        }
        if reply_to:
            body["reply_to"] = reply_to
        try:
            resp = self._http.post(RESEND_URL, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=10)
        except httpx2.HTTPError as exc:
            logger.warning("resend request failed: %s", type(exc).__name__)
            return False
        if resp.status_code >= 300:
            logger.warning("resend refused the message: status %s", resp.status_code)
            return False
        return True
