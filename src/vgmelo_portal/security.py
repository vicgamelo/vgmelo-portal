"""Small security primitives: tokens, hashes, TOTP, client IP, redirect targets."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import re
import secrets
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .config import Network

_hasher = PasswordHasher()
# Verified against when the account doesn't exist, so both paths cost one argon2 run.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

TOTP_STEP = 30
TOTP_WINDOW = 1  # ±1 step: a code lives at most ~90s


def new_token() -> str:
    return secrets.token_urlsafe(32)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Keyring:
    """Everything derived from SECRET_KEY."""

    def __init__(self, secret_key: str) -> None:
        def derive(info: bytes) -> bytes:
            return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(secret_key.encode())

        self._mac_key = derive(b"vgmelo-portal/mac")
        self._aead = AESGCM(derive(b"vgmelo-portal/totp"))

    def mac(self, value: str) -> str:
        """Keyed hash for low-entropy values (6-digit codes, emails, IPs)."""
        return hmac.new(self._mac_key, value.encode(), hashlib.sha256).hexdigest()

    def encrypt(self, plaintext: str) -> str:
        nonce = secrets.token_bytes(12)
        return base64.urlsafe_b64encode(nonce + self._aead.encrypt(nonce, plaintext.encode(), None)).decode()

    def decrypt(self, token: str) -> str:
        raw = base64.urlsafe_b64decode(token.encode())
        return self._aead.decrypt(raw[:12], raw[12:], None).decode()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored: str | None, password: str) -> bool:
    try:
        return _hasher.verify(stored or _DUMMY_HASH, password) and stored is not None
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(stored: str) -> bool:
    return _hasher.check_needs_rehash(stored)


def six_digit_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def recovery_codes(n: int = 10) -> list[str]:
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    raw = ["".join(secrets.choice(alphabet) for _ in range(10)) for _ in range(n)]
    return [f"{c[:5]}-{c[5:]}" for c in raw]


def normalize_recovery_code(code: str) -> str:
    return re.sub(r"[^a-z0-9]", "", code.lower())


def new_totp_secret() -> str:
    return pyotp.random_base32(32)


def totp_uri(secret: str, account: str, issuer: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=account, issuer_name=issuer)


def match_totp(secret: str, code: str, last_step: int, now: float | None = None) -> int | None:
    """Return the matched time step, or None. Steps at or before ``last_step`` are refused (no replay)."""
    if not re.fullmatch(r"\d{6}", code):
        return None
    totp = pyotp.TOTP(secret)
    current = int((now if now is not None else time.time()) // TOTP_STEP)
    for step in range(current - TOTP_WINDOW, current + TOTP_WINDOW + 1):
        if step > last_step and hmac.compare_digest(totp.at(step * TOTP_STEP), code):
            return step
    return None


def client_ip(peer: str | None, forwarded_for: str | None, trusted: Sequence[Network]) -> str:
    """The peer address, unless it is a trusted proxy: then the right-most untrusted hop of X-Forwarded-For."""

    def parse(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        try:
            return ipaddress.ip_address(value.strip())
        except ValueError:
            return None

    def is_trusted(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return any(addr in net for net in trusted)

    addr = parse(peer or "")
    if addr is None:
        return peer or "unknown"
    if not trusted or not is_trusted(addr) or not forwarded_for:
        return str(addr)
    for hop in reversed(forwarded_for.split(",")):
        hop_addr = parse(hop)
        if hop_addr is None:
            break  # garbage: stop at the last hop we could trust
        if not is_trusted(hop_addr):
            return str(hop_addr)
        addr = hop_addr
    return str(addr)


_SAFE_PATH = re.compile(r"^/(?![/\\])[A-Za-z0-9\-._~/%!$&'()*+,;=:@?]*$")


def safe_next(value: str | None) -> str:
    """Only same-site relative paths. Anything else (absolute, //host, /\\host, schemes) becomes '/'."""
    if not value or len(value) > 500 or not _SAFE_PATH.match(value):
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    return value.split("#", 1)[0]
