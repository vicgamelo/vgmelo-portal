"""Rate limits, lockouts, per-user daily quotas and token budgets (global and per user).

Every counter is updated with a single conditional UPDATE (or an INSERT when the row is new), so two
concurrent requests can't both take the last unit, on SQLite and on Postgres alike.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import BudgetAlert, RateCounter, TokenBudget, Usage, UserTokens, affected

DEMOS = ("agents", "score", "mcp", "traces", "studio")


class TooMany(HTTPException):
    def __init__(self, reason: str, retry_at: datetime) -> None:
        super().__init__(429, {"reason": reason, "retry_at": retry_at.isoformat(timespec="seconds")})
        self.headers = {"Retry-After": str(max(1, int((retry_at - datetime.now(UTC)).total_seconds())))}


@dataclass(frozen=True)
class Window:
    bucket: str
    seconds: int
    limit: int

    def start(self, now: float | None = None) -> int:
        t = int(now if now is not None else time.time())
        return t - t % self.seconds

    def retry_at(self, now: float | None = None) -> datetime:
        return datetime.fromtimestamp(self.start(now) + self.seconds, UTC)


def _count(db: Session, w: Window, key: str) -> int:
    row = db.get(RateCounter, (w.bucket, key, w.start()))
    return row.count if row else 0


def hit(db: Session, w: Window, key: str, reason: str = "rate_limited") -> None:
    """Count one event; raise 429 if this one goes over the limit."""
    start = w.start()
    for _ in range(3):
        res = db.execute(
            update(RateCounter)
            .where(
                RateCounter.bucket == w.bucket,
                RateCounter.key == key,
                RateCounter.window_start == start,
                RateCounter.count < w.limit,
            )
            .values(count=RateCounter.count + 1)
        )
        if affected(res):
            db.commit()
            return
        if db.get(RateCounter, (w.bucket, key, start)) is not None:
            db.rollback()
            raise TooMany(reason, w.retry_at())
        try:
            db.add(RateCounter(bucket=w.bucket, key=key, window_start=start, count=1))
            db.commit()
            return
        except IntegrityError:
            db.rollback()
    raise TooMany(reason, w.retry_at())


def check_locked(db: Session, w: Window, key: str) -> None:
    if _count(db, w, key) >= w.limit:
        raise TooMany("locked", w.retry_at())


def record_failure(db: Session, w: Window, key: str) -> None:
    with contextlib.suppress(TooMany):  # already at the limit; the next check_locked refuses
        hit(db, w, key, reason="locked")


def purge_old_counters(db: Session, older_than_s: int = 2 * 86400) -> None:
    db.execute(delete(RateCounter).where(RateCounter.window_start < int(time.time()) - older_than_s))


class Calendar:
    """Quota days follow QUOTA_TZ so "come back tomorrow" means the visitor's tomorrow."""

    def __init__(self, tz: str) -> None:
        self.tz = ZoneInfo(tz)

    def today(self) -> str:
        return datetime.now(self.tz).date().isoformat()

    def next_reset(self) -> datetime:
        now = datetime.now(self.tz)
        return datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), self.tz)


def take_quota(db: Session, user_id: str, day: str, demo: str, limit: int, reset: datetime) -> None:
    for _ in range(3):
        res = db.execute(
            update(Usage)
            .where(Usage.user_id == user_id, Usage.day == day, Usage.demo == demo, Usage.count < limit)
            .values(count=Usage.count + 1)
        )
        if affected(res):
            db.commit()
            return
        if limit <= 0 or db.get(Usage, (user_id, day, demo)) is not None:
            db.rollback()
            raise TooMany("daily_quota", reset)
        try:
            db.add(Usage(user_id=user_id, day=day, demo=demo, count=1))
            db.commit()
            return
        except IntegrityError:
            db.rollback()
    raise TooMany("daily_quota", reset)


def refund_quota(db: Session, user_id: str, day: str, demo: str) -> None:
    db.execute(
        update(Usage)
        .where(Usage.user_id == user_id, Usage.day == day, Usage.demo == demo, Usage.count > 0)
        .values(count=Usage.count - 1)
    )
    db.commit()


def used_today(db: Session, user_id: str, day: str) -> dict[str, int]:
    rows = db.execute(select(Usage.demo, Usage.count).where(Usage.user_id == user_id, Usage.day == day))
    return {demo: count for demo, count in rows}


def tokens_used(db: Session, day: str) -> int:
    return int(db.scalar(select(func.coalesce(func.sum(TokenBudget.tokens), 0)).where(TokenBudget.day == day)) or 0)


def check_budget(db: Session, day: str, budget: int, reset: datetime) -> None:
    if tokens_used(db, day) >= budget:
        raise TooMany("global_budget", reset)


def user_tokens_used(db: Session, user_id: str, day: str) -> int:
    row = db.get(UserTokens, (user_id, day))
    return row.tokens if row else 0


def check_user_tokens(db: Session, user_id: str, day: str, limit: int, reset: datetime) -> None:
    """A run starts only with balance left; it may end a little over, since tokens are known at the end."""
    if user_tokens_used(db, user_id, day) >= limit:
        raise TooMany("user_tokens", reset)


def _bump(db: Session, model: type[TokenBudget] | type[UserTokens], keys: dict[str, str], tokens: int) -> None:
    for _ in range(3):
        cond = [getattr(model, k) == v for k, v in keys.items()]
        res = db.execute(update(model).where(*cond).values(tokens=model.tokens + tokens))
        if affected(res):
            db.commit()
            return
        try:
            db.add(model(**keys, tokens=tokens))
            db.commit()
            return
        except IntegrityError:
            db.rollback()


def add_tokens(db: Session, day: str, tokens: int, user_id: str | None = None) -> None:
    """Charges the global budget and, given ``user_id``, that user's daily tokens."""
    if tokens <= 0:
        return
    _bump(db, TokenBudget, {"day": day}, tokens)
    if user_id:
        _bump(db, UserTokens, {"user_id": user_id, "day": day}, tokens)


def crossed_alerts(db: Session, day: str, budget: int, percents: list[int]) -> list[tuple[int, int]]:
    """Thresholds the day's tokens just crossed and nobody has claimed yet, as (percent, tokens used).

    Claiming is an insert on (day, percent): two runs ending together can't both send the same alert."""
    if budget <= 0 or not percents:
        return []
    used = tokens_used(db, day)
    claimed: list[tuple[int, int]] = []
    for pct in sorted(set(percents)):
        if used * 100 < budget * pct:
            break
        if db.get(BudgetAlert, (day, pct)) is not None:
            continue
        try:
            db.add(BudgetAlert(day=day, percent=pct))
            db.commit()
            claimed.append((pct, used))
        except IntegrityError:
            db.rollback()
    return claimed
