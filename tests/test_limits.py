from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from vgmelo_portal.db import Usage, User, make_engine, make_sessionmaker
from vgmelo_portal.limits import TooMany, Window, add_tokens, hit, take_quota, tokens_used


@pytest.fixture
def real_url(db_url: str, tmp_path: Path) -> str:
    # In-memory SQLite is one shared connection; a file gives real locking between threads.
    return f"sqlite:///{tmp_path / 'race.db'}" if db_url.startswith("sqlite") else db_url


def test_quota_holds_under_concurrency(real_url: str) -> None:
    Session = make_sessionmaker(make_engine(real_url))
    with Session() as db:
        db.add(User(id="u1", provider="github", subject="1"))
        db.commit()
    reset = datetime.now(UTC)

    def attempt(_: int) -> bool:
        with Session() as db:
            try:
                take_quota(db, "u1", "2026-11-09", "agents", 5, reset)
                return True
            except TooMany:
                return False

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(attempt, range(40)))
    assert sum(results) == 5
    with Session() as db:
        assert db.scalar(select(Usage.count)) == 5


def test_rate_window_holds_under_concurrency(real_url: str) -> None:
    Session = make_sessionmaker(make_engine(real_url))
    w = Window("t", 3600, 7)
    key = secrets.token_hex(8)

    def attempt(_: int) -> bool:
        with Session() as db:
            try:
                hit(db, w, key)
                return True
            except TooMany:
                return False

    with ThreadPoolExecutor(8) as pool:
        assert sum(pool.map(attempt, range(30))) == 7


def test_tokens_add_up(real_url: str) -> None:
    Session = make_sessionmaker(make_engine(real_url))

    def add(_: int) -> None:
        with Session() as db:
            add_tokens(db, "2026-11-09", 10)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(add, range(50)))
    with Session() as db:
        assert tokens_used(db, "2026-11-09") == 500
