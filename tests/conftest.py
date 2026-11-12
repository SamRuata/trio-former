"""Shared fixtures.

TestClient is used as a CONTEXT MANAGER so FastAPI's startup runs. Model tests
are marked slow: they download weights on first use, which should not be the
default for a suite meant to run on every push.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def have_supabase() -> bool:
    return bool(os.environ.get("SUPABASE_URL")
                and os.environ.get("SUPABASE_SERVICE_KEY"))


needs_supabase = pytest.mark.skipif(
    not have_supabase(),
    reason="SUPABASE_URL / SUPABASE_SERVICE_KEY not set")

# Opt in with: pytest -m slow
slow = pytest.mark.slow


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: downloads pretrained weights and runs inference")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def supabase():
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"],
                         os.environ["SUPABASE_SERVICE_KEY"])    