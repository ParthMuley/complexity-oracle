"""Complexity Oracle — API middleware.

Rate limiter using SlowAPI (built on the `limits` library).

Limit precedence (first match wins):
  1. RATE_LIMIT env var         — explicit override
  2. CLOUD_MODE=true            — defaults to "10/minute" (public endpoint protection)
  3. No env vars                — defaults to "1000/minute" (dev / local use)

The high default for local use means tests and development never hit the limit.
Set RATE_LIMIT=10/minute (or any limits-format string) in your deployment config.
"""
from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# Resolve effective rate limit at import time so it reads env vars set before startup.
def _effective_limit() -> str:
    explicit = os.environ.get("RATE_LIMIT")
    if explicit:
        return explicit
    if os.environ.get("CLOUD_MODE"):
        return "10/minute"
    return "1000/minute"


RATE_LIMIT: str = _effective_limit()

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
