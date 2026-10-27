"""Logging setup that keeps document contents and secrets out of log files.

Application code logs ids, counts and timings only. As a second safety net a
filter redacts anything that looks like an API key or bearer token.
"""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9._-]{10,}|api[_-]?key\s*[=:]\s*\S+)", re.IGNORECASE)
_CONFIGURED = False


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = _SECRET.sub("[REDACTED]", msg)
        if redacted != msg:
            record.msg, record.args = redacted, ()
        return True


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Configure the ``src`` logger once (Streamlit re-runs scripts often)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("src")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(fmt)
    handler.addFilter(RedactSecretsFilter())
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
