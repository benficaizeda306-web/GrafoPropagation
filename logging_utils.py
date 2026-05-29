"""
GrafoPropagation v26-APEX — Logging Utilities

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.markup import escape as _esc
    console = Console(width=180)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 90)
    console = _FallbackConsole()

_LOG_BUF: list = []
RUN_ID = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

_STYLE = {
    "INFO": "dim",
    "WARN": "yellow",
    "ERROR": "bold red",
    "METRIC": "bold green",
    "ATTN": "bold magenta",
    "SYS2": "bold cyan",
    "MCTS": "bold yellow",
}


def _ts() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def log(msg: str, level: str = "INFO"):
    """Append to buffer and print with Rich styling (or plain fallback)."""
    _LOG_BUF.append({"ts": _ts(), "run": RUN_ID, "lvl": level, "msg": msg})
    s = _STYLE.get(level, "")
    if HAS_RICH:
        safe = _esc(msg)
        console.print(
            f"[{s}][{level}] {safe}[/{s}]" if s else f"[{level}] {safe}"
        )
    else:
        print(f"[{level}] {msg}")
