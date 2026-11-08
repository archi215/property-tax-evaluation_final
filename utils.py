"""utils.py -- small, dependency-free helpers (no Streamlit)."""
from __future__ import annotations


def fmt_inr(value, symbol: str = "\u20b9") -> str:
    """Indian digit grouping: 1,23,45,678."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    sign = "-" if v < 0 else ""
    s = f"{abs(v):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return f"{sign}{symbol}{s}"


def fmt_compact(value, symbol: str = "\u20b9") -> str:
    """\u20b91.25 Cr / \u20b945.3 L / \u20b99,500."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}{symbol}{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}{symbol}{a / 1e5:.2f} L"
    return fmt_inr(v, symbol)


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
