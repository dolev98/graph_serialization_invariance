from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_INT_RE = re.compile(r"[+-]?\d+")
_PAIR_RE = re.compile(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")


_G1_FALLBACK = re.compile(r"the\s+(?:final\s+)?answer\s+is[:\s]*([^\n]+)", re.IGNORECASE)
_TEXT_WRAP = re.compile(r"\\text\{([^}]*)\}")


def _clean(s: str) -> str:
    s = _TEXT_WRAP.sub(r"\1", s.replace("$", ""))
    return s.strip().strip("*`_ \t").rstrip(".").strip()


def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def find_boxed_answer(text: str | None) -> str | None:
    """G1's extraction: the last \\boxed{...}, else the text after 'the [final] answer is'."""
    if not text:
        return None
    boxed = _last_boxed(text)
    if boxed is not None:
        return _clean(boxed)
    m = _G1_FALLBACK.findall(text)
    return _clean(m[-1]) if m else None


def _bracket_body(s: str) -> str | None:
    m = re.search(r"\[(.*?)\]", s, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{(.*?)\}", s, re.DOTALL)
    if m:
        return m.group(1)
    return None


def parse_value(s: str | None, answer_type: str) -> Any:
    """Return the typed value, or None when the text does not parse."""
    if s is None:
        return None
    s = s.strip()
    low = s.lower()
    if answer_type == "bool":
        if re.match(r"^(yes|true)\b", low):
            return True
        if re.match(r"^(no|false)\b", low):
            return False
        return None
    if answer_type in ("int", "node"):
        m = _NUM_RE.search(s)
        if not m:
            return None
        f = float(m.group())
        return int(f) if f.is_integer() else None
    if answer_type == "float":
        m = _NUM_RE.search(s)
        return float(m.group()) if m else None
    if answer_type in ("node_set", "path"):
        if low in ("[]", "{}", "none", "empty", "no nodes", "()"):
            return []
        body = _bracket_body(s)
        if body is None:
            body = s
        return [int(x) for x in _INT_RE.findall(body)]
    if answer_type == "edge_set":
        if low in ("[]", "{}", "none", "empty", "()"):
            return []
        pairs = _PAIR_RE.findall(s)
        return [[int(u), int(v)] for u, v in pairs] if pairs else None
    raise ValueError(f"unknown answer_type {answer_type}")
