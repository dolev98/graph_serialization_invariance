"""Per-tuple rules of the paper's metrics (scripts/paper_metrics.py).

A tuple is one model's answers, in one mode, to the reference serialization of an instance and one
factor's variants. `vals` holds one comparison value per serialization: "CORRECT" for a correct answer,
the answer key for a wrong one, "NA" for no answer.
"""
from __future__ import annotations

import math
from collections import Counter

READING_TASKS = ("connected_nodes", "disconnected_nodes", "neighbor")
# M3 never sees the graph text, so its structure and syntax answers repeat its reference answer; accuracy over
# these three axes scores every mode on the serializations that can change what it computes
LIKE_FOR_LIKE = ("canonical", "relabel", "order")


def divergence_site(vals: list, bad_copy: list[bool], all_answered: bool) -> str:
    """Where a flipped code tuple diverges: "copy", "crash" or "program".

    A wrong copy (a serialization whose declared graph is wrong) is blamed when its value is not the most
    common value among the correct copies. When several values tie for most common, none of them counts as
    the divergence, so the result does not depend on the order of the serializations.
    """
    good = Counter(v for v, b in zip(vals, bad_copy) if not b)
    top = {v for v, c in good.items() if c == max(good.values())} if good else set()
    if any(b and v not in top for v, b in zip(vals, bad_copy)):
        return "copy"
    return "program" if all_answered else "crash"


def failure_step(failure_class: str, declared_ok, correct: bool) -> str | None:
    """The first step that failed for a code answer, along copy -> solve -> run; None if it is correct.

    A program that declared a wrong graph failed at the copy even if it then crashed. "run" is every answer that
    gives no answer (the paper's crash): the program crashed or timed out ("execution"), or there was no program or
    no parseable `ans` ("format"), as localization (b) and the flip rule count it.
    """
    if correct:
        return None
    if declared_ok is False or failure_class == "transcription":
        return "copy"
    return {"execution": "run", "format": "run", "logic": "solve", "construction": "solve"}.get(failure_class, "other")


def spread(values: list[float], truth: float | None) -> tuple[float | None, bool | None]:
    """S = (max - min) / |truth| over a flip's answered numeric values (divided by 1 when the truth is 0),
    and whether the largest value is exactly double the smallest positive one. (None, None) when fewer than
    two distinct values or no numeric truth."""
    if len(set(values)) < 2 or truth is None:
        return None, None
    s = (max(values) - min(values)) / (abs(truth) if truth != 0 else 1.0)
    return s, min(values) > 0 and abs(max(values) - 2 * min(values)) < 1e-9


def reading_tag(task: str, answers: list[set], base: set, other: set | None) -> str:
    """Does a flipped node-list tuple take the question's other reading?

    `base` is the true answer's reading and `other` the alternative: for GraphQA's "connected to" questions
    adjacency vs reachability; for Erdős `neighbor` there is no alternative set and the test is whether some
    answer is a strict subset of the true neighbours, as a directed reading of an undirected graph gives.
    """
    if task == "neighbor":
        return "strict subset of true neighbours" if any(s < base for s in answers) else "other"
    if other == base:
        return "readings coincide"
    return "takes the other reading" if any(s == other for s in answers) else "other"


def like_for_like(axis: str) -> bool:
    """Whether an answer counts in the like-for-like accuracy: the reference, a relabeling or an edge order."""
    return axis in LIKE_FOR_LIKE


def _number(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_double(answer, truth) -> bool:
    """Whether a numeric answer is exactly twice a positive true value, as when a degree is counted once for
    each direction of an edge. Booleans and non-numbers are never doubles."""
    a, t = _number(answer), _number(truth)
    return a is not None and t is not None and t > 0 and abs(a - 2 * t) < 1e-9


def durkalski(b: list[int], c: list[int]) -> tuple[float, float]:
    """McNemar's test for pairs clustered in groups (Durkalski et al., 2003): b[k] and c[k] count cluster k's
    discordant pairs of each kind. X = (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2 is chi-square with 1 degree
    of freedom; returns (X, p). A cluster whose discordant pairs balance adds nothing; (0.0, 1.0) when all do."""
    d = [x - y for x, y in zip(b, c)]
    den = sum(v * v for v in d)
    if den == 0:
        return 0.0, 1.0
    x = sum(d) ** 2 / den
    return x, math.erfc(math.sqrt(x / 2))  # upper tail of chi-square(1)


def crash_only_flip(keys: list[str], variant_ids: list[str], suffix: str) -> bool:
    """Whether a flipped tuple flips only because one serialization (the variant whose id ends in `suffix`) gave no
    answer ("NA") while every other one answered. Section 5.2 counts NetworkX code's syntax flips this way."""
    target = [k for k, v in zip(keys, variant_ids) if v.endswith(suffix)]
    return target == ["NA"] and keys.count("NA") == 1
