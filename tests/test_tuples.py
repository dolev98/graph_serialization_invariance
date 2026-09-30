"""The per-tuple rules behind the paper's metrics (gsi.analysis.tuples)."""
import math
from itertools import permutations

import pytest

from gsi.analysis.tuples import (crash_only_flip, divergence_site, durkalski, failure_step, is_double, like_for_like,
                                 reading_tag, spread)


def test_divergence_site_does_not_depend_on_serialization_order():
    # two correct copies disagree 1-1 and a wrong copy agrees with one of them: no majority to diverge from
    vals, bad = ["CORRECT", "7", "7"], [False, False, True]
    sites = {divergence_site([vals[i] for i in p], [bad[i] for i in p], True) for p in permutations(range(3))}
    assert sites == {"program"}


def test_divergence_site_blames_a_wrong_copy_that_left_the_majority():
    assert divergence_site(["CORRECT", "CORRECT", "5"], [False, False, True], True) == "copy"
    assert divergence_site(["CORRECT", "CORRECT", "CORRECT", "5"], [False, False, True, False], True) == "program"
    assert divergence_site(["CORRECT", "NA"], [False, False], False) == "crash"
    assert divergence_site(["5", "6"], [True, True], True) == "copy"  # no correct copy to compare against


def test_failure_step_assigns_the_first_step_that_failed():
    assert failure_step("execution", False, False) == "copy"  # wrong copy, then a crash
    assert failure_step("execution", True, False) == "run"
    assert failure_step("execution", None, False) == "run"
    assert failure_step("transcription", False, False) == "copy"
    assert failure_step("logic", True, False) == "solve"
    assert failure_step("construction", True, False) == "solve"
    assert failure_step("format", True, False) == "run"  # no program, or no parseable ans: no answer
    assert failure_step("format", None, False) == "run"
    assert failure_step("unverifiable", None, False) == "other"
    assert failure_step("no_computation", True, True) is None  # a correct answer is not a failure
    assert failure_step("ok", False, True) is None


def test_spread():
    assert spread([3.0, 6.0], 3.0) == (1.0, True)
    assert spread([0.0, 2.0], 0.0) == (2.0, False)
    assert spread([4.0, 4.0], 4.0) == (None, None)
    assert spread([1.0, 2.0], None) == (None, None)


def test_reading_tag():
    base, other = {1, 2}, {1, 2, 3}
    assert reading_tag("connected_nodes", [{1, 2}, {1, 2, 3}], base, other) == "takes the other reading"
    assert reading_tag("connected_nodes", [{1, 2}, {1}], base, other) == "other"
    assert reading_tag("disconnected_nodes", [{4}, {5}], {4}, {4}) == "readings coincide"
    assert reading_tag("neighbor", [{1}, {1, 2}], {1, 2}, None) == "strict subset of true neighbours"
    assert reading_tag("neighbor", [{1, 2}, {1, 2, 3}], {1, 2}, None) == "other"


def test_like_for_like_keeps_the_reference_relabelings_and_orders():
    assert [a for a in ["canonical", "relabel", "order", "structure", "syntax"] if like_for_like(a)] == \
        ["canonical", "relabel", "order"]


def test_is_double():
    assert is_double("4", 2) and is_double(4.0, "2") and is_double("4.0", 2.0)
    assert not is_double("3", 2)
    assert not is_double("0", 0)  # a zero degree has no double
    assert not is_double(True, 0.5) and not is_double("null", 2) and not is_double(None, 2)
    assert not is_double("4", None)


def test_durkalski_statistic_and_p():
    # one cluster: (b - c)^2 / (b - c)^2 = 1, whatever the imbalance
    assert durkalski([5], [0]) == pytest.approx((1.0, math.erfc(math.sqrt(0.5))))
    # clusters that disagree in direction shrink the statistic below the unclustered McNemar's (b - c)^2 / (b + c)
    x, p = durkalski([3, 0, 2, 4], [0, 2, 0, 1])
    assert x == pytest.approx(36 / 26) and p == pytest.approx(math.erfc(math.sqrt(36 / 52)))
    assert x < (9 - 3) ** 2 / (9 + 3)
    # without clustering (one pair per cluster) it is McNemar's statistic without continuity correction
    b, c = [1] * 12 + [0] * 4, [0] * 12 + [1] * 4
    assert durkalski(b, c)[0] == pytest.approx((12 - 4) ** 2 / (12 + 4))
    assert durkalski([2, 1], [2, 1]) == (0.0, 1.0)  # every cluster balanced
    assert durkalski([], []) == (0.0, 1.0)


def test_durkalski_p_is_the_chi_square_tail():
    assert durkalski([1] * 3, [0] * 3)[1] == pytest.approx(0.0832645, abs=1e-6)  # X = 3
    stats = pytest.importorskip("scipy.stats")
    for b, c in [([4, 0, 3], [1, 2, 0]), ([7, 1], [0, 5])]:
        x, p = durkalski(b, c)
        assert p == pytest.approx(stats.chi2.sf(x, 1))


def test_crash_only_flip():
    ids = ["i::canonical", "i::syntax:plain", "i::syntax:json", "i::syntax:networkx_code"]
    nx = "::syntax:networkx_code"
    assert crash_only_flip(["3", "3", "3", "NA"], ids, nx)
    assert not crash_only_flip(["3", "NA", "3", "NA"], ids, nx)  # another serialization gave no answer too
    assert not crash_only_flip(["3", "4", "3", "5"], ids, nx)    # the NetworkX-code serialization answered
    assert not crash_only_flip(["3", "3"], ids[:2], nx)           # no NetworkX-code serialization in the tuple
