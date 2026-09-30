import networkx as nx
import pytest

from gsi.llm.client import extract_code
from gsi.score.compare import compare, normalize_answer, relabel_answer
from gsi.score.parse_answer import find_boxed_answer, parse_value


@pytest.mark.parametrize("text,expected", [
    ("Some reasoning.\nTherefore, the final answer is: $\\boxed{4}$.", "4"),
    ("\\boxed{[1, 2, 3]}", "[1, 2, 3]"),
    ("\\boxed{\\text{Yes}}", "Yes"),
    ("first \\boxed{1} then \\boxed{7}", "7"),
    ("no box here. The final answer is: 9.", "9"),
    ("the answer is [2, 5]", "[2, 5]"),
    ("I am not certain about this one.", None),
    ("", None),
])
def test_find_boxed_answer(text, expected):
    assert find_boxed_answer(text) == expected


@pytest.mark.parametrize("s,t,expected", [
    ("Yes", "bool", True), ("no, there is none", "bool", False), ("maybe", "bool", None),
    ("4", "int", 4), ("4.0", "int", 4), ("4.5", "int", None), ("degree 3", "int", 3),
    ("0.25", "float", 0.25), ("7", "node", 7),
    ("[2, 5, 7]", "node_set", [2, 5, 7]), ("[]", "node_set", []), ("None", "node_set", []),
    ("nodes 1 and 3", "node_set", [1, 3]),
    ("[7, 2, 5]", "path", [7, 2, 5]),
    ("[(1, 2), (2,3)]", "edge_set", [[1, 2], [2, 3]]), ("[]", "edge_set", []), ("nothing", "edge_set", None),
])
def test_parse_value(s, t, expected):
    assert parse_value(s, t) == expected


def test_compare_types():
    assert compare(True, True, "bool") and not compare(False, True, "bool")
    assert compare(4, 4, "int") and not compare(5, 4, "int")
    assert compare(0.2549, 0.25, "float") and not compare(0.3, 0.25, "float")
    assert compare([7, 2], [2, 7], "node_set") and not compare([7], [2, 7], "node_set")
    assert compare([[2, 1], [3, 2]], [[1, 2], [2, 3]], "edge_set")
    assert not compare(None, 4, "int")


def test_path_validity_with_ties():
    G = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3)])
    assert compare([0, 1, 3], [0, 2, 3], "path", G=G)
    assert not compare([0, 3], [0, 2, 3], "path", G=G)
    assert not compare([0, 1, 2, 3], [0, 2, 3], "path", G=G)


def test_relabel_answer_roundtrip():
    mapping = {0: 5, 1: 3, 2: 9}
    inv = {v: k for k, v in mapping.items()}
    for t, val in [("node", 1), ("node_set", [0, 2]), ("path", [0, 1, 2]), ("edge_set", [[0, 1], [1, 2]])]:
        assert relabel_answer(relabel_answer(val, t, mapping), t, inv) == val
    assert relabel_answer(4, "int", mapping) == 4


def test_normalize_answer():
    assert normalize_answer([3, 1], "node_set") == (1, 3)
    assert normalize_answer([[3, 1], [1, 3]], "edge_set") == ((1, 3),)
    assert normalize_answer([[3, 1], [1, 3]], "edge_set", directed=True) == ((1, 3), (3, 1))
    assert normalize_answer(0.123456, "float") == 0.1235


def test_extract_code_uses_codegraph_markers_only():
    assert extract_code("Sure:\n# CODE START\nans = 1\n# CODE END\nDone.") == "ans = 1"
    assert extract_code("# code start\nans = 2\n#CODE  END") == "ans = 2"
    assert extract_code("```python\nans = 3\n```") is None
    assert extract_code("import networkx as nx\nans = 4") is None
    assert extract_code("") is None
