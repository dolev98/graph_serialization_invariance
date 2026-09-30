"""The prompt contract: the template, the minimal pair, and the library axis."""
import pytest

from gsi.data.base import GraphInstance
from gsi.prompts.modes import LIBRARIES, build_prompt, template
from gsi.serial.render import render
from gsi.serial.variant import canonical, make_variants

INST = GraphInstance(id="t-node_degree", dataset="graphqa", task="node_degree", directed=False, weighted=False,
                     nodes=[0, 1, 2, 3], edges=[[0, 1], [1, 2], [1, 3], [2, 3]], query_args={"node": 1},
                     answer_type="int", ground_truth=3)


def canon():
    v = canonical(INST, structure="edge_list", syntax="graphqa_nl")
    v.text = render(v)
    return v


# ---------------- the template ----------------

def test_template_has_both_slots_and_a_free_logic_slot():
    t = template(INST, inject=False)
    assert t.startswith("# CODE START") and t.endswith("# CODE END")
    assert "nodes = <" in t and "edges = <" in t and "<your code here>" in t


def test_inject_template_has_the_slots_filled_and_nothing_else_differs():
    m2, m3 = template(INST, inject=False), template(INST, inject=True)
    assert "nodes = <" not in m3 and "already defined" in m3
    tail = "\n\n<your code here>\nans = <the computed result, a Python int>\n# CODE END"
    assert m2.endswith(tail) and m3.endswith(tail)


def test_the_answer_slot_describes_a_computation_not_a_value():
    """`<your answer>` would invite `ans = 3`. The slot names a type, and the instruction
    says to compute it."""
    p = build_prompt("code", INST, canon(), "networkx")
    assert "<your answer>" not in p.user
    assert "ans = <the computed result, a Python int>" in p.user
    assert "Compute the result - do not write it directly." in p.user


# ---------------- M2 / M3 as a minimal pair ----------------

def test_m3_is_m2_minus_the_graph_and_the_declaration():
    v = canon()
    m2 = build_prompt("code", INST, v, "networkx").user
    m3 = build_prompt("graph_as_code", INST, v, "networkx").user
    assert v.text in m2 and v.text not in m3
    for shared in ("For this task, please calculate the degree", "# CODE START", "<your code here>",
                   "You may use the networkx library.", "Q: What is the degree of node 1?"):
        assert shared in m2 and shared in m3


def test_m3_prompt_is_identical_across_syntax_variants():
    """M3 shows no graph text, so a syntax perturbation cannot reach it -- the prompts are
    byte-identical and the response cache collapses them into one call."""
    canon_v = canon()
    variants = make_variants(INST, canon_v, {"syntax": {"kinds": ["json", "networkx_code"]}})
    for v in variants:
        v.text = render(v)
    hashes = {build_prompt("graph_as_code", INST, v, "networkx").prompt_hash for v in variants}
    assert len(hashes) == 1
    assert len({build_prompt("code", INST, v, "networkx").prompt_hash for v in variants}) == len(variants)


# ---------------- the library axis ----------------

def test_library_changes_only_the_toolkit_sentence():
    v = canon()
    nx_user = build_prompt("code", INST, v, "networkx").user
    native_user = build_prompt("code", INST, v, "native").user
    assert "You may use the networkx library." in nx_user
    assert "Do not import networkx" in native_user
    assert nx_user.replace("You may use the networkx library.", "X") == \
        native_user.replace("Use only plain Python and its standard library. Do not import networkx or any "
                            "other third-party library.", "X")


def test_library_is_a_field_not_part_of_the_mode_name():
    v = canon()
    prompts = [build_prompt("code", INST, v, lib) for lib in LIBRARIES]
    assert {p.mode for p in prompts} == {"code"}
    assert {p.library for p in prompts} == set(LIBRARIES)
    assert len({p.prompt_id for p in prompts}) == 2        # the arms cannot collide in a sink
    assert len({p.prompt_hash for p in prompts}) == 2      # nor in the response cache


def test_direct_takes_no_library_and_code_requires_one():
    v = canon()
    assert build_prompt("direct", INST, v).library is None
    with pytest.raises(ValueError):
        build_prompt("direct", INST, v, "networkx")
    with pytest.raises(ValueError):
        build_prompt("code", INST, v)
    with pytest.raises(ValueError):
        build_prompt("code", INST, v, "pytorch_geometric")


def test_user_suffix_reaches_the_wire_but_not_the_prompt(tmp_path):
    """A model-specific control token is appended at call time only: the Prompt, its hash and
    therefore the response log stay model-independent."""
    from gsi.llm.client import LLMClient, ModelSpec
    v = canon()
    p = build_prompt("code", INST, v, "networkx")
    plain = LLMClient(ModelSpec(name="a", model="x"), tmp_path)
    suffixed = LLMClient(ModelSpec(name="b", model="x", user_suffix="/no_think"), tmp_path)
    assert plain.messages(p)[1]["content"] == p.user
    assert suffixed.messages(p)[1]["content"] == p.user + " /no_think"
    assert build_prompt("code", INST, v, "networkx").prompt_hash == p.prompt_hash


def test_inject_graph_type_states_undirected_in_m3_only():
    from gsi.prompts.modes import UNDIRECTED_NOTE
    v = canon()
    assert not INST.directed
    base = build_prompt("graph_as_code", INST, v, "native")
    fixed = build_prompt("graph_as_code", INST, v, "native", inject_graph_type=True)
    assert UNDIRECTED_NOTE.strip() not in base.user
    assert UNDIRECTED_NOTE.strip() in fixed.user
    # same record identity, different cache identity
    assert fixed.prompt_id == base.prompt_id
    assert fixed.prompt_hash != base.prompt_hash
    # the sentence is the only difference
    assert fixed.user.replace(UNDIRECTED_NOTE, "") == base.user
    # M2 sees the graph text and is never touched by the flag
    m2 = build_prompt("code", INST, v, "native")
    assert build_prompt("code", INST, v, "native", inject_graph_type=True).user == m2.user
