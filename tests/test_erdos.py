from gsi.data.erdos import question, row_to_instance
from gsi.prompts.modes import build_prompt
from gsi.serial.render import render
from gsi.serial.variant import canonical, make_variants

PROMPT = ("The task is to determine the shortest path between two nodes of a weighted graph.\n\n"
          "The input nodes are guaranteed to be connected.\n\n"
          "Here is a directed graph containing nodes from 1 to 4. The edges are: (1, 2, 5), (2, 3, 1), (1, 3, 9), (3, 4, 2).\n\n"
          "Question: What is the shortest path between node 1 and node 4?\n\n"
          "You need to format your answer as a list of nodes, e.g., [node-1, node-2, ..., node-n].")

ROW = {"task": "weighted_shortest_path", "prompt": PROMPT, "answer": "[1, 2, 3, 4]", "direction": "directed",
       "nodes": "(1, 4)", "edges": "[(1, 2, 5), (2, 3, 1), (1, 3, 9), (3, 4, 2)]", "answer_type": "node_list"}


def test_row_to_instance_parses_prompt():
    inst = row_to_instance(ROW, 7)
    assert inst.directed and inst.weighted and inst.nodes == [1, 2, 3, 4]
    assert inst.edges == [[1, 2, 5], [2, 3, 1], [1, 3, 9], [3, 4, 2]]
    assert inst.query_args == {"u": 1, "v": 4} and inst.ground_truth == [1, 2, 3, 4] and inst.answer_type == "path"
    assert inst.meta["preamble"].startswith("The task is") and "format" not in inst.meta["question"]
    assert inst.meta["format_hint"] == "You need to format your answer as a list of nodes, e.g., [node-1, node-2, ..., node-n]."


def test_canonical_matches_native_prompt_and_relabel_propagates():
    inst = row_to_instance(ROW, 7)
    canon = canonical(inst, structure="edge_list", syntax="erdos_nl")
    canon.text = render(canon)
    assert canon.text in inst.native_prompt
    v = make_variants(inst, canon, {"relabel": {"seeds": 1}})[1]
    v.text = render(v)
    assert "nodes from 1 to 4" in v.text
    assert question(inst, v.label_map) == f"What is the shortest path between node {v.label_map[1]} and node {v.label_map[4]}?"
    p = build_prompt("code", inst, v, "networkx")
    assert p.user.startswith("The task is") and "The graph is directed" in p.user
    # exactly one template block, and the graph text is the last thing before "A:"
    # ("# CODE START" also appears inline in CodeGraph's instruction sentence)
    assert p.user.count("# CODE START\n") == 1 and p.user.rstrip().endswith("A:")
    assert v.text in p.user and p.library == "networkx"
    d = build_prompt("direct", inst, v)
    assert d.user.rstrip().endswith("$\\boxed{ANSWER}$.'") and inst.meta["format_hint"] in d.user


def test_weights_taken_from_prompt_not_column():
    row = dict(ROW, edges="[(1, 2), (2, 3), (1, 3), (3, 4)]")
    inst = row_to_instance(row, 0)
    assert inst.weighted and inst.edges[0] == [1, 2, 5]


def test_unsupported_task_is_skipped():
    assert row_to_instance(dict(ROW, task="topological_sort"), 0) is None
