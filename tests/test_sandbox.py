from gsi.data.base import GraphInstance
from gsi.exec import sandbox
from gsi.score.classify import classify
from gsi.score.compare import compare, relabel_answer
from gsi.score.parse_answer import parse_value
from gsi.serial.variant import canonical, relabel

# path 0-1-2-3 plus edge 1-3; degree of node 1 is 3
INST = GraphInstance(id="t-node_degree", dataset="graphqa", task="node_degree", directed=False, weighted=False,
                     nodes=[0, 1, 2, 3], edges=[[0, 1], [1, 2], [1, 3], [2, 3]], query_args={"node": 1},
                     answer_type="int", ground_truth=3)
CANON = canonical(INST, structure="edge_list", syntax="plain")

# The template as a model would fill it: the two declaration slots, then the solving code.
DECL = "nodes = [0, 1, 2, 3]\nedges = [(0,1),(1,2),(1,3),(2,3)]\n"
DECL_SHORT = "nodes = [0, 1, 2, 3]\nedges = [(0,1),(1,2),(2,3)]\n"      # one edge lost in transcription
BUILD = "import networkx as nx\nG = nx.Graph()\nG.add_nodes_from(nodes)\nG.add_edges_from(edges)\n"
BUILD_SHORT = "import networkx as nx\nG = nx.Graph()\nG.add_nodes_from(nodes)\nG.add_edges_from(edges[1:])\n"


def run_and_classify(code, v=CANON, mode="code", timeout=3):
    ex = sandbox.run(code, v, mode, timeout=timeout)
    parsed = parse_value(ex.ans, INST.answer_type)
    pred = relabel_answer(parsed, INST.answer_type, v.inverse_label_map)
    ok = compare(pred, INST.ground_truth, INST.answer_type)
    return ex, classify(mode, ok, parsed, ex.to_dict())


# ---------------- the three-way localization ----------------

def test_correct_program():
    ex, cls = run_and_classify(DECL + BUILD + "ans = G.degree(1)")
    assert cls == "ok" and ex.ans == "3"
    assert ex.declared_ok is True and ex.construction_ok is True and ex.graph_match is True


def test_non_ascii_source_runs():
    """Model comments can hold characters such as an arrow, which the Windows default encoding rejects."""
    ex, cls = run_and_classify(DECL + BUILD + "# degree " + chr(0x2192) + " neighbours\nans = G.degree(1)")
    assert cls == "ok" and ex.ans == "3"


def test_dropped_edge_in_the_declaration_is_transcription():
    ex, cls = run_and_classify(DECL_SHORT + BUILD + "ans = G.degree(1)")
    assert cls == "transcription" and ex.declared_ok is False


def test_declared_right_but_built_wrong_is_construction():
    """The slots hold the true graph, so transcription succeeded; the graph the code then
    built is missing an edge. Without this class the failure would be charged to logic."""
    ex, cls = run_and_classify(DECL + BUILD_SHORT + "ans = G.degree(1)")
    assert cls == "construction" and ex.declared_ok is True and ex.construction_ok is False


def test_wrong_algorithm_is_logic():
    ex, cls = run_and_classify(DECL + BUILD + "ans = G.degree(1) + 1")
    assert cls == "logic" and ex.declared_ok is True and ex.construction_ok is True


# ---------------- no_computation ----------------

def test_hardcoded_answer_is_no_computation_even_when_correct():
    ex, cls = run_and_classify(DECL + BUILD + "ans = 3")
    assert ex.ans == "3" and ex.ans_is_literal is True
    assert cls == "no_computation"      # correct, but nothing was computed


def test_literal_under_control_flow_is_computed_not_hardcoded():
    """The pilot's most common false positive: a default plus a conditional override. Which
    literal ends up in `ans` is decided by the program, so it is a computation."""
    code = DECL + "ans = False\nfor u, v in edges:\n    if u == 1 or v == 1:\n        ans = True\n"
    ex, cls = run_and_classify(code)
    assert ex.ans_is_literal is False and cls != "no_computation"
    code = DECL + "def f():\n    return True\nif f():\n    ans = 3\nelse:\n    ans = 0\n"
    ex, cls = run_and_classify(code)
    assert ex.ans_is_literal is False and cls == "ok"


def test_list_grown_by_append_is_an_accumulator():
    """`ans = []` is a literal, but `ans.append` inside a function mutates it in place -- the
    pilot's second false-positive shape (a bridges finder)."""
    code = DECL + "ans = []\ndef walk():\n    for u, v in edges:\n        if 1 in (u, v):\n            ans.append((u, v))\nwalk()\nans = len(ans)\n"
    ex, cls = run_and_classify(code)
    assert ex.ans_is_literal is False and cls == "ok"
    code = DECL + "ans = {}\nans['k'] = sum(1 for e in edges if 1 in e)\nans = ans['k']\n"
    ex, cls = run_and_classify(code)
    assert ex.ans_is_literal is False and cls == "ok"


def test_two_toplevel_literals_are_still_hardcoded():
    ex, cls = run_and_classify(DECL + BUILD + "ans = 2\nans = 3\n")
    assert ex.ans_is_literal is True and cls == "no_computation"


def test_accumulator_is_not_a_literal():
    code = DECL + "ans = 0\nfor e in edges:\n    ans += 1 if 1 in e else 0\n"
    ex, cls = run_and_classify(code)
    assert ex.ans_is_literal is False and ex.ans == "3" and cls == "ok"


# ---------------- crash-resilient declaration read ----------------

def test_declaration_survives_a_crash_in_the_solving_code():
    """The template assigns the slots first, so a program that transcribed correctly and
    then crashed still tells us the transcription was fine."""
    ex, cls = run_and_classify(DECL + BUILD + "ans = G.degree(1)\nraise ValueError('nope')")
    assert cls == "execution" and ex.exit_code == 1 and ex.ans is None
    assert ex.declared_nodes == [0, 1, 2, 3] and ex.declared_ok is True


def test_infinite_loop_times_out():
    ex, cls = run_and_classify(DECL + "while True:\n    pass", timeout=2)
    assert cls == "execution" and ex.timed_out and ex.wall_s < 10


# ---------------- off-template and format ----------------

def test_off_template_without_networkx_is_unverifiable():
    code = ("adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}\n"
            "counts = {len(adj): 2}\nans = counts[len(adj)]")
    ex, cls = run_and_classify(code)
    assert cls == "unverifiable" and ex.graphs == [] and ex.declared_ok is None


def test_off_template_with_networkx_still_localizes_via_the_recovered_graph():
    code = "import networkx as nx\nG = nx.Graph([(0,1),(1,2),(2,3)])\nans = G.degree(1)"
    ex, cls = run_and_classify(code)
    assert cls == "transcription" and ex.declared_ok is None and ex.graph_match is False


def test_missing_ans_is_format_and_printing_is_not_a_fallback():
    ex, cls = run_and_classify(DECL + BUILD + "print('ANSWER:', G.degree(1))")
    assert cls == "format" and ex.ans is None and "ANSWER: 3" in ex.stdout


# ---------------- graph recovery ----------------

def test_digraph_on_undirected_task_flags_mismatch():
    code = (DECL + "import networkx as nx\nG = nx.DiGraph()\nG.add_nodes_from(nodes)\n"
            "G.add_edges_from(edges)\nG.add_edges_from([(v, u) for u, v in edges])\nans = G.out_degree(1)")
    ex, cls = run_and_classify(code)
    assert cls == "ok" and ex.directedness_mismatch and ex.graph_match is True


def test_multiple_graphs_picks_best_match():
    code = DECL + BUILD + "H = nx.Graph([(0, 1)])\nK = G.copy()\nans = G.degree(1)"
    ex, cls = run_and_classify(code)
    assert cls == "ok" and len(ex.graphs) == 3 and ex.graph_match is True


def test_string_labels_normalize():
    code = ("nodes = ['0','1','2','3']\nedges = [('0','1'),('1','2'),('1','3'),('2','3')]\n"
            "import networkx as nx\nG = nx.Graph()\nG.add_edges_from(edges)\nans = G.degree('1')")
    ex, cls = run_and_classify(code)
    assert cls == "ok" and ex.declared_ok is True and ex.graph_match is True


# ---------------- the native library arm ----------------

def test_native_arm_verifies_transcription_without_networkx():
    code = DECL + "ans = sum(1 for e in edges if 1 in e)"
    ex, cls = run_and_classify(code)
    assert cls == "ok" and ex.declared_ok is True and ex.graphs == []


def test_native_arm_folds_construction_into_logic():
    """No graph object exists to compare against, so `construction_ok` is None and a
    wrong answer on a correctly declared graph can only be reported as logic."""
    code = DECL + "ans = sum(1 for e in edges if 1 in e) + 1"
    ex, cls = run_and_classify(code)
    assert cls == "logic" and ex.declared_ok is True and ex.construction_ok is None


def test_native_arm_transcription_still_separates():
    code = DECL_SHORT + "ans = sum(1 for e in edges if 1 in e)"
    ex, cls = run_and_classify(code)
    assert cls == "transcription" and ex.declared_ok is False


# ---------------- inject mode ----------------

def test_inject_mode_predefines_nodes_and_edges_with_the_variant_labels():
    v = relabel(CANON, 0)
    q = v.label_map[1]
    code = f"assert nodes == sorted(nodes)\nans = sum(1 for e in edges if {q} in e)"
    ex, cls = run_and_classify(code, v=v, mode="graph_as_code")
    assert cls == "ok" and ex.ans == "3" and ex.graphs == []


def test_inject_mode_overwriting_the_given_data_is_construction_not_transcription():
    code = ("edges = edges[1:]\nimport networkx as nx\nG = nx.Graph()\n"
            "G.add_nodes_from(nodes)\nG.add_edges_from(edges)\nans = G.degree(1)")
    ex, cls = run_and_classify(code, mode="graph_as_code")
    assert cls == "construction" and ex.declared_ok is False


def test_inject_mode_edges_are_tuples():
    ex = sandbox.run("ans = all(isinstance(e, tuple) and len(e) == 2 for e in edges)", CANON,
                     "graph_as_code", timeout=3)
    assert ex.ans == "Yes"


# ---------------- plumbing ----------------

def test_ans_formatting_for_bool_and_sets():
    ex = sandbox.run("ans = {3, 1, 2}", CANON, "code", timeout=3)
    assert ex.ans == "[1, 2, 3]"
    ex = sandbox.run("ans = True", CANON, "code", timeout=3)
    assert ex.ans == "Yes"
    ex = sandbox.run("ans = [(2, 1), (3, 2)]", CANON, "code", timeout=3)
    assert ex.ans == "[(2, 1), (3, 2)]"


def test_exec_cache(tmp_path):
    code = DECL + BUILD + "ans = G.degree(1)"
    a = sandbox.run(code, CANON, "code", timeout=3, cache_dir=tmp_path)
    b = sandbox.run(code, CANON, "code", timeout=3, cache_dir=tmp_path)
    assert not a.cached and b.cached and a.ans == b.ans == "3"


def test_execution_records_its_environment():
    """A result produced under a different networkx is a different measurement, so the version
    travels with the result rather than living only in someone's shell."""
    import networkx as nx
    ex = sandbox.run(DECL + BUILD + "ans = G.degree(1)", CANON, "code", timeout=5)
    assert ex.env and ex.env["networkx"] == nx.__version__
    assert ex.env["python"].startswith(".".join(__import__("sys").version.split()[0].split(".")[:2]))


def test_a_timeout_is_retried_and_never_cached(tmp_path, monkeypatch):
    calls = []
    real = sandbox._execute

    def slow_once(*args):  # the first attempt loses the race against the wall clock
        calls.append(1)
        r = real(*args)
        if len(calls) == 1:
            r.timed_out, r.exit_code, r.ans = True, -9, None
        return r
    monkeypatch.setattr(sandbox, "_execute", slow_once)
    ex = sandbox.run(DECL + "ans = 3", CANON, "code", timeout=5, cache_dir=tmp_path)
    assert len(calls) == 2 and not ex.timed_out and ex.ans == "3"
    assert len(list(tmp_path.glob("*.json"))) == 1  # the retry's result is cached

    ex = sandbox.run("while True:\n    pass\nans = 1", CANON, "code", timeout=1, cache_dir=tmp_path / "loop")
    assert ex.timed_out and not list((tmp_path / "loop").glob("*.json"))


def test_programs_read_an_empty_stdin(monkeypatch):
    """The program must not read the harness's stdin (what it would read depends on how the run was launched)."""
    import subprocess
    seen = []
    real = subprocess.Popen

    def spy(*args, **kwargs):
        seen.append(kwargs.get("stdin"))
        return real(*args, **kwargs)
    monkeypatch.setattr(sandbox.subprocess, "Popen", spy)
    ex, cls = run_and_classify(DECL + "n = int(input())\nans = n")
    assert seen == [subprocess.DEVNULL]
    assert cls == "execution" and "EOFError" in (ex.exception or "")
