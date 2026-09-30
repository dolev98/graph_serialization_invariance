"""Interaction modes and prompt construction.

Three modes, two of which are crossed with a `library` arm:

  M1 direct         answer in prose; G1/Herbst wording, final answer in \\boxed{}.
  M2 code           CodeGraph format; the model fills a template whose first two lines
                    declare `nodes` and `edges`, then writes the solving code.
  M3 graph_as_code  the identical template with those two lines filled by the harness
                    and the graph text removed: M2 minus transcription.

`library` selects the solving toolkit (`networkx` or `native`) and is crossed with the
two code modes, never nested inside the mode name -- so `mode in CODE_MODES` stays a
two-element membership test no matter how many library arms exist.

The template is what makes transcription observable. Because `nodes` and `edges` are
always assigned first and always by that name, the harness can read what graph the model
*declared* and compare it to the truth, independently of how the model went on to solve.
M3 receives the same skeleton, so the two modes differ in exactly one thing: who fills
those two lines.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from gsi.data.base import GraphInstance
from gsi.serial.variant import Variant

MODES = ("direct", "code", "graph_as_code")
CODE_MODES = ("code", "graph_as_code")
INJECT_MODES = ("graph_as_code",)
LIBRARIES = ("networkx", "native")

SYSTEM_DIRECT = "You are a helpful assistant."

# CodeGraph, Table 6
ROLE = ("You are an expert in graph networks and Python programming. The user is looking for guidance from "
        "an AI that is knowledgeable in graph networks, proficient in Python, and capable of providing bug-free "
        "solutions to graph network problems.")

CODEGRAPH_INSTRUCTION = ("Write a piece of Python code to return the answer in a variable 'ans'. "
                         "Please enclose the code with # CODE START and # CODE END.")

# G1 evaluation suffix (eval_erdos.py), also what Herbst et al. ran
G1_SUFFIX = ("Solve the above problem efficiently and clearly. The last line of your response should be of the "
             "following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$.'")

LIBRARY_INSTRUCTION = {
    "networkx": "You may use the networkx library.",
    "native": ("Use only plain Python and its standard library. Do not import networkx or any other "
               "third-party library."),
}

# What `ans` should hold, by answer type. Phrased as a description so the slot cannot be
# mistaken for a place to write the answer itself.
ANS_HINT = {
    "bool": "the computed result, a Python bool",
    "int": "the computed result, a Python int",
    "float": "the computed result, a Python float",
    "node": "the computed result, a single node label",
    "node_set": "the computed result, a list of node labels in ascending order",
    "path": "the computed result, a list of node labels in path order",
    "edge_set": "the computed result, a list of (u, v) tuples in ascending order",
}

# Erdős-style answer-format sentences, used when the dataset does not carry its own
FORMAT_HINT = {
    "bool": "Your answer should be Yes or No.",
    "int": "Your answer should be an integer.",
    "float": "You need to format your answer as a float number.",
    "node": "Your answer should be a single node.",
    "node_set": "You need to format your answer as a list of nodes in ascending order, e.g., [node-1, node-2, ..., node-n].",
    "path": "You need to format your answer as a list of nodes, e.g., [node-1, node-2, ..., node-n].",
    "edge_set": "You need to format your answer as a list of edges in ascending dictionary order, e.g., [(u1, v1), (u2, v2), ..., (un, vn)].",
}

TASK_SENTENCE = {
    "edge_existence": "determine whether there is an edge between two given nodes",
    "node_degree": "calculate the degree of a given node",
    "node_count": "count the number of nodes",
    "edge_count": "count the number of edges",
    "connected_nodes": "list all the nodes connected to a given node",
    "disconnected_nodes": "list all the nodes that are not connected to a given node",
    "cycle_check": "determine whether the graph contains a cycle",
}


@dataclass
class Prompt:
    prompt_id: str
    instance_id: str
    variant_id: str
    mode: str
    library: str | None
    system: str
    user: str
    answer_type: str
    prompt_hash: str
    # Which identical re-ask this is. It is deliberately NOT part of `prompt_hash`: the model must
    # see byte-identical text, or the measurement is not "the same prompt twice". It separates the
    # cache entry and the record id instead. repeat>0 exists only to measure the nondeterminism
    # floor -- how often the answer changes with no perturbation at all.
    repeat: int = 0


def is_code(mode: str) -> bool:
    return mode in CODE_MODES


def is_inject(mode: str) -> bool:
    return mode in INJECT_MODES


def libraries_for(mode: str, libraries: list[str] | tuple[str, ...]) -> list[str | None]:
    """The library arms a mode is crossed with. `direct` writes no code, so it has none."""
    return list(libraries) if is_code(mode) else [None]


def question_for(inst: GraphInstance, label_map: dict[int, int]) -> str:
    if inst.dataset == "graphqa":
        from gsi.data.graphqa import question
        return question(inst, label_map)
    if inst.dataset == "erdos":
        from gsi.data.erdos import question
        return question(inst, label_map)
    raise ValueError(f"no question template for dataset {inst.dataset}")


def task_line(inst: GraphInstance) -> str:
    if inst.meta.get("preamble"):
        return inst.meta["preamble"]
    kind = "directed" if inst.directed else "undirected"
    return f"For this task, please {TASK_SENTENCE.get(inst.task, 'answer the question')} in the {kind} graph G."


def _edge_shape(inst: GraphInstance) -> str:
    return "(u, v, w) triples, w being the edge weight" if inst.weighted else "(u, v) pairs"


UNDIRECTED_NOTE = " The graph is undirected: (u, v) and (v, u) denote the same edge."


def _graph_notes(inst: GraphInstance, state_undirected: bool = False) -> str:
    notes = ""
    if inst.directed:
        notes += " The graph is directed: (u, v) is an edge from u to v."
    elif state_undirected:
        # M3 removes the graph text, and this function only ever noted directedness, so an
        # undirected M3 prompt stated its type only if the task preamble did. Where the preamble is
        # silent or mentions only the directed case (Erdős `neighbor`), models read the graph as
        # directed (paper, Section 5.4). This states the type explicitly so
        # the control receives the same fact as M1/M2. Off by default so the original runs stay
        # reproducible.
        notes += UNDIRECTED_NOTE
    if inst.weighted:
        notes += " Each edge is given as (u, v, w) where w is its weight."
    return notes


def _direct(inst: GraphInstance, v: Variant, q: str) -> str:
    parts = []
    if inst.meta.get("preamble"):
        parts.append(inst.meta["preamble"])
    parts.append(v.text)
    hint = inst.meta.get("format_hint") or FORMAT_HINT[inst.answer_type]
    parts.append(f"Question: {q}\n\n{hint}")
    parts.append(G1_SUFFIX)
    return "\n\n".join(parts)


def template(inst: GraphInstance, inject: bool) -> str:
    """The fixed program skeleton. Identical in both code modes except for the two
    declaration lines, which M3 receives already filled."""
    ans = ANS_HINT[inst.answer_type]
    if inject:
        head = "# 'nodes' and 'edges' are already defined."
    else:
        head = f"nodes = <the nodes of G>\nedges = <the edges of G, as {_edge_shape(inst)}>"
    return f"# CODE START\n{head}\n\n<your code here>\nans = <{ans}>\n# CODE END"


def _template_instruction(inst: GraphInstance, inject: bool) -> str:
    if inject:
        what = (f"The variables 'nodes' (the nodes of G) and 'edges' (the edges of G, as {_edge_shape(inst)}) "
                "are already defined; use them directly and do not redefine them.")
    else:
        what = ("'nodes' must list every node of G and 'edges' must list every edge of G as "
                f"{_edge_shape(inst)}. Then write the code that computes the answer from 'nodes' and 'edges'.")
    return ("Fill in the template below and keep its structure. " + what +
            " Compute the result - do not write it directly.")


def _code(inst: GraphInstance, v: Variant, q: str, mode: str, library: str,
          inject_graph_type: bool = False) -> str:
    inject = is_inject(mode)
    parts = [
        task_line(inst) + "\n" + CODEGRAPH_INSTRUCTION + _graph_notes(inst, inject and inject_graph_type),
        _template_instruction(inst, inject),
        template(inst, inject),
        LIBRARY_INSTRUCTION[library],
        (f"Q: {q}" if inject else f"Q: {q}\n{v.text}") + "\nA:",
    ]
    return "\n\n".join(parts)


def build_prompt(mode: str, inst: GraphInstance, v: Variant, library: str | None = None,
                 repeat: int = 0, inject_graph_type: bool = False) -> Prompt:
    """`inject_graph_type` adds the explicit undirected sentence to M3 prompts only. It changes
    the prompt text (so the hash and the cache entry) but not `prompt_id`, so records of a run
    with the flag pair one-to-one with the base run on `record_id`."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    if is_code(mode):
        if library not in LIBRARIES:
            raise ValueError(f"mode {mode} requires library in {LIBRARIES}, got {library!r}")
    elif library is not None:
        raise ValueError(f"mode {mode} takes no library, got {library!r}")
    if not is_inject(mode) and not v.text:
        raise ValueError(f"variant {v.variant_id} has no rendered text")
    q = question_for(inst, v.label_map)
    if mode == "direct":
        system, user = SYSTEM_DIRECT, _direct(inst, v, q)
    else:
        system, user = ROLE, _code(inst, v, q, mode, library, inject_graph_type)
    h = hashlib.sha256((system + "\n\x00\n" + user).encode()).hexdigest()
    return Prompt(
        prompt_id=f"{v.variant_id}::{mode}" + (f"::{library}" if library else "") + (f"::r{repeat}" if repeat else ""),
        instance_id=inst.id,
        variant_id=v.variant_id,
        mode=mode,
        library=library,
        system=system,
        user=user,
        answer_type=inst.answer_type,
        prompt_hash=h,
        repeat=repeat,
    )
