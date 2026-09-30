"""Canned-response "LLM" so the whole pipeline runs with zero API calls.

Behaviours map onto the failure taxonomy: correct, wrong_logic, drop_edge
(transcription), bad_construction, no_computation, crash, timeout, off_template
(unverifiable) and bad_format. Replies follow the real contracts: \\boxed{} for direct
mode, and for code modes the filled template inside CodeGraph's `# CODE START … # CODE END`
markers with the answer in `ans`.

The stub honours the `library` arm: the networkx arm builds a networkx graph, the native
arm builds an adjacency dict and imports nothing. Its answers are canned by construction,
so its numbers are meaningless as results -- only the plumbing is being exercised.
"""
from __future__ import annotations

import hashlib
from typing import Any

from gsi.data.base import GraphInstance
from gsi.llm.client import LLMResponse, ModelSpec, extract_code
from gsi.prompts.modes import Prompt, is_inject
from gsi.prompts.solutions import solution_expr
from gsi.score.compare import relabel_answer
from gsi.serial.variant import Variant

BEHAVIOURS = ("correct", "wrong_logic", "drop_edge", "bad_construction", "no_computation",
              "crash", "timeout", "off_template", "bad_format")
MIX_WEIGHTS = {"correct": 45, "wrong_logic": 12, "drop_edge": 10, "bad_construction": 6,
               "no_computation": 4, "crash": 5, "timeout": 3, "off_template": 7, "bad_format": 8}

# Behaviours that need something the arm cannot express, and what they degrade to.
FALLBACK = {
    ("drop_edge", "inject"): "bad_construction",          # nothing to mis-transcribe
    ("bad_construction", "native"): "wrong_logic",        # no graph object to build wrong
}

WRONG = {
    "bool": "not ans",
    "int": "ans + 1",
    "node": "ans + 1",
    "float": "ans + 1.0",
    "node_set": "ans[1:] if ans else [sorted(G.nodes)[0]]",
    "path": "ans[:-1]",
    "edge_set": "ans[1:] if ans else [tuple(sorted(G.nodes)[:2])]",
}


def format_answer(value: Any, answer_type: str) -> str:
    if answer_type == "bool":
        return "Yes" if value else "No"
    if answer_type == "float":
        return f"{float(value):.4f}"
    if answer_type == "edge_set":
        return str([tuple(e) for e in value])
    return str(value)


def wrong_answer(value: Any, answer_type: str, nodes: list[int]) -> Any:
    if answer_type == "bool":
        return not value
    if answer_type in ("int", "node"):
        return value + 1
    if answer_type == "float":
        return value + 1.0
    if answer_type == "node_set":
        return value[1:] if value else [nodes[0]]
    if answer_type == "path":
        return value[:-1]
    return value[1:] if value else [[nodes[0], nodes[1]]]


class StubLLM:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.behaviour = spec.stub.get("behaviour", "mix")
        if self.behaviour != "mix" and self.behaviour not in BEHAVIOURS:
            raise ValueError(f"unknown stub behaviour {self.behaviour!r}; use 'mix' or one of {BEHAVIOURS}")

    def _pick(self, prompt: Prompt) -> str:
        if self.behaviour != "mix":
            return self.behaviour
        r = int(hashlib.sha256(prompt.prompt_hash.encode()).hexdigest()[:8], 16) % sum(MIX_WEIGHTS.values())
        for name, w in MIX_WEIGHTS.items():
            if r < w:
                return name
            r -= w
        return "correct"

    def complete(self, prompt: Prompt, *, context: dict | None = None) -> LLMResponse:
        if context is None:
            raise ValueError("StubLLM needs context={'inst': GraphInstance, 'variant': Variant}")
        inst: GraphInstance = context["inst"]
        v: Variant = context["variant"]
        behaviour = self._pick(prompt)
        if prompt.mode == "direct":
            text = self._direct(inst, v, behaviour)
        else:
            inject = is_inject(prompt.mode)
            library = prompt.library or "networkx"
            behaviour = FALLBACK.get((behaviour, "inject" if inject else ""), behaviour)
            behaviour = FALLBACK.get((behaviour, library), behaviour)
            text = self._code(inst, v, behaviour, inject=inject, library=library)
        return LLMResponse(prompt_id=prompt.prompt_id, model=self.spec.name, prompt_hash=prompt.prompt_hash,
                           raw_text=text, code=extract_code(text), usage={"stub_behaviour": behaviour}, latency_s=0.0)

    def _gt_variant(self, inst: GraphInstance, v: Variant) -> Any:
        return relabel_answer(inst.ground_truth, inst.answer_type, v.label_map)

    def _direct(self, inst: GraphInstance, v: Variant, behaviour: str) -> str:
        ans = self._gt_variant(inst, v)
        if behaviour != "correct":
            ans = wrong_answer(ans, inst.answer_type, v.node_seq)
        if behaviour == "bad_format":
            return "I am not certain about this one; it could be several values."
        return f"Let me reason about the graph.\nTherefore, the final answer is: $\\boxed{{{format_answer(ans, inst.answer_type)}}}$."

    def _query(self, inst: GraphInstance, v: Variant) -> dict:
        return {k: v.label_map.get(val, val) for k, val in inst.query_args.items()}

    def _drop_index(self, inst: GraphInstance, v: Variant, edges: list[tuple]) -> int:
        """Which edge to lose. Prefer one touching a queried node: dropping an arbitrary
        edge usually leaves the answer unchanged, which would make the transcription
        behaviours score as `ok` and leave the class unexercised."""
        touched = set(self._query(inst, v).values())
        for i, e in enumerate(edges):
            if touched & set(e[:2]):
                return i
        return 0

    def _declaration(self, inst: GraphInstance, v: Variant, behaviour: str) -> list[str]:
        edges = [tuple(e) for e in v.edge_seq]
        if behaviour == "drop_edge" and edges:
            edges = [e for i, e in enumerate(edges) if i != self._drop_index(inst, v, edges)]
        return [f"nodes = {list(v.node_seq)}", f"edges = {edges}"]

    def _networkx_body(self, inst: GraphInstance, v: Variant, behaviour: str) -> list[str]:
        cls = "DiGraph" if v.directed else "Graph"
        adder = "add_weighted_edges_from" if v.weighted else "add_edges_from"
        # bad_construction declares the graph correctly, then loses an edge while building it
        source = "edges"
        if behaviour == "bad_construction":
            k = self._drop_index(inst, v, [tuple(e) for e in v.edge_seq])
            source = f"[e for i, e in enumerate(edges) if i != {k}]"
        expr = solution_expr(inst) or repr(self._gt_variant(inst, v))
        return ["import networkx as nx", f"G = nx.{cls}()", "G.add_nodes_from(nodes)", f"G.{adder}({source})",
                f"q = {self._query(inst, v)}", f"ans = {expr}"]

    def _native_body(self, inst: GraphInstance, v: Variant, behaviour: str) -> list[str]:
        # No third-party import. `ans` is looked up rather than written down, so the
        # no_computation detector does not fire on a canned answer.
        value = self._gt_variant(inst, v)
        if behaviour == "wrong_logic":
            value = wrong_answer(value, inst.answer_type, list(v.node_seq))
        return ["adj = {u: [] for u in nodes}",
                "for e in edges:\n"
                "    adj.setdefault(e[0], []).append(e[1])\n"
                "    adj.setdefault(e[1], []).append(e[0])",
                f"answers = {{len(adj): {value!r}}}",
                "ans = answers[len(adj)]"]

    def _code(self, inst: GraphInstance, v: Variant, behaviour: str, inject: bool, library: str) -> str:
        lines = ["# CODE START"]
        if behaviour == "no_computation":
            lines += [f"ans = {self._gt_variant(inst, v)!r}", "# CODE END"]
            return "Here is the code:\n" + "\n".join(lines)
        if behaviour == "off_template":
            # ignores the template slots entirely and uses no networkx: nothing to verify
            bad = wrong_answer(self._gt_variant(inst, v), inst.answer_type, list(v.node_seq))
            lines += [f"adjacency = {{{', '.join(f'{u}: []' for u in v.node_seq)}}}",
                      f"candidates = {{len(adjacency): {bad!r}}}",
                      "ans = candidates[len(adjacency)]",
                      "# CODE END"]
            return "Here is the code:\n" + "\n".join(lines)

        if not inject:
            lines += self._declaration(inst, v, behaviour)
        if library == "networkx":
            lines += self._networkx_body(inst, v, behaviour)
            if behaviour == "wrong_logic":
                lines.append(f"ans = {WRONG[inst.answer_type]}")
        else:
            lines += self._native_body(inst, v, behaviour)
        if behaviour == "crash":
            lines.append("raise RuntimeError('stub crash')")
        if behaviour == "timeout":
            lines.append("while True:\n    pass")
        if behaviour == "bad_format":
            lines = [ln.replace("ans =", "result =") for ln in lines]  # markers present, no `ans`
        lines.append("# CODE END")
        return "Here is the code:\n" + "\n".join(lines)
