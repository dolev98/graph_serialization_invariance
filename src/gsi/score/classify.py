"""Where a wrong answer was born.

The template gives every code-mode program the same three visible stages, so a failure
can be attributed to one of them rather than lumped into "wrong":

    truth --[1]--> declared nodes/edges --[2]--> the graph the code used --[3]--> ans

  [1] transcription   the model copied the graph wrong
  [2] construction    it declared the graph correctly, then built something else from it
  [3] logic           it had the right graph and still got the answer wrong

`no_computation` sits outside that chain: the program wrote `ans` as a literal, so it
never entered the chain at all. It is checked before correctness, since a hardcoded
answer that happens to be right is not a solved task.

`construction` needs a networkx graph to compare against, so it is separable in the
`networkx` library arm only. In the `native` arm there is no graph object to recover and
construction errors are reported as `logic`; the two must never be pooled across arms.
"""
from __future__ import annotations

from gsi.prompts.modes import is_inject


def classify(mode: str, correct: bool, parsed, exec_result: dict | None) -> str:
    if mode == "direct":
        if correct:
            return "ok"
        return "format" if parsed is None else "wrong"

    if exec_result is None or exec_result.get("timed_out") or exec_result.get("exit_code", 1) != 0:
        return "execution"
    if parsed is None:
        return "format"

    # Checked before correctness on purpose: a program that writes the answer down as a
    # literal has not computed anything, and counting a lucky one as `ok` would inflate
    # exactly the number this project measures.
    if exec_result.get("ans_is_literal") is True:
        return "no_computation"
    if correct:
        return "ok"

    inject = is_inject(mode)
    declared_ok = exec_result.get("declared_ok")
    if declared_ok is False:
        # In an inject mode the harness supplied `nodes`/`edges`, so a mismatch means the
        # model overwrote correct data -- that is construction, not transcription.
        return "construction" if inject else "transcription"
    if exec_result.get("construction_ok") is False:
        return "construction"

    if declared_ok is None and not inject:
        # The model ignored the template slots. Fall back to the recovered networkx graph.
        if exec_result.get("graph_match") is False:
            return "transcription"
        if not exec_result.get("graphs"):
            return "unverifiable"
    return "logic"
