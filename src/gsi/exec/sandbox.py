"""Host-side wrapper: runs model code in a subprocess with a timeout, recovers the graph
the model *declared* (`nodes` / `edges`) and the networkx graph(s) it actually built.
Not a security boundary (no network isolation).

Three comparisons come out of one run, and together they localize where a wrong answer
was born:

    truth  --declared_ok--> declared nodes/edges  --construction_ok--> graph used  -> ans

`declared_ok` is the transcription check and needs no networkx: it reads the two template
slots. `construction_ok` needs a networkx graph to compare against, so it is available in
the `networkx` library arm only; in the `native` arm it is None and construction errors
are indistinguishable from solving errors.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gsi.prompts.modes import INJECT_MODES
from gsi.serial.variant import Variant, norm_label, normalize_edges, true_edge_set

RUNNER = Path(__file__).with_name("runner.py")
CACHE_VERSION = 7
GRAPH_VARS = ("nodes", "edges")


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    wall_s: float
    graphs: list[dict] = field(default_factory=list)
    chosen_idx: int | None = None
    graph_match: bool | None = None
    directedness_mismatch: bool = False
    exception: str | None = None
    ans: str | None = None
    declared_nodes: list | None = None
    declared_edges: list | None = None
    declared_ok: bool | None = None
    construction_ok: bool | None = None
    env: dict | None = None
    ans_is_literal: bool | None = None
    reads_graph_vars: bool | None = None
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_literal(k) and _is_literal(v)
                   for k, v in zip(node.keys, node.values))
    return False


COMPOUND = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.Match, ast.ClassDef)


def ans_is_literal(code: str) -> bool | None:
    """Was `ans` written down rather than computed? True only when every assignment to `ans`
    is a literal, none of them sits inside a compound statement, and nothing mutates it.

    The pilot showed why the nesting condition is essential: the common cycle-check shape is
    `ans = False` at top level and `ans = True` inside a loop or branch. Both are literals,
    but which one runs is decided by the program -- that is computation, and the first
    version of this detector flagged all 24 such programs as hardcoded. A literal at top
    level with no conditional sibling (`ans = 3`) is the hardcoding the template invites.
    Accumulators are not literals either, whether they grow by `ans += 1` or by `ans.append(...)`
    -- the second shape (a bridges finder collecting into `ans = []` from inside a DFS) was the
    pilot's other false positive. None if the source does not parse, which surfaces as an
    execution failure anyway."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    literal = other = 0

    def is_ans(n):
        return isinstance(n, ast.Name) and n.id == "ans"

    def visit(node, nested):
        nonlocal literal, other
        for ch in ast.iter_child_nodes(node):
            inside = nested or isinstance(node, COMPOUND)
            if isinstance(ch, ast.Assign) and any(is_ans(t) for t in ch.targets):
                if _is_literal(ch.value) and not inside:
                    literal += 1
                else:
                    other += 1
            elif isinstance(ch, (ast.AugAssign, ast.AnnAssign)) and is_ans(ch.target):
                other += 1
            elif isinstance(ch, ast.For) and is_ans(ch.target):
                other += 1
            # in-place mutation: `ans.append(x)`, `ans.add(x)`, `ans.sort()`, `ans[k] = v`, `del ans[k]`.
            # `ans = []` followed by these is an accumulator, not a written-down answer.
            elif isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute) and is_ans(ch.func.value):
                other += 1
            elif isinstance(ch, ast.Subscript) and is_ans(ch.value) and isinstance(ch.ctx, (ast.Store, ast.Del)):
                other += 1
            visit(ch, inside)

    visit(tree, False)
    return literal > 0 and other == 0


def reads_graph_vars(code: str) -> bool | None:
    """Does the program ever read `nodes` or `edges`? Reported as a template-compliance
    diagnostic; failures are not classified on it, because a program can ignore the
    template slots and still compute the answer from an inline graph."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return any(isinstance(n, ast.Name) and n.id in GRAPH_VARS and isinstance(n.ctx, ast.Load)
               for n in ast.walk(tree))


def _declared_checks(side: dict, v: Variant) -> tuple[bool | None, frozenset[str] | None]:
    """(declared_ok, normalized declared edge set). declared_ok compares the model's
    `nodes`/`edges` against the variant it was shown -- node set and edge set both."""
    edges, nodes = side.get("declared_edges"), side.get("declared_nodes")
    if edges is None:
        return None, None
    norm = normalize_edges(edges, v.directed)
    ok = norm == true_edge_set(v)
    if nodes is not None:
        ok = ok and {norm_label(n) for n in nodes} == {norm_label(n) for n in v.node_seq}
    return ok, norm


def _cache_key(code: str, v: Variant, mode: str, timeout: float, mem_mb: int) -> str:
    # the limits and the networkx version are part of the measurement, so they are part of the key
    import networkx
    payload = json.dumps({"v": CACHE_VERSION, "inject": mode in INJECT_MODES, "code": code, "node_seq": v.node_seq,
                          "edge_seq": v.edge_seq, "directed": v.directed, "weighted": v.weighted,
                          "timeout": timeout, "mem_mb": mem_mb, "networkx": networkx.__version__}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


_inflight_guard = threading.Lock()
_inflight: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    with _inflight_guard:
        return _inflight.setdefault(key, threading.Lock())


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a concurrent reader never sees a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows will not replace a file another process holds open. Cache keys are content
        # hashes, so a file already there holds an equivalent entry: keep it and drop ours.
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise


def _read_cached(path: Path) -> ExecutionResult | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    d["cached"] = True
    return ExecutionResult(**d)


def _match_graphs(raw_graphs: list[dict], v: Variant) -> tuple[list[dict], int | None, bool | None, bool]:
    truth = true_edge_set(v)
    graphs = []
    best_idx, best_j = None, -1.0
    for i, g in enumerate(raw_graphs):
        directed_g = bool(g.get("directed"))
        norm = normalize_edges(g.get("edges", []), directed_g and v.directed)
        j = len(norm & truth) / len(norm | truth) if (norm or truth) else 1.0
        graphs.append({"cls": g.get("cls"), "directed": directed_g, "n": g.get("n"), "m": g.get("m"),
                       "edges_norm": sorted(norm), "jaccard": j, "truncated": g.get("truncated", False)})
        if j > best_j:
            best_idx, best_j = i, j
    if best_idx is None:
        return graphs, None, None, False
    chosen = graphs[best_idx]
    mismatch = chosen["directed"] != v.directed
    match = frozenset(chosen["edges_norm"]) == truth and not (v.directed and not chosen["directed"])
    return graphs, best_idx, match, mismatch


def run(code: str, v: Variant, mode: str, *, timeout: float = 20, mem_mb: int = 1024,
        cache_dir: str | Path | None = None) -> ExecutionResult:
    key = _cache_key(code, v, mode, timeout, mem_mb)
    cache_path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    # identical executions (same code, same data) are serialized so only one of them runs
    with _key_lock(key) if cache_path else nullcontext():
        if cache_path and cache_path.exists():
            cached = _read_cached(cache_path)
            if cached is not None:
                return cached
        result = _execute(code, v, mode, timeout, mem_mb)
        if result.timed_out:
            # The limit is wall-clock time, so a busy machine can time out a program that finishes in a
            # fraction of a second (every timeout of the published run did). Try once more, and never
            # cache a timeout: a later run gets its own chance.
            result = _execute(code, v, mode, timeout, mem_mb)
        if cache_path and not result.timed_out:
            atomic_write_text(cache_path, json.dumps(result.to_dict()))
        return result


def _execute(code: str, v: Variant, mode: str, timeout: float, mem_mb: int) -> ExecutionResult:
    with tempfile.TemporaryDirectory(prefix="gsi-exec-", ignore_cleanup_errors=True) as tmp:
        (Path(tmp) / "code.py").write_text(code, encoding="utf-8")  # the Windows default cannot encode e.g. "→"
        (Path(tmp) / "variant.json").write_text(json.dumps({
            "node_seq": v.node_seq, "edge_seq": v.edge_seq, "directed": v.directed, "weighted": v.weighted,
            "inject": mode in INJECT_MODES, "mem_mb": mem_mb, "timeout": timeout,
        }))
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "LANG": "C.UTF-8", "MPLBACKEND": "Agg",
               # One BLAS thread: OpenBLAS otherwise sizes buffers for every core the host reports, and
               # on a many-core Linux host that alone exceeds the mem_mb address-space limit, so numpy's
               # import fails before the program runs. Results are unaffected; only thread count is.
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        t0 = time.time()
        timed_out = False
        proc = subprocess.Popen([sys.executable, "-I", str(RUNNER), tmp], cwd=tmp, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            _, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # Windows has no process groups here; kill the child itself
                    proc.kill()
            except ProcessLookupError:
                pass
            _, stderr = proc.communicate()
            exit_code = -9
        wall = time.time() - t0

        sidecar = Path(tmp) / "sidecar.json"
        side = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    graphs, chosen, match, mismatch = _match_graphs(side.get("graphs", []), v)
    declared_ok, declared_norm = _declared_checks(side, v)
    construction_ok = None
    if chosen is not None and declared_norm is not None:
        construction_ok = frozenset(graphs[chosen]["edges_norm"]) == declared_norm
    return ExecutionResult(
        stdout=side.get("stdout", ""), stderr=(stderr or "")[-4000:], exit_code=exit_code, timed_out=timed_out,
        wall_s=round(wall, 3), graphs=graphs, chosen_idx=chosen, graph_match=match,
        directedness_mismatch=mismatch, exception=side.get("exception"), ans=side.get("ans"),
        declared_nodes=side.get("declared_nodes"), declared_edges=side.get("declared_edges"),
        declared_ok=declared_ok, construction_ok=construction_ok,
        env=side.get("env"), ans_is_literal=ans_is_literal(code), reads_graph_vars=reads_graph_vars(code),
    )
