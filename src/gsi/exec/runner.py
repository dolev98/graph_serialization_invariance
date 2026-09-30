"""Runs inside the sandbox subprocess. Standalone on purpose: no gsi imports.

usage: python -I runner.py <workdir>
Reads <workdir>/code.py and <workdir>/variant.json, writes <workdir>/sidecar.json.

The answer is the variable `ans` left in the program's namespace (CodeGraph's
convention); when the program raised or never defined it, `ans` is null.
For inject modes the harness predefines `nodes` and `edges` before execution.

`nodes` and `edges` are also read back out of the namespace and reported as the graph
the model *declared*. They are read even when the program raised, because the template
assigns them first: a program that transcribed the graph and then crashed in its solving
code still tells us whether the transcription was right.
"""
import contextlib
import io
import json
import os
import sys
import traceback

MAX_DUMP = 20000
_MISSING = object()


def _limits(cfg):
    try:
        import resource
        mem = int(cfg.get("mem_mb", 1024)) * 2**20
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            lim = getattr(resource, name, None)
            if lim is not None:
                try:
                    resource.setrlimit(lim, (mem, mem))
                except (ValueError, OSError):
                    pass
        cpu = int(cfg.get("timeout", 20)) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass


def _format(ans):
    try:
        import numpy as np
        if isinstance(ans, np.generic):
            ans = ans.item()
    except Exception:
        pass
    if isinstance(ans, bool):
        return "Yes" if ans else "No"
    if isinstance(ans, (set, frozenset)):
        try:
            ans = sorted(ans)
        except TypeError:
            ans = list(ans)
    if isinstance(ans, tuple):
        ans = list(ans)
    if isinstance(ans, list):
        ans = [tuple(x) if isinstance(x, (list, tuple)) else x for x in ans]
    return str(ans)


def _declared_nodes(value):
    """JSON-safe copy of the model's `nodes`; None when it is not a usable sequence."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return None
    out = []
    for i, n in enumerate(value):
        if i >= MAX_DUMP:
            break
        out.append(n if isinstance(n, (int, float)) and not isinstance(n, bool) else str(n))
    return out


def _declared_edges(value):
    """JSON-safe copy of the model's `edges`; None when it is not a usable sequence of pairs."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return None
    out = []
    for i, e in enumerate(value):
        if i >= MAX_DUMP:
            break
        if isinstance(e, (str, bytes)) or not hasattr(e, "__iter__"):
            return None
        parts = list(e)
        if len(parts) < 2:
            return None
        out.append([p if isinstance(p, (int, float)) and not isinstance(p, bool) else str(p) for p in parts])
    return out


def main():
    workdir = sys.argv[1]
    with open(os.path.join(workdir, "variant.json")) as f:
        cfg = json.load(f)
    _limits(cfg)
    with open(os.path.join(workdir, "code.py"), encoding="utf-8") as f:
        code = f.read()

    import networkx as nx

    registry = []

    def install_hook():
        for cls in (nx.Graph, nx.DiGraph):
            orig = cls.__init__

            def make(orig):
                def __init__(self, *a, **k):
                    orig(self, *a, **k)
                    registry.append(self)
                return __init__

            cls.__init__ = make(orig)

    ns = {"__name__": "__main__"}
    if cfg.get("inject"):
        ns["nodes"] = list(cfg["node_seq"])
        ns["edges"] = [tuple(e) for e in cfg["edge_seq"]]

    out = io.StringIO()
    result = {"stdout": "", "exception": None, "exit_code": 0, "graphs": [], "ans": None,
              "declared_nodes": None, "declared_edges": None,
              # A result produced under a different networkx is a different measurement: what
              # nx.Graph() deduplicates and which algorithms exist are part of what we measure.
              # Recorded per run so a scored row is self-describing.
              "env": {"python": sys.version.split()[0], "networkx": nx.__version__,
                      "platform": sys.platform}}
    failed = False
    try:
        install_hook()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "solution.py", "exec"), ns)
    except SystemExit as e:
        if e.code not in (None, 0):
            result["exception"] = f"SystemExit({e.code})"
            result["exit_code"] = 1
            failed = True
    except BaseException:
        result["exception"] = traceback.format_exc()
        result["exit_code"] = 1
        failed = True
    finally:
        if not failed:
            ans = ns.get("ans", _MISSING)
            if ans is not _MISSING:
                try:
                    result["ans"] = _format(ans)
                except Exception as e:
                    result["exception"] = f"could not format ans: {e!r}"
        # read even on failure: the template assigns these before the solving code runs
        for name, coerce in (("nodes", _declared_nodes), ("edges", _declared_edges)):
            val = ns.get(name, _MISSING)
            if val is not _MISSING:
                try:
                    result[f"declared_{name}"] = coerce(val)
                except Exception:
                    result[f"declared_{name}"] = None
        result["stdout"] = out.getvalue()
        seen = set()
        for g in registry:
            if id(g) in seen:
                continue
            seen.add(id(g))
            try:
                edges = list(g.edges())
                result["graphs"].append({
                    "cls": type(g).__name__,
                    "directed": bool(g.is_directed()),
                    "n": g.number_of_nodes(),
                    "m": len(edges),
                    "edges": [[str(u), str(v)] for u, v in edges[:MAX_DUMP]],
                    "truncated": len(edges) > MAX_DUMP,
                })
            except Exception as e:
                result["graphs"].append({"cls": type(g).__name__, "error": repr(e), "edges": [], "n": 0, "m": 0,
                                         "directed": False, "truncated": False})
        with open(os.path.join(workdir, "sidecar.json"), "w") as f:
            json.dump(result, f)
        if result["exception"]:
            sys.stderr.write(result["exception"])
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
