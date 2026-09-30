from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx


@dataclass
class GraphInstance:
    id: str
    dataset: str
    task: str
    directed: bool
    weighted: bool
    nodes: list[int]
    edges: list[list]
    query_args: dict[str, Any]
    answer_type: str
    ground_truth: Any
    native_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GraphInstance":
        return cls(**d)

    def to_nx(self) -> nx.Graph:
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(self.nodes)
        if self.weighted:
            G.add_weighted_edges_from((u, v, w) for u, v, w in self.edges)
        else:
            G.add_edges_from((u, v) for u, v, *_ in self.edges)
        return G


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def trim_partial_line(path: str | Path) -> bool:
    """Repair the end of a JSONL file that a killed writer left without a final newline, so that appends start
    on a new line: a complete last row gets its newline, a row cut off mid-write is removed. Returns whether a
    row was removed."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    start = data.rfind(b"\n") + 1
    try:
        json.loads(data[start:])
    except ValueError:
        path.write_bytes(data[:start])
        print(f"warning: {path}: removed a row cut off mid-write", file=sys.stderr)
        return True
    with path.open("ab") as f:
        f.write(b"\n")
    return False


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
