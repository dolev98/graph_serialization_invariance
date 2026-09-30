"""Restore the scored records from the public Hugging Face dataset, so the analysis runs offline.

    python scripts/restore_from_hf.py [--revision SHA] [--datasets graphqa erdos erdos_m3fix]

Downloads the Parquet records of the dataset (pinned to a revision) and writes them in the layout
the analysis scripts read, one file per model:

    records/graphqa         -> results/graphqa/scored/<model>.jsonl
    records/erdos           -> results/erdos/scored/<model>.jsonl
    graph_type_rerun/erdos  -> results/erdos_m3fix/scored/<model>.jsonl

Each row is rebuilt with the keys, key order and value types of `gsi.experiment.run.score_record`,
and rows keep the Parquet row order (the bootstrap in paper_metrics.py resamples graphs in
first-seen order, so the order matters). The reply text and the extracted code are not written:
nothing downstream of scoring reads them. No model is called.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Dolevabudi/graph-serialization-invariance-full"
REVISION = "2519e49cbca169a9589ed7a3db8725e353bc997d"

# results dir -> Parquet file in the dataset
SOURCES = {
    "graphqa": "data/records/graphqa.parquet",
    "erdos": "data/records/erdos.parquet",
    "erdos_m3fix": "data/graph_type_rerun/erdos.parquet",
}
# score_record's keys, in its order
KEYS = ("record_id", "dataset", "task", "graph_id", "instance_id", "variant_id", "axis", "params", "mode",
        "library", "model", "answer_type", "prompt_hash", "finish_reason", "ground_truth", "answer_line",
        "parsed", "parsed_canonical", "answer_key", "correct", "failure_class", "graph_match", "n_graphs",
        "declared_ok", "construction_ok", "ans_is_literal", "reads_graph_vars", "timed_out", "exit_code",
        "directedness_mismatch", "stub_behaviour")
# stored in Parquet as JSON text; the scored files hold the decoded value
JSON_COLUMNS = ("params", "ground_truth", "parsed", "parsed_canonical")


def to_scored(row: dict) -> dict:
    out = {}
    for k in KEYS:
        v = row.get(k)
        if k in JSON_COLUMNS and v is not None:
            v = json.loads(v)
        elif k == "answer_key" and v is None:
            v = "null"  # score_record stores json.dumps(None)
        out[k] = v
    return out


def restore(name: str, revision: str) -> dict[str, int]:
    path = hf_hub_download(REPO_ID, SOURCES[name], repo_type="dataset", revision=revision)
    out_dir = ROOT / "results" / name / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    files, counts = {}, {}
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000):
            for row in batch.to_pylist():
                m = row["model"]
                if m not in files:  # written next to the target and renamed at the end: never half a file
                    files[m] = (out_dir / f".{m}.jsonl.part").open("w")
                    counts[m] = 0
                files[m].write(json.dumps(to_scored(row)) + "\n")
                counts[m] += 1
    finally:
        for f in files.values():
            f.close()
    for m in files:
        os.replace(out_dir / f".{m}.jsonl.part", out_dir / f"{m}.jsonl")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", default=REVISION, help="dataset commit to download (default: the pinned one)")
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args()
    for name in args.datasets:
        counts = restore(name, args.revision)
        print(f"{name}: " + ", ".join(f"{m} {n:,}" for m, n in counts.items()))


if __name__ == "__main__":
    main()
