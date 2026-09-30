#!/bin/bash
# Regenerate every main-experiment table, the paper's metrics and the paper's figures from the
# published records, offline. No model is called: the API keys are blanked for the whole run.
#
#   scripts/reproduce.sh            # about 5 minutes; downloads ~80 MB of Parquet once
#   scripts/reproduce.sh --pdf      # also rebuild paper/main.pdf (needs latexmk)
#
# Afterwards `git status` shows what changed. CSV and JSON outputs come out byte-identical to the
# committed ones; PNG and PDF figures can differ in bytes (library version, timestamps) but not in
# content. The repeat floor's model calls (scripts/nondeterminism_floor.py) are not rerun; its statistics are.
set -euo pipefail
cd "$(dirname "$0")/.."
# the interpreter: $PYTHON if set, else the project venv (POSIX or Windows layout), else python3
if [ -n "${PYTHON:-}" ]; then PY=$(command -v "$PYTHON")
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python ]; then PY=.venv/Scripts/python
else PY=$(command -v python3 || command -v python)
fi
# No key reaches any client, and the dataset is downloaded anonymously (not with a stored login).
export HF_TOKEN= HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MPLBACKEND=Agg
MODELS=(hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b hf-deepseek-v4-flash)
step() { echo "=== $(date '+%T') $*"; }

step "download the scored records (results/<dataset>/scored/)"
"$PY" scripts/restore_from_hf.py

for cfg in graphqa erdos erdos_m3fix; do
  step "$cfg: tables and figures (results/$cfg/)"
  "$PY" scripts/analyze.py --config configs/$cfg.yaml > /dev/null
  if [ "$cfg" != erdos_m3fix ]; then
    "$PY" scripts/analyze.py --config configs/$cfg.yaml --models "${MODELS[@]}" > /dev/null
  fi
done

step "graph-type rerun: before/after and ablation (results/erdos_m3fix/tables/{compare,ablation}/)"
"$PY" scripts/m3fix_compare.py > /dev/null
"$PY" scripts/graph_type_ablation.py > /dev/null

step "paired graph bootstrap (results/review/{summary.json,paired_intervals.csv,per_instance.csv})"
"$PY" scripts/audit_results.py > /dev/null

step "paper metrics (results/review/paper_metrics.json; about 2 minutes)"
"$PY" scripts/paper_metrics.py > /dev/null

step "repeat floor (results/review/floor/floor_stats_*.csv)"
"$PY" scripts/floor_stats.py --rule literal > /dev/null
"$PY" scripts/floor_stats.py --rule flip > /dev/null

step "paper figures (paper/figures/)"
"$PY" paper/figures/make_figures.py > /dev/null

if [ "${1:-}" = "--pdf" ]; then
  step "paper PDF (paper/main.pdf)"
  (cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex > /dev/null)
fi

step "done; changed tables (none expected):"
git status --short -- '*.csv' '*.json' || true
