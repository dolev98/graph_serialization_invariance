#!/bin/bash
# Full experiment for one model: both datasets and the Erdős M3 rerun that states the graph type, all
# stages, analysis. Calls the model API. Resumable at every stage, so re-running after any failure
# continues where it stopped. Keeps the machine awake.
#
#   nohup scripts/run_model.sh hf-qwen3-8b-nscale > results/hf-qwen3-8b-nscale.log 2>&1 &
#   tail -f results/hf-qwen3-8b-nscale.log
set -u
MODEL="${1:?usage: run_model.sh <model-name-from-configs/models.yaml>}"
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv/Scripts/python   # Windows venv layout
# caffeinate keeps macOS awake. Elsewhere, hold ES_CONTINUOUS|ES_SYSTEM_REQUIRED for the life of the
# command on Windows (a PC that slept mid-run once turned 4 programs into timeouts); no-op otherwise.
command -v caffeinate >/dev/null 2>&1 || caffeinate() {
  [ "$1" = "-i" ] && shift
  "$PY" -c 'import ctypes, shutil, subprocess, sys
k = getattr(ctypes, "windll", None)
if k: k.kernel32.SetThreadExecutionState(0x80000001)
cmd = sys.argv[1:]
cmd[0] = shutil.which(cmd[0]) or cmd[0]   # CreateProcess does not add .exe to a relative path
sys.exit(subprocess.call(cmd))' "$@"
}
echo "=== $(date '+%F %T') START model=$MODEL host=$(hostname) ==="
for cfg in graphqa erdos erdos_m3fix; do
  echo "=== $(date '+%F %T') $cfg: llm ==="
  # A non-zero exit means some calls failed (or the run aborted); run and score what was answered anyway, and
  # re-run this script later to retry the rest.
  caffeinate -i $PY scripts/run.py --config configs/$cfg.yaml --models "$MODEL" --stages data,variants,llm || { echo "=== llm stage for $cfg left records unanswered; continuing ==="; FAILED=1; }
  echo "=== $(date '+%F %T') $cfg: exec,score ==="
  caffeinate -i $PY scripts/run.py --config configs/$cfg.yaml --models "$MODEL" --stages exec,score || { echo "=== exec/score failed for $cfg; stopping ==="; exit 3; }
  echo "=== $(date '+%F %T') $cfg: analyze ==="
  $PY scripts/analyze.py --config configs/$cfg.yaml --models "$MODEL" > "results/$cfg/analyze_$MODEL.txt" 2>&1 || echo "=== analyze failed for $cfg (non-fatal) ==="
done
echo "=== $(date '+%F %T') ALL DONE model=$MODEL ==="
exit "${FAILED:-0}"
