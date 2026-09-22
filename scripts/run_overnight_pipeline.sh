#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p outputs
exec >> outputs/overnight_pipeline.log 2>&1

python_bin="$project_dir/.venv/bin/python"
full_service="dataset-cartography-snli-full-optimized.service"
deadline=$(( $(date +%s) + 18 * 3600 ))

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

check_metrics() {
  "$python_bin" - "$1" "$2" <<'PY'
import json
import math
import sys
from pathlib import Path

path, expected = sys.argv[1:]
result = json.loads(Path(path).read_text())
assert result["subset"] == expected, result
assert result["model"] == "roberta-large", result
assert result["effective_batch_size"] == 96, result
for field in ("id_accuracy", "validation_accuracy", "ood_accuracy"):
    assert math.isfinite(float(result[field])), (field, result)
PY
}

remaining_seconds() {
  local remaining=$((deadline - $(date +%s)))
  if (( remaining < 600 )); then
    log "Stopping: less than 10 minutes remain in the 18-hour budget"
    exit 1
  fi
  printf '%s\n' "$remaining"
}

run_condition() {
  local subset="$1"
  local config="$2"
  local result_file="$3"
  if [[ -s "$result_file" ]]; then
    check_metrics "$result_file" "$subset"
    log "$subset already has validated results; skipping"
    return
  fi
  local remaining
  remaining="$(remaining_seconds)"
  log "Starting $subset with $remaining seconds left in budget"
  timeout --signal=TERM --kill-after=120 "${remaining}s" "$python_bin" scripts/train_snli.py --config "$config"
  check_metrics "$result_file" "$subset"
  "$python_bin" scripts/paper_vs_reproduction.py
  log "Completed $subset"
}

if [[ ! -x "$python_bin" ]]; then
  log "Missing project Python environment: $python_bin"
  exit 1
fi

export HF_HOME="${HF_HOME:-/tmp/cartography-hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$project_dir/outputs/matplotlib_cache"

log "Waiting for full SNLI service to finish"
while systemctl --user is-active --quiet "$full_service"; do
  remaining_seconds >/dev/null
  sleep 30
done

check_metrics outputs/checkpoints/full_seed93078/metrics.json full
"$python_bin" scripts/paper_vs_reproduction.py
log "Validated full SNLI result"

run_condition random33 configs/snli_random33.yaml outputs/checkpoints/random33_seed93078/metrics.json
run_condition ambiguous33 configs/snli_ambiguous33.yaml outputs/checkpoints/ambiguous33_seed93078/metrics.json

remaining_seconds >/dev/null
log "Building our SNLI data map and comparison"
"$python_bin" scripts/collect_training_dynamics.py
"$python_bin" scripts/build_data_map.py
"$python_bin" scripts/compare_maps.py
"$python_bin" scripts/paper_vs_reproduction.py
run_condition hard33 configs/snli_hard33.yaml outputs/checkpoints/hard33_seed93078/metrics.json
log "Overnight pipeline complete"
