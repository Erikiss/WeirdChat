#!/usr/bin/env bash
# Phase 1b — build the baseline ("normal behavior") corpus for draft training.
#
# Wraps DeepSpec's data scripts, but regenerates with the *WeirdChat protocol*:
# temperature 1.0, no top-p/top-k restriction, thinking disabled, 1024 new
# tokens — the settings the dataset's transcripts were sampled with, not the
# Qwen-recommended 0.7/0.8 defaults from DeepSpec's example.
#
# Prerequisites:
#   - DEEPSPEC_ROOT points at a DeepSpec checkout with its requirements installed.
#   - The target model is being served on OpenAI-compatible endpoints
#     (see $DEEPSPEC_ROOT/scripts/data/launch_sglang_server.sh for the pattern).
#
# Usage:
#   DEEPSPEC_ROOT=~/DeepSpec TARGET_MODEL=Qwen/Qwen3.6-35B-A3B \
#     bash phase1_baseline.sh 127.0.0.1:30000 [more host:port ...]

set -euo pipefail

: "${DEEPSPEC_ROOT:?set DEEPSPEC_ROOT to a DeepSpec checkout}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.6-35B-A3B}"
OUT_DIR="${OUT_DIR:-$(pwd)/data}"
HELDOUT_FRAC="${HELDOUT_FRAC:-0.02}"
if [ "$#" -lt 1 ]; then
  echo "usage: bash phase1_baseline.sh <server host:port> [more ...]" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

# Step 1: prompt corpus (same source DeepSpec trains its released drafts on).
python "${DEEPSPEC_ROOT}/scripts/data/download_and_split.py" \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path "${OUT_DIR}/perfectblend_train.jsonl" \
    --test-output-dir "${OUT_DIR}/eval_heldout" \
    --skip-existing

# Step 2: regenerate assistant answers under the WeirdChat protocol.
python "${DEEPSPEC_ROOT}/scripts/data/generate_train_data.py" \
    --model "${TARGET_MODEL}" \
    --server-address "$@" \
    --concurrency 32 \
    --temperature 1.0 \
    --max-tokens 1024 \
    --disable-thinking \
    --resume \
    --input-file-path "${OUT_DIR}/perfectblend_train.jsonl" \
    --output-file-path "${OUT_DIR}/baseline_regen_full.jsonl"

# Step 3: split off a held-out slice — the scoring null distribution. The split
# is deterministic (line hash) so reruns are stable.
python - "$OUT_DIR" "$HELDOUT_FRAC" <<'PY'
import hashlib, json, sys

out_dir, frac = sys.argv[1], float(sys.argv[2])
train, heldout = [], []
with open(f"{out_dir}/baseline_regen_full.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        digest = hashlib.sha256(line.encode()).digest()
        (heldout if digest[0] / 255.0 < frac else train).append(line)
with open(f"{out_dir}/baseline_train.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(train) + ("\n" if train else ""))
with open(f"{out_dir}/baseline_heldout.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(heldout) + ("\n" if heldout else ""))
print(f"baseline split: {len(train)} train / {len(heldout)} heldout")
PY

echo "Phase 1b done:"
echo "  draft-training corpus: ${OUT_DIR}/baseline_train.jsonl"
echo "  scoring null corpus:   ${OUT_DIR}/baseline_heldout.jsonl"
