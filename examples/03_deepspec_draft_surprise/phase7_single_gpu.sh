#!/usr/bin/env bash
# Phase 7 on a single GPU — self-host the subject model, sweep, analyze.
#
# Serves Qwen/Qwen3.6-35B-A3B-FP8 with vLLM on ONE GPU (the model is only
# ~3B active / ~35 GB in FP8), runs the temperature sweep against the local
# server (so weakly-OpenRouter-reproducing behaviours like language-switching
# are finally measurable), and keeps the reference rubric judge on OpenRouter.
# Then fits Arrhenius vs super-Arrhenius and prints the report.
#
# Why one GPU: qwen3.6-35b-a3b is a small MoE. A single H100 80GB or L40S 48GB
# (AWS g6e) is plenty — you do NOT need a p5/p6 8-GPU node. FP8 needs an Ada or
# Hopper GPU (L40S sm89 / H100 sm90); it will NOT run on Ampere (A10G/A100 sm8x)
# without dequantization.
#
# Requirements (env vars):
#   HF_TOKEN             — to download the FP8 checkpoint + dataset
#   OPENROUTER_API_KEY   — for the reference judge (Gemma), which stays remote
#
# Usage:
#   HF_TOKEN=hf_... OPENROUTER_API_KEY=sk-or-... bash phase7_single_gpu.sh
#   # override anything: SAMPLES=256 BEHAVIORS="language-switching-english" bash phase7_single_gpu.sh
#
# Rough cost/time: one H100 ~$2-4/h; the default sweep is a few hours -> tens of
# euros, well under any four-figure budget. The judge (OpenRouter, rate-limited)
# is usually the throughput bottleneck, not the GPU.

set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (needed to download the FP8 checkpoint and dataset)}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (needed for the reference judge)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" OPENROUTER_API_KEY

# ------------------------------- config -------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"   # weights vLLM downloads
SERVED_NAME="qwen/qwen3.6-35b-a3b"                     # MUST equal the dataset slug
PORT="${PORT:-8000}"
DATA_DIR="${DATA_DIR:-$HOME/weirdspec_data}"
REPO_DIR="${REPO_DIR:-$HOME/WeirdChat}"
BRANCH="${BRANCH:-claude/repo-published-weights-u71yew}"

# Sweep scale. Single-prompt patterns mean samples-per-prompt is the main knob.
BEHAVIORS="${BEHAVIORS:-language-switching-english chemtrails-assertion recommends-drunk-driving}"
PATTERNS="${PATTERNS:-6}"
PROMPTS="${PROMPTS:-8}"
SAMPLES="${SAMPLES:-128}"
TEMPS="${TEMPS:-0.3 0.4 0.5 0.6 0.7 0.85 1.0 1.15 1.3}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-32}"   # local vLLM; H100 can take 64-128, L40S keep ~32
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"  # OpenRouter rate-limited
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# ----------------------------------------------------------------------------

if [ ! -d "$REPO_DIR" ]; then
  git clone --branch "$BRANCH" https://github.com/Erikiss/WeirdChat "$REPO_DIR"
fi
# SKIP_INSTALL=1 lets an orchestrator (e.g. SkyPilot `setup:`) do the heavy
# installs once and reuse them across job restarts.
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "==> installing deps (vllm + WeirdChat)"
  pip install -q -U vllm >/tmp/pip_vllm.log 2>&1 || { echo "vllm install failed:"; tail -20 /tmp/pip_vllm.log; exit 1; }
  pip install -q -e "$REPO_DIR" >/tmp/pip_wc.log 2>&1 || { echo "weirdchat install failed:"; tail -20 /tmp/pip_wc.log; exit 1; }
fi
cd "$REPO_DIR/examples/03_deepspec_draft_surprise"
mkdir -p "$DATA_DIR"

echo "==> starting vLLM ($MODEL_PATH -> served as '$SERVED_NAME') on port $PORT"
VLLM_LOG="/tmp/vllm_$PORT.log"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
# Always shut the server down on exit (success, error, or Ctrl-C).
trap 'echo "==> stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "==> waiting for vLLM to load (first run downloads ~35 GB; up to ~30 min)"
ready=0
for i in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process died during startup. Last log lines:"; tail -40 "$VLLM_LOG"; exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
[ "$ready" = 1 ] || { echo "vLLM did not become ready in time. Log tail:"; tail -40 "$VLLM_LOG"; exit 1; }
echo "==> vLLM ready"

echo "==> phase 7 sweep (generation local, judge on OpenRouter)"
python phase7_sweep.py \
  --output "$DATA_DIR/temp_sweep_gpu.jsonl" \
  --base-url "http://localhost:$PORT/v1" \
  --api-key-env NONE \
  --model "$SERVED_NAME" \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --behaviors $BEHAVIORS \
  --patterns-per-behavior "$PATTERNS" \
  --prompts-per-pattern "$PROMPTS" \
  --samples-per-prompt "$SAMPLES" \
  --temperatures $TEMPS \
  --concurrency "$GEN_CONCURRENCY" \
  --judge-concurrency "$JUDGE_CONCURRENCY" \
  --max-tokens "$MAX_TOKENS"

echo "==> phase 7 analysis"
python phase7_arrhenius.py \
  --sweep "$DATA_DIR/temp_sweep_gpu.jsonl" \
  --output "$DATA_DIR/arrhenius_report_gpu.md" \
  --key behavior_id

echo
echo "==================== REPORT ===================="
cat "$DATA_DIR/arrhenius_report_gpu.md"
echo "================================================"
echo "sweep : $DATA_DIR/temp_sweep_gpu.jsonl"
echo "report: $DATA_DIR/arrhenius_report_gpu.md"
echo "Copy both off the box before terminating the instance."
