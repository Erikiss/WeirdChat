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
# OpenRouter is only needed when the judge stays remote. If JUDGE_MODEL_PATH is
# set (ultra / self-hosted judge), no OpenRouter key is required.
if [ -z "${JUDGE_MODEL_PATH:-}" ]; then
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (remote reference judge); or set JUDGE_MODEL_PATH to self-host it}"
fi
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [ -n "${OPENROUTER_API_KEY:-}" ]; then export OPENROUTER_API_KEY; fi

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
SUBJECT_GPU="${SUBJECT_GPU:-0}"            # which GPU the subject server binds to

# Optional local judge (ultra / 2-GPU mode). Leave JUDGE_MODEL_PATH empty to
# keep the judge on OpenRouter (the tested single-GPU path). Set it to serve
# Gemma on a second GPU, which removes the OpenRouter rate limit *and* the
# per-call judge bill — the real money sink at high sample counts.
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-google/gemma-4-31b-it}"  # keep = dataset judge slug
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_GPU="${JUDGE_GPU:-1}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-12288}"  # judge prompt + up to 8192 out
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"  # e.g. low; empty keeps the dataset default
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

# Poll a vLLM server's /v1/models until ready, aborting if the process dies.
# args: <pid> <port> <logfile> <label>
wait_ready() {
  local pid="$1" port="$2" logf="$3" label="$4" i
  echo "==> waiting for $label to load (first run downloads weights; up to ~30 min)"
  for i in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label process died during startup. Last log lines:"; tail -40 "$logf"; return 1
    fi
    if curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then echo "==> $label ready"; return 0; fi
    sleep 10
  done
  echo "$label did not become ready in time. Log tail:"; tail -40 "$logf"; return 1
}

echo "==> starting subject vLLM ($MODEL_PATH -> '$SERVED_NAME') on GPU $SUBJECT_GPU port $PORT"
VLLM_LOG="/tmp/vllm_$PORT.log"
CUDA_VISIBLE_DEVICES="$SUBJECT_GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Optionally start the judge on a second GPU.
JUDGE_PID=""
if [ -n "$JUDGE_MODEL_PATH" ]; then
  echo "==> starting judge vLLM ($JUDGE_MODEL_PATH -> '$JUDGE_SERVED_NAME') on GPU $JUDGE_GPU port $JUDGE_PORT"
  JUDGE_LOG="/tmp/vllm_$JUDGE_PORT.log"
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL_PATH" \
    --served-model-name "$JUDGE_SERVED_NAME" \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --trust-remote-code \
    >"$JUDGE_LOG" 2>&1 &
  JUDGE_PID=$!
fi

# Always shut both servers down on exit (success, error, budget timeout, Ctrl-C).
# INT/TERM are included so the SkyPilot budget-guard `timeout` still cleans up.
trap 'echo "==> stopping vLLM servers"; kill "$VLLM_PID" ${JUDGE_PID:+"$JUDGE_PID"} 2>/dev/null || true' EXIT INT TERM

wait_ready "$VLLM_PID" "$PORT" "$VLLM_LOG" "subject" || exit 1
if [ -n "$JUDGE_PID" ]; then
  wait_ready "$JUDGE_PID" "$JUDGE_PORT" "$JUDGE_LOG" "judge" || exit 1
fi

# Point the judge at the local server when we brought one up, else OpenRouter.
JUDGE_ARGS=()
if [ -n "$JUDGE_MODEL_PATH" ]; then
  echo "==> phase 7 sweep (generation AND judge local)"
  JUDGE_ARGS=(--judge-base-url "http://localhost:$JUDGE_PORT/v1" --judge-api-key-env NONE --judge-model "$JUDGE_SERVED_NAME")
  [ -n "$JUDGE_REASONING_EFFORT" ] && JUDGE_ARGS+=(--judge-reasoning-effort "$JUDGE_REASONING_EFFORT")
else
  echo "==> phase 7 sweep (generation local, judge on OpenRouter)"
fi
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
  --max-tokens "$MAX_TOKENS" \
  "${JUDGE_ARGS[@]}"

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
