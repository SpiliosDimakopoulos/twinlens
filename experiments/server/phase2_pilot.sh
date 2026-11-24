#!/usr/bin/env bash
# Plan v2, phase 2 pilot (one server session; run from the parent of twinlens/ and circuit-tracer/).
#   1. attribution patching on Gemma-2-2B (PLT and CLT) with the fixed gradient, and the D10-C
#      detach variants, on the 56 development prompts;
#   2. the D9 external prompt sets (GPT-2 and Gemma);
#   3. the D10 pilot: one prompt per family, CLT then PLT, with time and peak memory.
# A failing step is logged and the next one still runs. Outputs: twinlens/experiments/results,
# twinlens/experiments/prompts_external_*.json, twinlens/logs.
set -uo pipefail
cd "$(dirname "$0")/../.."  # twinlens/
mkdir -p logs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a logs/pilot.log; }
step() {  # step NAME CMD...: run, record seconds, exit code and peak RSS
  local name=$1; shift
  log "start $name"
  local t0=$SECONDS
  /usr/bin/time -v "$@" > "logs/$name.log" 2>&1
  local rc=$?
  local rss
  rss=$(grep "Maximum resident" "logs/$name.log" | awk '{print $6}')
  log "end $name rc=$rc seconds=$((SECONDS - t0)) peak_rss_kb=${rss:-?}"
}

if ! swapon --show | grep -q /swapfile; then  # TransformerLens holds two copies while loading
  fallocate -l 24G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
fi
command -v /usr/bin/time >/dev/null || (apt-get -qq update && apt-get -qq install -y time >/dev/null)
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e ../circuit-tracer -e ".[dev]"
PY=.venv/bin/python
log "setup done; $(nproc) cpus; $(free -g | awk '/Mem/{print $2}') GB"

step clt_prepare $PY experiments/gemma_clt_prepare.py  # the twin reads the CLT from its own cache
for tc in plt clt426k; do
  for d in none ln attn both; do
    step "atp_${tc}_${d}" $PY experiments/atp_baseline.py --model gemma2 --transcoders "$tc" --detach "$d"
  done
done

step prompts_gemma2 $PY experiments/build_external_prompts.py --model gemma2
step prompts_gpt2 $PY experiments/build_external_prompts.py --model gpt2

for tc in clt plt; do
  step "d10_${tc}_atp" $PY experiments/circuits_sweep.py --tc "$tc" --phase atp --per-family 1 --tag _pilot
  step "d10_${tc}_main" $PY experiments/circuits_sweep.py --tc "$tc" --phase main --per-family 1 \
    --tag _pilot --max-minutes 150
done
log "all done"
