#!/usr/bin/env bash
# Plan v2, phase 2 full run (32 GB server; run from the parent of twinlens/ and circuit-tracer/).
# In order of importance, each step capped in minutes and resumable (a rerun skips finished prompts):
#   1. D10 on the CLT (the pre-registered primary test): AtP gradients, then graphs + interventions;
#   2. D9 external sweeps (GPT-2, Gemma PLT, Gemma CLT) and attribution patching on their edits;
#   3. D10 on the PLT.
# Needs experiments/prompts_external_{gpt2,gemma2}.json (from the pilot). A failing step is logged
# and the next one still runs.
set -uo pipefail
cd "$(dirname "$0")/../.."  # twinlens/
mkdir -p logs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a logs/full.log; }
step() {
  local name=$1; shift
  log "start $name"
  local t0=$SECONDS
  /usr/bin/time -v "$@" >> "logs/$name.log" 2>&1
  local rc=$?
  local rss
  rss=$(grep "Maximum resident" "logs/$name.log" | tail -1 | awk '{print $6}')
  log "end $name rc=$rc seconds=$((SECONDS - t0)) peak_rss_kb=${rss:-?}"
}

if ! swapon --show | grep -q /swapfile; then
  fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
fi
command -v /usr/bin/time >/dev/null || (apt-get -qq update && apt-get -qq install -y time >/dev/null)
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -e ../circuit-tracer -e ".[dev]"
fi
PY=.venv/bin/python
log "setup done; $(nproc) cpus; $(free -g | awk '/Mem/{print $2}') GB"
step clt_prepare $PY experiments/gemma_clt_prepare.py

# 1. D10, CLT (primary)
step d10_clt_atp $PY experiments/circuits_sweep.py --tc clt --phase atp --per-family 20
step d10_clt_main $PY experiments/circuits_sweep.py --tc clt --phase main --per-family 20 --max-minutes 600

# 2. D9 external sweeps, then attribution patching on the same edits
step ext_gpt2 $PY experiments/fidelity_sweep.py --model gpt2 --prompts external --max-minutes 120
step ext_gemma2_clt $PY experiments/fidelity_sweep.py --model gemma2 --transcoders clt426k \
  --prompts external --n-top 3 --n-random 3 --max-minutes 360
step ext_gemma2 $PY experiments/fidelity_sweep.py --model gemma2 --prompts external \
  --n-top 3 --n-random 3 --max-minutes 360
step ext_atp_gpt2 $PY experiments/atp_baseline.py --model gpt2 --prompts external
step ext_atp_gemma2_clt $PY experiments/atp_baseline.py --model gemma2 --transcoders clt426k --prompts external
step ext_atp_gemma2 $PY experiments/atp_baseline.py --model gemma2 --prompts external

# 3. D10, PLT
step d10_plt_atp $PY experiments/circuits_sweep.py --tc plt --phase atp --per-family 20
step d10_plt_main $PY experiments/circuits_sweep.py --tc plt --phase main --per-family 20 --max-minutes 300
log "all done"
