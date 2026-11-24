#!/usr/bin/env bash
# D10 on the CLT (the pre-registered primary test), on its own 32 GB server.
# Run from the parent of twinlens/ and circuit-tracer/. Resumable: a rerun skips finished prompts.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a logs/d10.log; }
step() {
  local name=$1; shift
  log "start $name"; local t0=$SECONDS
  /usr/bin/time -v "$@" >> "logs/$name.log" 2>&1; local rc=$?
  log "end $name rc=$rc seconds=$((SECONDS - t0)) peak_rss_kb=$(grep 'Maximum resident' "logs/$name.log" | tail -1 | awk '{print $6}')"
}
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
fi
command -v /usr/bin/time >/dev/null || (apt-get -qq update && apt-get -qq install -y time >/dev/null)
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -e ../circuit-tracer -e ".[dev]"
fi
PY=.venv/bin/python
log "setup done; $(nproc) cpus; $(free -g | awk '/Mem/{print $2}') GB"
step d10_clt_atp $PY experiments/circuits_sweep.py --tc clt --phase atp --per-family 20
step d10_clt_main $PY experiments/circuits_sweep.py --tc clt --phase main --per-family 20 --max-minutes 1200
log "all done"
