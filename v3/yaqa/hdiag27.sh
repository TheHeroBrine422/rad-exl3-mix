#!/bin/bash
# hdiag27.sh NSEQ [YAQA_PREC=x2 in the env] -- #33-13: the diagonal of YAQA Sketch B's output factor for Qwen3.8-27B (collect.py YAQA_DIAG=1,
# fp32), on the qad pool's 2048-token windows (pools-v3). Needs BOTH cards (TP2: the bf16 model is 54 GB): run only
# when the user has given card 1. Steps:
#   0. gate: 2 sequences, no files: time per sequence, peak memory, the chunked-vs-whole G check
#   1. collection, resumable (hdiag-state.pt every 25 sequences): a snapshot every 128 sequences (SNAPS); three
#      estimators per save (YAQA_DIAG_EXTRA): hdiag* (Sketch B), hdiag*-dy (sum dy^2), hdiag*-xw (sum ||x||^2 dy^2)
#      long qad windows only (#33-21: short docs neither help nor hurt, and more long windows do; YAQA_SHORT=256 adds
#      them and the -long / -dy-long sums). Past the 1,350 windows they repeat with fresh labels
#      (with YAQA_SHORT=256: + 2,041 qad docs of 256-2047 tokens, 3,391 sequences, one shuffle)
#      -> /data/yaqa/hdiag/hdiag-snapN, final /data/yaqa/hdiag/hdiag (ourq.py --hout reads "hout_diag")
# Stop early: touch /data/yaqa/hdiag/STOP (collect.py checks it before every sequence; rerun this script to resume). Costs: fp32 ~30 s / sequence (gate 2b) -> 1K ~8.5 h.
# Then (separate, pass rule first): requant K3 / K2.5 trunks with --hout <snapshot> on card 0 and swap-test them
# against the ladder's levels (d0.sh path), paired on a search-pool slice.
: ${RXM_WORK:=/root}; export RXM_WORK
N=${1:?NSEQ}; Y=/data/yaqa O=/data/yaqa/hdiag PY=/data/venv-exl3/bin/python
mkdir -p $O; log() { echo "$(date -u +%H:%M) $*" | tee -a $O/hdiag27.log; }
export YAQA_SHORT=${YAQA_SHORT:-0} YAQA_FP32=1 YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_SEED=0 YAQA_PREC=${YAQA_PREC:-fp32} TRITON_CACHE_AUTOTUNING=1   # YAQA_PREC=x2 after prectest.py passes
COL="exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock 7>${RXM_WORK}/ram.lock; flock 8; flock 9; flock -s 7"
# collect.py and bf16x.py are copied to $Y by the operator before the run (iceman has its own /workspace)
cd ${RXM_WORK}
if [ ! -f $O/gate.done ]; then
  ( eval "$COL"; cd $Y && YAQA_GATE=1 $PY -I collect.py $O 2 2048 ) > $O/gate.log 2>&1
  grep -q "gate check" $O/gate.log && grep -q "^DONE" $O/gate.log || { log "GATE FAIL (see gate.log)"; exit 1; }
  log "gate: $(grep 'gate check' $O/gate.log | tail -1); $(grep '^seq' $O/gate.log | tail -1 | cut -c1-120)"
  touch $O/gate.done
fi
SNAPS=${SNAPS:-$(seq -s, 128 128 3328),$N}   # a checkpoint every 128 sequences (~16 MB per estimator, 6 per snapshot: ~2.6 GB for all 27)
# one process for the whole run (the fp32 path has a ~15 min Triton warm-up per launch); a failure or a STOP
# leaves hdiag-state.pt, and running this script again continues from it
( eval "$COL"; cd $Y && YAQA_SNAP=$SNAPS YAQA_SAVE_EVERY=25 $PY -I collect.py $O $N 2048 ) >> $O/collect.log 2>&1
[ -f $O/STOP ] && { log "stopped by STOP file: $(grep '^seq' $O/collect.log | tail -1 | cut -c1-60)"; exit 0; }
grep -q "^DONE" <(tail -n 3 $O/collect.log) || { log "collect FAIL (see collect.log; rerun to resume)"; exit 1; }
log "HDIAG27 DONE: $N sequences -> $O/hdiag"
