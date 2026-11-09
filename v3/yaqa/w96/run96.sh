#!/bin/bash
# run96.sh [STEP...] -- w96 (#127) on Thor: block sums for the search-gen (held out) and qad-dump (tune) rows, G1,
# census, tune, final, curve of the best arm (and base). Restartable: finished block files / outputs are skipped.
set -u
W=/data/work/w96 C=/data/work/w96/code/v3/yaqa/w96 CK=/data/models/qwen38-27b-bf16
export W CODE=/data/work/w96/code/v3/yaqa CK OMP_NUM_THREADS=24
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run96.log; }
cd $W
for k in 3 4 5 6 7; do [ -f in/K$k/.done ] || { log "donor K$k not copied yet"; exit 1; }; done
for s in search qad; do
  T=24 nice -n 10 python3 -I $C/score96.py in/dump/$s in/q in/hs/perm.npy $CK sums/$s 512 >> $W/score-$s.log 2>&1 \
    || { log "score $s FAIL"; exit 1; }
  log "sums $s done"
done
P="nice -n 10 python3 -I $C/plans96.py"
[ -s out/g1.json ] || $P g1 || { log "g1 crashed"; exit 1; }
python3 -c "import json, sys; sys.exit(0 if json.load(open('out/g1.json'))['pass'] else 1)" || { log "G1 FAIL: stop"; exit 1; }
[ -s out/census.json ] || $P census || { log "census crashed"; exit 1; }
[ -s out/tune.json ] || $P tune || { log "tune crashed"; exit 1; }
[ -s out/final.json ] || $P final || { log "final crashed"; exit 1; }
best=$(python3 -c "import json; print(json.load(open('out/final.json'))['best'])")
for a in base $best; do [ -s out/sim-$a/sim-search.npz ] || $P curve $a || { log "curve $a crashed"; exit 1; }; done
log "RUN96 DONE (best $best)"
