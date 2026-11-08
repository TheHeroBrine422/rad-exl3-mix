#!/bin/bash
# w93 #123 integrity scan on colo (../w93.md item 1): nice 19, one thread a file, ffn_down (k 17408, ~3.6 GB in the
# Cholesky) one at a time beside 4 lanes of the rest (<= 8 GB RSS in total). Restartable: finished OUT.json skipped.
#   bash scan93.sh            from /data/work/w13/w93 (code in ./code, outputs in ./scan)
set -u
cd /data/work/w13/w93
HS=/data/challenge/hessians/calib; CK=/data/models/qwen38-27b-bf16; HO=/data/work/hout27/hout-bn
PY="nice -n 19 python3 -I code/scan93.py"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p scan/hin logs
log() { echo "$(date -u +%H:%M:%S) $*" >> logs/scan.log; }
log "start"
$PY selftest scan/selftest > logs/selftest.log 2>&1 || { log "SELFTEST FAIL"; exit 1; }
log "selftest PASS"
$PY hout $HO code/keys400.txt $CK scan/hout-colo.json > logs/hout.log 2>&1; log "hout rc $?"
one() { f=$1; o=scan/hin/$(basename $f .safetensors).json; [ -s $o ] || $PY hin $f $CK $o >> logs/hin.log 2>&1 || echo "FAIL $f" >> logs/scan.log; }
export -f one; export PY CK
ls $HS/*mlp.down_proj.input.safetensors | xargs -P 1 -I{} bash -c 'one {}' &
ls $HS/*.safetensors | grep -v mlp.down_proj.input | xargs -P 4 -I{} bash -c 'one {}'
wait
log "hin files done: $(ls scan/hin/*.json | wc -l)"
log "SCAN93 DONE"
