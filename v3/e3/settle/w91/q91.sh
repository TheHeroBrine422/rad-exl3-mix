#!/bin/bash
# q91.sh -- ../w91.md item 2 (rules written before data): 0.8B drift / variance split from the 6-seed plain levels
# (/data/yb/q-k{2,3}base{,_s1..s5}) + item 2b's bias oracle (s0-s2). 4070: one GPU unit at a time under
# /data/yb/gpu.lock (held per group: K2 dumps; K3 dumps + bias oracle; ~3-6 min each, so the queue does not alternate unit by unit with
# another queue's 15-min units), VRAM_CAP_GB 7.2. Idempotent (output files written .part -> rename); rerun after any kill.
# Runs from the snapshot /data/yb/w91/snap (re-snapshot after editing). Stop: touch /data/yb/w91/STOP.
O=/data/yb/w91 SN=$O/snap M=/data/models/qwen35-08b S=/data/g08-search1 F=/data/g08-fit1 NFIT=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} P2B=/workspace/rad-exl3-mix/proxy2b
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stop() { [ -f $O/STOP ] && { log "STOP"; exit 0; }; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
# dump K SEED: Delta dump + rows on search1; G3: rows == the stored /data/yb/k<K>base<sfx>.f32
dump() { local t=k$1base$(sfx $2); [ -f $O/$t.npy ] && [ -f $O/$t.f32 ] && return; stop
  log "$t: $(python3 $SN/drift91.py dump $M $S /data/yb/q-$t/wq $O/$t.npy $O/$t.f32 2>>$O/$t.err | tail -1)"
  [ -f $O/$t.f32 ] || { log "dump $t FAIL ($O/$t.err)"; return 1; }; }
g3() { local t=k$1base$(sfx $2)
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -c "
import numpy as np; a = np.fromfile('$O/$t.f32', np.float32).reshape(-1, 4); b = np.fromfile('/data/yb/$t.f32', np.float32).reshape(-1, 4)
d = float(np.abs(a[:, 0] - b[:, 0]).max()) if a.shape == b.shape else float('inf'); g = a[:, 2] > 0
print(f'G3 $t: max|dKL| {d:.3g} kld {a[g, 0].mean():.6f} vs stored {b[g, 0].mean():.6f} ->', 'PASS' if d <= 1e-6 else 'FAIL')"; }
bias() { local t=kb$1$(sfx $2); [ -f $O/$t.f32 ] && return; stop
  log "$t: $(python3 $SN/drift91.py dump $M $S /data/yb/q-k$1base$(sfx $2)/wq - $O/$t.f32 --bias $F $NFIT 2>>$O/$t.err | tail -1)"
  [ -f $O/$t.f32 ] || { log "bias $t FAIL ($O/$t.err)"; return 1; }; }
alldone() { for x in "$@"; do [ -f $x ] || return 1; done; }

# smoke: the bias path on 3 docs (fails fast before the long units)
if [ ! -f $O/smoke.ok ]; then stop; lock
  DRIFT_MAXDOCS=3 python3 $SN/drift91.py dump $M $S /data/yb/q-k2base/wq - $O/smoke-b.f32 --bias $F 4096 > $O/smoke.log 2>&1 \
    && DRIFT_MAXDOCS=3 python3 $SN/drift91.py dump $M $S /data/yb/q-k2base/wq $O/smoke.npy $O/smoke.f32 >> $O/smoke.log 2>&1 && touch $O/smoke.ok
  unlock; [ -f $O/smoke.ok ] || { log "smoke FAIL ($O/smoke.log)"; exit 1; }
  log "smoke: $(grep -h '"kld"' $O/smoke.log | tr '\n' ' ')"; rm -f $O/smoke*.f32 $O/smoke.npy
fi
# one hold for the K2 dumps, one for the K3 dumps + the bias oracle (each ~3-6 min)
alldone $(for s in 0 1 2 3 4 5; do echo $O/k2base$(sfx $s).npy; done) || {
  lock; for s in 0 1 2 3 4 5; do dump 2 $s || { unlock; exit 1; }; done; unlock; }
alldone $(for s in 0 1 2 3 4 5; do echo $O/k3base$(sfx $s).npy; done) $(for k in 2 3; do for s in 0 1 2; do echo $O/kb$k$(sfx $s).f32; done; done) || {
  lock; for s in 0 1 2 3 4 5; do dump 3 $s || { unlock; exit 1; }; done
  for k in 2 3; do for s in 0 1 2; do bias $k $s || { unlock; exit 1; }; done; done; unlock; }
for k in 2 3; do for s in 0 1 2 3 4 5; do g3 $k $s; done; done | tee $O/g3.txt | tee -a $O/result.txt
grep -q FAIL $O/g3.txt && { log "G3 FAIL: the split is not read"; }
for k in 2 3; do [ -f $O/split-k$k.json ] && continue; stop
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python3 $SN/drift91.py split $S $O/split-k$k.json \
    $(for s in 0 1 2 3 4 5; do echo $O/k${k}base$(sfx $s).npy; done) 2>>$O/split.err | tee -a $O/result.txt; done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 $SN/readout91.py $O | tee $O/readout.txt | tee -a $O/result.txt
log "Q91 DONE"
