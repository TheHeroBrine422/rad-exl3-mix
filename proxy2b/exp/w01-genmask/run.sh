#!/bin/bash
# run.sh -- w01-genmask (ranked.md A1): the 1K Sketch-B diagonal with the sampled-label loss on the scored (gen)
# positions only, vs the all-position one (yaqaFact7's c18, same windows / seed / labels), 0.8B, K3 / K2 x 3 seeds.
# 4070, one GPU, idempotent and resumable (rerun after any failure). Needs genmask.patch (and the w01-failloud
# patches, optional) merged into v3/yaqa/collect.py.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w01-genmask M=/data/models/qwen35-08b G=/data/g08-search1
Y=/data/yb O=/data/yb/w01-genmask C=/data/yb/w01-genmask/cmask CL=/workspace/rad-exl3-mix/v3/yaqa/collect.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
f32() { [ -f $Y/$1.f32 ] && echo $Y/$1.f32 || echo $O/$1.f32; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $(f32 $1) $(f32 $2) | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
grep -q YAQA_GENMASK $CL || { log "collect.py has no YAQA_GENMASK: merge $X/genmask.patch first"; exit 1; }
[ -f $Y/c18/hdiag-snap1024/halves.json ] || { log "control diagonal $Y/c18/hdiag-snap1024 missing: run yaqaFact7.sh's collection first"; exit 1; }
W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
[ -e $W/challenge/golden/calib/gen.u8 ] || { log "no gen.u8 in $W/challenge/golden/calib"; exit 1; }
COL="RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_SEED=0"
cd /tmp
# 0. GPU no-op gate: patched collector, mask off, 128 sequences == c18's 128-sequence snapshot (~3 min)
if [ ! -f $O/noop.done ]; then
  rm -rf $O/noop; env $COL YAQA_SNAP=128 YAQA_SAVE_EVERY=64 python3 $CL $O/noop 128 2048 > $O/noop.log 2>&1
  python3 - $Y/c18/hdiag-snap128 $O/noop/hdiag-snap128 <<'PY' | tee -a $O/result.txt
import os, sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; fs = sorted(f for f in os.listdir(a) if f.endswith(".safetensors")); worst, exact = 0.0, True
assert len(fs) == 150 and sorted(f for f in os.listdir(b) if f.endswith(".safetensors")) == fs, "file sets differ"
for f in fs:
    with safe_open(f"{a}/{f}", "pt") as h: x = h.get_tensor("hout_diag")
    with safe_open(f"{b}/{f}", "pt") as h: y = h.get_tensor("hout_diag")
    exact &= torch.equal(x, y); worst = max(worst, float((x - y).norm() / x.norm()))
print(f"noop gate: 150 linears, bit-exact {exact}, max rel L2 {worst:.2e} -> {'PASS' if worst <= 1e-6 else 'FAIL'}")
sys.exit(0 if worst <= 1e-6 else 1)
PY
  [ ${PIPESTATUS[0]} = 0 ] || { log "NOOP GATE FAIL: the patched collector changed the unmasked diagonal"; exit 1; }
  touch $O/noop.done
fi
# 1. the masked collection (~25 min; windows without a scored position are skipped, ~10%)
if [ ! -f $C/hdiag-snap1024/halves.json ]; then
  env $COL YAQA_GENMASK=1 YAQA_SNAP=256,512,1024 YAQA_SAVE_EVERY=64 python3 $CL $C 1024 2048 >> $O/cmask.log 2>&1
  grep -q "^DONE" $O/cmask.log || { log "COLLECT FAIL (see cmask.log; rerun resumes)"; exit 1; }
  log "collected: $(grep '^DONE' $O/cmask.log | tail -1); skipped windows $(grep -c 'no scored position' $O/cmask.log)"
fi
# 2. quants: masked arm x 3 seeds; controls (yaqaFact7's eB arms, base) quantized here only if absent in /data/yb
for s in 0 1 2; do for k in 3 2; do
  sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  [ -f $Y/k${k}eB$sfx.f32 ] || q k${k}eB$sfx $k --hout $Y/c18/hdiag-snap1024 --sigma-reg-out 0.01 --seed-offset $s
  [ -f $Y/k${k}base$sfx.f32 ] || q k${k}base$sfx $k --seed-offset $s
  q k${k}gm$sfx $k --hout $C/hdiag-snap1024 --sigma-reg-out 0.01 --seed-offset $s
  pair k${k}eB$sfx k${k}gm$sfx
done; done
python3 $X/summ.py $G $O $Y | tee -a $O/result.txt
log GENMASK DONE
