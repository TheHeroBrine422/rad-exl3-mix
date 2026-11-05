#!/bin/bash
# more.sh -- d1110 follow-up (user, 2026-11-05): spare 0.8B goldens built up front, all disjoint (docs excluded via
# the existing goldens' "orig" indices). Not scored against until a decision needs them (record each first use in LOG).
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/d1110 M=/data/models/qwen35-08b CM=/data/challenge/calib-meta O=/data/yb/d1110
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
calib() { g=$1 pool=$2 n=$3 seed=$4; shift 4; [ -f $g/rows.f32 ] && return
  idx=$(python3 -I - "$pool" "$n" "$seed" "$@" <<'PY'
import json, random, sys, os
pool, n, seed, exc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
m = json.load(open("/data/challenge/calib-meta/kld.json")); p = json.load(open("/workspace/rad-exl3-mix/eval/pools/pools-v3.json"))["pool"]
used = {d["orig"] for g in exc if os.path.exists(f"{g}/kld.json") for d in json.load(open(f"{g}/kld.json"))["docs"]}
v = [i for i, d in enumerate(m["docs"]) if p[i] == pool and i not in used]; random.Random(seed).shuffle(v)
print(" ".join(f"{i}:2048" for i in sorted(v[:n])))
PY
)
  lock; python3 $P/mkgold.py $M $CM $g $idx > $O/gold-$(basename $g).log 2>&1; unlock; log "$(basename $g): $(tail -1 $O/gold-$(basename $g).log)"; }
xc() { g=$1 seed=$2; shift 2; [ -f $g/rows.f32 ] && return; d=$O/xmeta-$(basename $g)
  [ -f $d/kld.json ] || CUDA_VISIBLE_DEVICES= python3 -I $X/xgold.py /data/xcorp/v1/meta $d 180 $seed 2048 "$@" >> $O/gold-$(basename $g).log 2>&1
  n=$(python3 -c "import json; print(len(json.load(open('$d/kld.json'))['docs']))")
  lock; python3 $P/mkgold.py $M $d $g $(seq 0 $((n - 1))) >> $O/gold-$(basename $g).log 2>&1; unlock; log "$(basename $g): $(tail -1 $O/gold-$(basename $g).log)"; }
while [ ! -f /data/g08-verify1/rows.f32 ] || [ ! -f /data/g08-xcorp1/rows.f32 ]; do sleep 120; done   # after d1110's own goldens
calib /data/g08-search2 search 170 2 /data/g08-search1
calib /data/g08-search3 search 170 3 /data/g08-search1 /data/g08-search2
calib /data/g08-verify2 verify 180 2 /data/g08-verify1
xc /data/g08-xcorp2 2 /data/yb/d1110/xmeta; xc /data/g08-xcorp3 3 /data/yb/d1110/xmeta /data/yb/d1110/xmeta-g08-xcorp2
xc /data/g08-xcorp4 4 /data/yb/d1110/xmeta /data/yb/d1110/xmeta-g08-xcorp2 /data/yb/d1110/xmeta-g08-xcorp3
log "MORE GOLDENS DONE"
