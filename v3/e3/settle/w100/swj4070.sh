#!/bin/bash
# swj4070.sh -- w100 #128-01 (../w100.md item 4, "swj at sub-1", redteam R18): WRITTEN, NOT RUN (the 4070 is off
# limits since 11-07 7:30 AM CST; run only when the lead / user frees it, under /data/yb/gpu.lock, VRAM_CAP_GB 7.2).
# 0.8B, i8, K0.5 and K1, seeds 0 / 1, d1103's stack helpers (q2b.sh: qtf2, scaled Hessian, sub-1 damping, H_out c18
# Σdy² as d1103 "edy"): base quant vs the same codes with swj (w02-fixedscale/swiglufit.py on g08-fit1, the fit pool),
# paired KL on g08-search1 (pcmp.py). Fixed codes: the pair differs only in the gate / up output scales.
# Idempotent (finished quants / scores are skipped); stop: touch /data/yb/night.stop. Est. ~1 h (4 quants ~8 min,
# 4 swj fits ~3 min, 8 evals ~2 min each).
# Rule (../w100.md #128-01): per K, r = geo mean over the 2 seeds of KL(swj) / KL(base). swj keeps K if r <= 1.000
# and neither seed > 1.005; otherwise propose swj k_min above that K (v3.toml [post.swj], lead decides).
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X2=$P/exp/w02-fixedscale
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/w100swj; mkdir -p $O
f32() { echo $O/$1.f32; }
wqd() { [ $(nq $O/q-$1/wq) -ge 150 ] && echo $O/q-$1/wq; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }
unlock() { flock -u 9; }
step swj-sub1
for k in 0.5 1; do for s in 0 1; do
  [ -f /data/yb/night.stop ] && break 2
  b=k${k}edyi8$(sfx $s); sw=${b}_swj
  lock
  [ -n "$(wqd $b)" ] || q $b $k i8 $HO --seed-offset $s || { unlock; continue; }
  [ -f $O/$b.f32 ] || ev $b $(wqd $b)
  if [ ! -f $O/q-$sw/.done ]; then src=$O/q-${b}src; mkdir -p $src; ln -sfn $(wqd $b) $src/wq; rm -rf $O/q-$sw $O/q-${b}_swu
    python3 $X2/swiglufit.py $M $src $F $O/q-$sw $O/q-${b}_swu > $O/q-$sw.log 2>&1 && touch $O/q-$sw/.done || { log "swiglufit FAIL $b"; unlock; continue; }; fi
  ev $sw $O/q-$sw/wq; pair $b $sw
  unlock
done; done
grep "^.* pair " $O/result.txt   # the four pair lines (ratio per seed) decide the rule
log "W100 SWJ DONE"
