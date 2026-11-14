#!/bin/bash
# s3score.sh K [CARDS] -- RULES-177 stage 3 after s3level.sh, one K (2.625 | 4.125), both arms; restartable (each step skips
# when its output exists). Per arm (i = q27v3-144dd6e1-k<K>-i8, h = …-hash under /data/q27/w139/s3/):
#   pack   rad-convert of final.recipe with box2's hostd pack_cmd (rad10 home-pack, --no-error), conv.lock excl. +
#          ram.lock shared after MemAvailable >= 16 GiB (box2 / ds8 rule, #161)
#   mix    radmix of the pack with w138's a8.mix (act a8g128 on the 336 trunk linears), ram.lock shared; the pack is
#          deleted after the mix
#   check  (both arms mixed) radents.py: every non-trunk entry byte-identical, trunk byte counts equal -> check.json
#   score  rad10-perf radiance, home-base3 (arm i) / home-hq (arm h, EXL3I_BENCH_IQRH 1, A 0xd0474b), bf16 KV,
#          max-model-len 16384, max-num-seqs 1, on golden-search27c-c and golden-ver3-8k; card lock + ram.lock shared
# Out /data/q27/w139/s3/<level>/{mix-a8.rad,check.json,score-<golden>.json(.rows)}. (w139, #177)
set -u; K=$1; Q=/data/q27; H=$Q/w139; L=$Q/locks; S=$H/s3; RP=$Q/rad10-perf/install/bin/radiance
I=$S/q27v3-144dd6e1-k$K-i8; HH=$S/q27v3-144dd6e1-k$K-hash; LOG=$S/s3score-$K.log
say() { echo "$(date -u +%T) $*" >> $LOG; }
for D in $I $HH; do
  [ -f $D/DONE ] || { say "$(basename $D): s3level not done"; exit 1; }
  if [ ! -f $D/mix-a8.rad ]; then
    if [ ! -f $D/pack.rad ]; then
      ( flock -x 7; flock -s 6                     # box2 / ds8 rule (#161): conv.lock excl., ram.lock shared after RAM admission
        while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 16 ]; do sleep 20; done; say "pack $(basename $D)"
        cd $Q && RADIANCE_HOME=$Q/rad10/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 $Q/rad10/install/bin/rad-convert --no-error \
          models/qwen38-27b-bf16 --recipe $D/final.recipe --set quantization_config.quant_method=exl3 -o $D/pack.rad.part > $D/pack.log 2>&1 \
          && mv $D/pack.rad.part $D/pack.rad; say "pack $(basename $D): rc $?" ) 7>$L/conv.lock 6>$L/ram.lock
    fi
    [ -f $D/pack.rad ] || { say "PACK FAIL $D"; rm -f $D/pack.rad.part; exit 1; }
    ( flock -s 6; $Q/tools/radmix $D/mix-a8.rad.part $D/pack.rad $Q/w138/recipes/a8.mix > $D/mix.log 2>&1 && mv $D/mix-a8.rad.part $D/mix-a8.rad
      say "mix $(basename $D): $(tail -n 1 $D/mix.log)" ) 6>$L/ram.lock
    [ -f $D/mix-a8.rad ] && rm -f $D/pack.rad
  fi
done
if [ ! -f $S/check-$K.json ]; then
  python3 -I - $Q/w139/tools $I/mix-a8.rad $HH/mix-a8.rad $S/check-$K.json $Q/w138/recipes/a8.mix <<'EOF2' && say "check $K: $(cat $S/check-$K.json | cut -c1-200)" || { say "CHECK FAIL $K"; exit 1; }
import json, os, sys
sys.path.insert(0, sys.argv[1]); import radents as R
a, b, out = sys.argv[2:5]; A, B = R.entries(a), R.entries(b)
T = {l.split()[1] for l in open(sys.argv[5]) if l.startswith("act ")}          # the 336 exl3i trunk linears
names = sorted(set(A) | set(B)); tr = [n for n in names if n in T]; nt = [n for n in names if n not in T]
missing = [n for n in names if n not in A or n not in B]
nt_diff = [n for n in nt if n in A and n in B and (A[n][1] != B[n][1] or R.digest(a, A[n][0], A[n][1]) != R.digest(b, B[n][0], B[n][1]))]
tr_bytes = [n for n in tr if n in A and n in B and A[n][1] != B[n][1]]
r = {"entries": len(names), "trunk": len(tr), "non_trunk": len(nt), "missing": missing, "non_trunk_differ": nt_diff,
     "trunk_bytes_differ": tr_bytes, "size_a": os.path.getsize(a), "size_b": os.path.getsize(b)}
r["pass"] = not (missing or nt_diff or tr_bytes) and r["size_a"] == r["size_b"]
json.dump(r, open(out + ".part", "w")); os.replace(out + ".part", out); sys.exit(0 if r["pass"] else 1)
EOF2
fi
grep -q '"pass": true' $S/check-$K.json || { say "check $K not passed: no scores (RULES-177 confounder check)"; exit 1; }
for g in golden-search27c-c golden-ver3-8k; do
  for a in i h; do
    D=$I; hm=base3; [ $a = h ] && { D=$HH; hm=hq; }
    o=$D/score-$g; [ -f $o.json ] && continue
    while :; do for c in ${2:-3 2 1 0}; do exec 5>$L/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 30; done
    ( flock -s 6; say "score $(basename $D) $g card $c"
      ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$H/home-$hm $RP --model $D/mix-a8.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
        --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $Q/goldens/$g --kld-out $o.json.part > $o.log 2>&1 \
        && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json; say "score $(basename $D) $g rc $?" ) 6>$L/ram.lock
    exec 5>&-
  done
done
say "S3SCORE $K DONE"
