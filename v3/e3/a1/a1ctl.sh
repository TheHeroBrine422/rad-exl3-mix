#!/bin/bash
# a1ctl.sh "LAYERS" P -- #81-02 control arms (ugperm, uperm at K2.5 / K3; stage-1 predictors already on disk), then
# #81-03 if the control clears (README): stage 1 + arms D / ug / u at K2.5 for the extra layers in $A1_MORE.
cd /data/work/e3t; export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data/work/a1; L="$1"; PD=${2:-8}
have() { [ -f $A/a1.jsonl ] && grep -q "\"layer\": $1, \"w\": \"ffn_down\", \"arm\": \"$3\", \"K\": $(python3 -c "print(float('$2'))")," $A/a1.jsonl; }
for l in $L; do for K in 2.5 3; do for arm in ugperm uperm; do
  have $l $K $arm || echo "E3_FACTORS=1.0,1.15 python3 -W ignore a1test.py down $l $K $arm $A/gu $A/a1.jsonl >> $A/logs/down-$l-$K-$arm.log 2>&1"
done; done; done | xargs -P $PD -I{} nice -n 10 bash -c "{}"
echo "$(date -u +%T) A1CTL DONE"
[ -n "$A1_MORE" ] || exit 0
python3 - $A/a1.jsonl <<'PY' || { echo "control not cleared: no #81-03"; exit 0; }
import json, sys
R = {}
for l in open(sys.argv[1]):
    r = json.loads(l); R[(r["layer"], r["K"], r["arm"])] = r["rel_D"]
x = [R[(l, 2.5, "ugperm")] / R[(l, 2.5, "D")] for (l, k, a) in R if k == 2.5 and a == "ugperm" and (l, 2.5, "D") in R]
y = [R[(l, 2.5, "ug")] / R[(l, 2.5, "D")] for (l, k, a) in R if k == 2.5 and a == "ugperm" and (l, 2.5, "ug") in R]
mx, my = sum(x) / len(x), sum(y) / len(y)
print(f"control ugperm/D {mx:.3f}, ug/D {my:.3f}; real share of the reduction {(1 - my - (1 - mx)) / (1 - my):.2f}")
sys.exit(0 if (1 - mx) <= 0.5 * (1 - my) else 1)
PY
echo "$(date -u +%T) #81-03 stage 1"
for l in $A1_MORE; do for r in $(seq 0 1024 34815); do
  [ -f $A/gu/L$l-K2.5/gu-$(printf %05d $r).npy ] || echo "E3_FACTORS=1.0 python3 -W ignore a1test.py uq $l 2.5 $r $((r + 1024)) $A/gu >> $A/logs/uq-$l-2.5.log 2>&1"
done; done | xargs -P ${PU:-36} -I{} nice -n 10 bash -c "{}"
echo "$(date -u +%T) #81-03 stage 2"
for l in $A1_MORE; do for arm in D ug u; do
  have $l 2.5 $arm || echo "E3_FACTORS=1.0,1.15 python3 -W ignore a1test.py down $l 2.5 $arm $A/gu $A/a1.jsonl >> $A/logs/down-$l-2.5-$arm.log 2>&1"
done; done | xargs -P $PD -I{} nice -n 10 bash -c "{}"
echo "$(date -u +%T) A1MORE DONE"
