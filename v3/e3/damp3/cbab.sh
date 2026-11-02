#!/bin/bash
# cbab.sh CARD -- i8 vs plain mul1 codebook at K6-6.75 (user: keep plain K6-6.75 in the ladder only if they differ).
# Layers 3/10/43 on H_A (fixed sigma 0.005, qtf2), d2score on H_A / H_B. Rule (fixed before the run): drop plain
# K6-6.75 from the queue if the held-out geo-mean i8/plain ratio is <= 1.005 at every K and no family exceeds 1.01.
: ${RXM_WORK:=/root}
C=${1:-1} O=/data/damp3/cb PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src HS=/data/d2/hsplit
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
while pgrep -f "^/bin/bash ./damp3.sh 1 0.05" > /dev/null; do sleep 10; done
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
A=""
for cb in i8 mul1; do
  $PY -I /data/damp3/ourq/ourq.py -K 6,6.25,6.5,6.75 -o "$O/$cb/K{K}" --layers 3-43 --only /data/damp3/only.txt --hin $HS/hA \
    --sigma-reg 0.005 --save-wq --device 0 --codebook $cb --qtf2 --frac-ext $SRC > $O/$cb.log 2>&1; log "$cb rc=$?"
  for K in 6 6.25 6.5 6.75; do A="$A K$K-$cb:ours:$O/$cb/K$K"; done
done
$PY -I /data/damp3/d2score.py $O/score.jsonl 3,10,43 $HS/hA $HS/hB $RXM_WORK/challenge/hessians/calib $A > $O/score.log 2>&1
python3 - $O/score.jsonl <<'PY' | tee -a $O/run.log
import json, sys, math, collections
d = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    x = json.loads(l); K, cb = x["arm"][1:].split("-"); d[(x["key"], K)][cb] = x
fam = lambda k: {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "in_proj_qkv": "gdn", "in_proj_z": "gdn",
                 "o_proj": "out", "out_proj": "out", "gate_proj": "gate/up", "up_proj": "gate/up"}[k.split(".")[-1]]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)); ok = True
for K in ("6", "6.25", "6.5", "6.75"):
    c = {k: v for (k, KK), v in d.items() if KK == K and len(v) == 2}
    r = g([v["i8"]["rel_B"] / v["mul1"]["rel_B"] for v in c.values()]); ok &= r <= 1.005
    fr = {f: g([v["i8"]["rel_B"] / v["mul1"]["rel_B"] for k, v in c.items() if fam(k) == f]) for f in sorted({fam(k) for k in c})}
    ok &= max(fr.values()) <= 1.01
    print(f"K{K}: i8/plain on H_B {r:.4f}; " + ", ".join(f"{f} {x:.4f}" for f, x in fr.items()))
print("RULE: drop plain K6-6.75" if ok else "RULE: keep plain K6-6.75")
PY
rm -rf $O/*/K*/wq
