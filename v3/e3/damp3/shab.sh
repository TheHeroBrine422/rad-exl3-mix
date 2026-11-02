#!/bin/bash
# shab.sh CARD -- scaled Hessian (-sh) A/B, held out (trunk review item 1): layers 3/10/43 (split-Hessian layers),
# quantized on H_A at K2.5 / K4 (i8, qtf2, fixed sigma 0.005), -sh off vs on; d2score on H_A / H_B / published H.
# Out /data/damp3/sh/score.jsonl (arms K<k>-sh<0|1>). Rule (fixed before the run, review + ours): adopt -sh if the
# held-out (H_B) geo-mean ratio on/off over all scored tensors is <= 0.99 at both K, and no family (attn q/k/v, GDN
# qkv/z, o/out_proj, gate/up) is worse than 1.005 at either K.
: ${RXM_WORK:=/root}
C=${1:-0} O=/data/damp3/sh PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src HS=/data/d2/hsplit
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/run.log; }
cp /data/damp3/ourq.py.sh /data/damp3/ourq/ourq.py
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
for sh in 0 1; do
  F=""; [ $sh = 1 ] && F="--scaled-hessian"
  s=$(date +%s)
  $PY -I /data/damp3/ourq/ourq.py -K 2.5,4 -o "$O/sh$sh/K{K}" --layers 3-43 --only /data/damp3/only.txt --hin $HS/hA \
    --sigma-reg 0.005 --save-wq --device 0 --codebook i8 --qtf2 --frac-ext $SRC $F > $O/sh$sh.log 2>&1
  log "sh=$sh rc=$? $(( $(date +%s) - s ))s"
  $PY -I /data/damp3/d2score.py $O/score.jsonl 3,10,43 $HS/hA $HS/hB $RXM_WORK/challenge/hessians/calib \
    K2.5-sh$sh:ours:$O/sh$sh/K2.5 K4-sh$sh:ours:$O/sh$sh/K4 > $O/score-sh$sh.log 2>&1
done
python3 - $O/score.jsonl <<'PY' | tee -a $O/run.log
import json, sys, math, collections
r = [json.loads(l) for l in open(sys.argv[1])]; d = collections.defaultdict(dict)
for x in r: K, sh = x["arm"][1:].split("-sh"); d[(x["key"], K)][sh] = x
fam = lambda k: {"q_proj": "attn qkv", "k_proj": "attn qkv", "v_proj": "attn qkv", "in_proj_qkv": "gdn qkvz", "in_proj_z": "gdn qkvz",
                 "o_proj": "out", "out_proj": "out", "gate_proj": "gate/up", "up_proj": "gate/up"}[k.split(".")[-1]]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
ok = True
for K in ("2.5", "4"):
    c = {k: v for (k, KK), v in d.items() if KK == K and "0" in v and "1" in v}
    for H in ("A", "B", "pub"):
        print(f"K{K} on H_{H}: on/off {g([v['1'][f'rel_{H}'] / v['0'][f'rel_{H}'] for v in c.values()]):.4f}")
    allB = g([v["1"]["rel_B"] / v["0"]["rel_B"] for v in c.values()]); ok &= allB <= 0.99
    for f in sorted({fam(k) for k in c}):
        fb = g([v["1"]["rel_B"] / v["0"]["rel_B"] for k, v in c.items() if fam(k) == f]); ok &= fb <= 1.005
        print(f"   {f:9s} H_B on/off {fb:.4f}")
print("RULE: ADOPT -sh" if ok else "RULE: keep -sh off")
PY
rm -rf $O/sh*/K*/wq
