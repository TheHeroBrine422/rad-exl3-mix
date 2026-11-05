#!/bin/bash
# knap.sh HEADCURVE.json OUTDIR [TREE] -- CPU (cpu1 / VM, ~20-40 min): knap4 (V3 table, quality only) at 2.0 / 2.5 / 3.0
# with and without the low-budget slab-head curve (knap4-headcurve.patch, KN_HEADCURVE), plus the no-op regression:
# patched knap4 without KN_HEADCURVE must equal the unpatched one byte for byte. TREE = a repo checkout (default the
# script's), whose v3/nf/knap4.py is the UNPATCHED one; the patched copy is made in OUTDIR.
HC=$(readlink -f $1) O=$(readlink -f $2) R=$(readlink -f ${3:-$(dirname $0)/../../..}); mkdir -p $O
cp $R/v3/nf/knap4.py $O/knap4-orig.py && cp $O/knap4-orig.py $O/knap4-hc.py
patch -s $O/knap4-hc.py $(dirname $(readlink -f $0))/knap4-headcurve.patch || { echo "patch FAIL"; exit 1; }
cd $R/v3/nf     # knap4 finds mkmask / embmenu relative to its own dir: run both copies from here
cp $O/knap4-orig.py .k4w05-orig.py; cp $O/knap4-hc.py .k4w05-hc.py
M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl KN_MODES=f16 KN_WRITE_CHECK=0
COMMON="KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1 KN_V3=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1"
T=2.0,2.5,3.0
env $COMMON python3 .k4w05-orig.py $M $A $F $P $O/orig.jsonl $T 0 > $O/orig.log 2>&1 &
env $COMMON python3 .k4w05-hc.py $M $A $F $P $O/nohc.jsonl $T 0 > $O/nohc.log 2>&1 &
env $COMMON KN_HEADCURVE=$HC python3 .k4w05-hc.py $M $A $F $P $O/hc.jsonl $T 0 > $O/hc.log 2>&1 &
wait; rm -f .k4w05-orig.py .k4w05-hc.py
cmp -s $O/orig.jsonl $O/nohc.jsonl && echo "REGRESSION OK: KN_HEADCURVE unset == unpatched" || echo "REGRESSION FAIL"
python3 - $O <<'EOF'
import json, sys
O = sys.argv[1]
a = {r["target"]: r for r in map(json.loads, open(f"{O}/orig.jsonl"))}
b = {r["target"]: r for r in map(json.loads, open(f"{O}/hc.jsonl"))}
for t in sorted(a):
    hc = {k: v for k, v in b[t]["param_share"].items() if k.startswith("ours-hc")}
    print(f"{t:.2f}: KLD {a[t]['kld']:.5f} -> {b[t]['kld']:.5f} ({b[t]['kld'] / a[t]['kld'] - 1:+.2%}); head picks {hc or 'no slab head'}")
EOF
