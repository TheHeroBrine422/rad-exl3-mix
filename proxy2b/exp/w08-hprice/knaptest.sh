#!/bin/bash
# knaptest.sh -- CPU tests of knap4-hratio.patch (colo, ~20-40 min, background). Run from a tree with v3/nf/knap4_old.py
# (the attached knap4) and v3/nf/knap4.py (patched), data/v3, v3/e1b, v3/e3/data, v3/bench/data, ident.jsonl,
# demo08.jsonl (mkratio.py). Out: ./kt/.
#  1. regression: switch off == old knap4, byte for byte (V2-like menu, and the V3 table)
#  2. identity: KN_HRATIO with r = 1 everywhere, KN_HF 1, KN_HONLY 1 == the V3 run without it (same KLD, bpw, time)
#  3. illustration: the 0.8B's per-type ratios on the 27B tensors (K2-3.5), KN_HF 1.07, both ladders offered
cd "$(dirname "$0")"/v3/nf; O=../../kt; mkdir -p $O
M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1
V3="KN_V3=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1"
python3 knap4_old.py $M $A $F $P $O/old.jsonl 2.5,4.0 0,3e-6 > $O/old.log 2>&1 &
python3 knap4.py $M $A $F $P $O/new.jsonl 2.5,4.0 0,3e-6 > $O/new.log 2>&1 &
env $V3 python3 knap4_old.py $M $A $F $P $O/v3old.jsonl 2.5,3.0 0,3e-6 > $O/v3old.log 2>&1 &
env $V3 python3 knap4.py $M $A $F $P $O/v3new.jsonl 2.5,3.0 0,3e-6 > $O/v3new.log 2>&1 &
env $V3 KN_HRATIO=../../ident.jsonl KN_HF=1 KN_HONLY=1 python3 knap4.py $M $A $F $P $O/v3ident.jsonl 2.5,3.0 0,3e-6 > $O/v3ident.log 2>&1 &
env $V3 KN_HRATIO=../../demo08.jsonl KN_HF=1.07 python3 knap4.py $M $A $F $P $O/v3demo.jsonl 2.5,3.0 0,3e-6 > $O/v3demo.log 2>&1 &
wait
cmp -s $O/old.jsonl $O/new.jsonl && echo "1a REGRESSION OK: identical" || echo "1a REGRESSION FAIL"
cmp -s $O/v3old.jsonl $O/v3new.jsonl && echo "1b REGRESSION (V3) OK: identical" || echo "1b REGRESSION (V3) FAIL"
python3 - $O <<'PY'
import json, sys
O = sys.argv[1]; rd = lambda f: [json.loads(l) for l in open(f"{O}/{f}")]
a, b, c = rd("v3new.jsonl"), rd("v3ident.jsonl"), rd("v3demo.jsonl")
ok = all(abs(x["kld"] - y["kld"]) < 1e-12 and abs(x["bpw"] - y["bpw"]) < 1e-12 and abs(x["t_lin_ms"] - y["t_lin_ms"]) < 1e-9 for x, y in zip(a, b))
print(f"2 IDENTITY {'OK' if ok and len(a) == len(b) else 'FAIL'}: KLD / bpw / time equal with r = 1 (oursH replacing ours)")
for x, y in zip(a, c):
    sh = sum(v for k, v in y["param_share"].items() if k.startswith("oursH-"))
    print(f"3 illustration {x['target']} lam {x['lambda']:.0e}: predicted KLD {x['kld']:.5f} -> {y['kld']:.5f} ({y['kld'] / x['kld'] - 1:+.1%}), "
          f"oursH share of trunk params {sh:.0%}")
PY
grep -h "H_out options" $O/v3ident.log $O/v3demo.log
echo KNAPTEST DONE
