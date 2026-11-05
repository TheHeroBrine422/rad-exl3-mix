#!/bin/bash
# price27.sh ONE H K1[,K2..] [SWAPS] -- 27B oursH pricing (ranked F1), CPU only (cpu1 / VM; no GPU, no locks). Needs the
# merged ourq-hprice.patch and ladder levels quantized WITH --hprice <the 27B diagonal> (add it to ladder2.sh's X for
# the Phase-B requant, both ladders; and to both arms of the 27B swap test):
#   ONE/K<k>/proxy.jsonl   one-sided ladder level k (proxy_err2 = its h-weighted error)
#   H/K<k>/proxy.jsonl     the H_out ladder level k, same settings + --hout
#   SWAPS                  optional lines "K KL_one KL_H" (uniform-trunk swap evals, same head / embedding / golden)
# Out ./price27/: hratio.jsonl (knap4 KN_HRATIO), hf.jsonl (KN_HF per swap), knap4 solves with and without oursH.
X=$(cd "$(dirname "$0")" && pwd); R=$(cd $X/../../.. && pwd); ONE=$1 H=$2 KS=$3 SW=$4; O=${OUT:-./price27}; mkdir -p $O
: > $O/hratio.jsonl
for k in ${KS//,/ }; do python3 $X/hratio.py $k $ONE/K$k/proxy.jsonl $H/K$k/proxy.jsonl >> $O/hratio.jsonl; done
echo "$(wc -l < $O/hratio.jsonl) ratios; one-sided vs two-sided by level:"
python3 - $O/hratio.jsonl <<'PY'
import json, math, sys, collections
a = collections.defaultdict(list)
for l in open(sys.argv[1]): d = json.loads(l); a[d["K"]].append((math.log(d["e1_ratio"]), math.log(d["r"])))
for K, v in sorted(a.items()):
    print(f"  K{K:g}: {len(v)} linears, one-sided x{math.exp(sum(x for x, _ in v) / len(v)):.3f}, two-sided x{math.exp(sum(y for _, y in v) / len(v)):.3f}, "
          f"one-sided > 1.05 on {sum(x > math.log(1.05) for x, _ in v) / len(v):.0%}")
PY
MJ=$R/data/v3/kfit2-model-final.json; HF=1.0
if [ -n "$SW" ]; then
  : > $O/hf.jsonl; while read -r k a b; do [ -n "$k" ] && python3 $X/hfit.py $MJ $O/hratio.jsonl $k $a $b >> $O/hf.jsonl; done < $SW
  HF=$(python3 -c "import json,math; v=[json.loads(l)['KN_HF'] for l in open('$O/hf.jsonl')]; print(math.exp(sum(map(math.log, v)) / len(v)))")
  echo "KN_HF from $(wc -l < $O/hf.jsonl) swaps: $HF"; cat $O/hf.jsonl
fi
cd $R/v3/nf; A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1 KN_V3=1 KN_LEVELS=$ONE KN_DAMP=1
T=${TARGETS:-2.0,2.5,3.0,3.5,4.0}
python3 knap4.py $MJ $A $F $P $O/k4-one.jsonl $T 0,3e-6 > $O/k4-one.log 2>&1 &
KN_HRATIO=$O/hratio.jsonl KN_HF=$HF python3 knap4.py $MJ $A $F $P $O/k4-h.jsonl $T 0,3e-6 > $O/k4-h.log 2>&1 &
wait
python3 - $O <<'PY'
import json, sys
O = sys.argv[1]; a = [json.loads(l) for l in open(f"{O}/k4-one.jsonl")]; b = [json.loads(l) for l in open(f"{O}/k4-h.jsonl")]
for x, y in zip(a, b):
    sh = sum(v for k, v in y["param_share"].items() if k.startswith("oursH-"))
    print(f"{x['target']} lam {x['lambda']:.0e}: predicted KLD {x['kld']:.5f} -> {y['kld']:.5f} ({y['kld'] / x['kld'] - 1:+.1%}), oursH {sh:.0%} of trunk params")
PY
