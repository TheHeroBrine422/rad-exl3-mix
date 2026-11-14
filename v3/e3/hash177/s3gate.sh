#!/bin/bash
# s3gate.sh -- RULES-177 gate: two i8 jobs of box2's ladder (k4.125 L00-g0, k2.625 L05-g1) re-run with the patched tree
# (env-w139.sh) on a free card; their t-*.safetensors tensors must equal the ladder's out/ byte for byte (metadata aside).
# Out /data/q27/w139/s3/gate. (w139, #177)
Q=/data/q27; H=$Q/w139; O=$H/s3/gate; mkdir -p $O; source $H/tools/env-w139.sh
exec >> $O/gate.log 2>&1
say() { echo "$(date -u +%T) $*"; }
while :; do for c in ${CARDS:-3 2 1 0}; do exec 5>$Q/locks/card$c.lock; if flock -n 5; then CARD=$c; break 2; fi; exec 5>&-; done; sleep 20; done
exec 4>$Q/locks/ram.lock; flock -s 4; say "card $CARD"
for p in k4.125:L00-g0 k2.625:L05-g1; do
  lv=q27v3-144dd6e1-${p%%:*}-i8; n=${p#*:}; s=$O/$lv-$n.json; o=$O/$lv-$n
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['config']=sys.argv[2]; json.dump(d,open(sys.argv[3],'w'),indent=1)" \
    $Q/ladder/$lv/specs/$n.json $H/code/pipeline/config/v3.toml $s
  [ -f $o/job.json ] || ROCR_VISIBLE_DEVICES=$CARD nice -n 5 $PY -m rxq job $s -o $o --device 0 >> $o.log 2>&1 || say "JOBFAIL $lv $n"
  $PY - $Q/ladder/$lv/out/$n $o <<'PY'
import glob, os, sys
from safetensors import safe_open
a, b = sys.argv[1:3]; bad = n = 0
for f in sorted(glob.glob(f"{a}/t-*.safetensors")):
    g = os.path.join(b, os.path.basename(f))
    with safe_open(f, "pt") as x, safe_open(g, "pt") as y:
        ka, kb = sorted(x.keys()), sorted(y.keys())
        if ka != kb: print("KEYS DIFFER", f); bad += 1; continue
        for k in ka:
            n += 1
            if not x.get_tensor(k).contiguous().view(-1).view(dtype=__import__("torch").uint8).equal(y.get_tensor(k).contiguous().view(-1).view(dtype=__import__("torch").uint8)): print("DIFF", k); bad += 1
        print(os.path.basename(f), "code", x.metadata().get("rxq_code"), "->", y.metadata().get("rxq_code"))
print(f"GATE {a.split('/')[-3]} {os.path.basename(a)}: {n} tensors, {bad} differ ->", "SAME" if bad == 0 and n else "FAIL")
PY
done
say "gate done (card $CARD released)"
