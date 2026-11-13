#!/bin/bash
# s2b.sh PAR TOP -- #177 stage 2b on colo: the TOP A of each s2a search family (by the max ratio over K 2-6, 64 tiles) on
# the fixed set's tiles (512, seed 12, K 1-6), one job an A; the ratio against fx-m1i8 (same tiles) comes in the report.
# Resumable as s2a.sh. (w139)
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p res logs; export OMP_NUM_THREADS=1
python3 - ${2:-6} > s2b.cands <<'PY'
import json, glob, collections, sys
d = collections.defaultdict(dict)
for f in glob.glob("res/as-*.jsonl"):
    fam = f.split("as-")[1].rsplit("-", 1)[0]
    for l in open(f):
        o = json.loads(l); d[(fam, o["spec"])][o["K"]] = o["ratio"]
for fam in sorted({k[0] for k in d}):
    rows = sorted((max(v.values()), s) for (f, s), v in d.items() if f == fam and len(v) == 5)
    for m, s in rows[:int(sys.argv[1])]: print(s)
PY
while read s; do
  o=res/fx-$(echo $s | tr ':' '_').jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 6 ] && continue
  echo "python3 hscreen.py eval $s 1,2,3,4,5,6 16 512 12 $o > logs/fx-$(echo $s | tr ':' '_').log 2>&1"
done < s2b.cands | xargs -P ${1:-20} -I{} bash -c "nice -n 19 {}"
echo S2B DONE
