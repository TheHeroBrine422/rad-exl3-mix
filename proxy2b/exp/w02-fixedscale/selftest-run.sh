#!/bin/bash
# selftest-run.sh -- dry run of run.sh's control flow on stubs (CPU, temp dir): fake goldens, fake source quants, stub
# evalq.py (random scores), stub refitq.py / swiglufit.py (empty arm dirs); the real pcmp.py and pstat.py. Checks every
# stage finishes, a second run does nothing (idempotent), and prints the verdict tables. Output: selftest-run.txt.
H=$(cd $(dirname $0) && pwd) T=$(mktemp -d); trap "rm -rf $T" EXIT
mkdir -p $T/P $T/X $T/Y $T/gs $T/gf
cp $H/pstat.py $T/X/; cp /workspace/rad-exl3-mix/proxy2b/pcmp.py $T/P/
python3 - $T <<'PY'
import json, sys, numpy as np
T = sys.argv[1]
for g, nd in (("gs", 40), ("gf", 20)):
    docs, r = [], 0
    for i in range(nd): docs.append({"tokens": 101, "score_from": 0, "row0": r}); r += 100
    json.dump({"rows": r, "docs": docs}, open(f"{T}/{g}/kld.json", "w"))
PY
cat > $T/P/evalq.py <<'PY'
import sys, json, zlib, numpy as np
a = sys.argv; g = a[2]; wq = a[a.index("--wq") + 1]; out = a[a.index("--out") + 1]
R = json.load(open(f"{g}/kld.json"))["rows"]; rng = np.random.default_rng(zlib.crc32(wq.encode()))
base = np.random.default_rng(1).gamma(2, 0.05, R)
kl = base * (1 + 0.01 * rng.standard_normal(R)) * (0.99 if "_h6" in wq or "astar" in wq else 1.0)
np.stack([kl, np.ones(R), np.ones(R), np.zeros(R)], 1).astype(np.float32).tofile(out)
print(json.dumps({"kld": float(kl.mean()), "top1": 0.9}))
PY
cat > $T/X/refitq.py <<'PY'
import sys, os
d = sys.argv[2]; os.makedirs(f"{d}/wq"); open(f"{d}/.done", "w").close(); print("stub", sys.argv[1:], file=sys.stderr)
PY
cat > $T/X/swiglufit.py <<'PY'
import sys, os
for d in sys.argv[4:6]: os.makedirs(f"{d}/wq"); open(f"{d}/.done", "w").close()
PY
for t in k3base k2base k3bdiag k2bdiag k3base_s1 k2base_s1 k3bdiag_s1 k2bdiag_s1 k3base_s2 k2base_s2 k3bdiag_s2 k2bdiag_s2; do
  mkdir -p $T/Y/q-$t/wq; for i in $(seq 150); do : > $T/Y/q-$t/wq/$i.safetensors; done
  python3 $T/P/evalq.py x $T/gs --wq $T/Y/q-$t/wq0 --out $T/Y/$t.f32 > /dev/null
done
export X=$T/X P=$T/P Y=$T/Y O=$T/O GS=$T/gs GF=$T/gf
bash $H/run.sh all > $T/run1.txt 2>&1; echo "first run rc $?"; grep -c . $T/O/result.txt
n1=$(ls $T/O/*.f32 | wc -l); bash $H/run.sh all > $T/run2.txt 2>&1; n2=$(ls $T/O/*.f32 | wc -l)
echo "score files: $n1 after run 1, $n2 after run 2; run 2 log lines (verdicts + DONE only): $(grep -vc 'pair\|^ \|^K' $T/run2.txt)"
grep -E "FAIL|alpha\*|PASS|fail|screen|DONE" $T/O/result.txt | sort | uniq -c | head -40
B1_FULL=1 bash $H/run.sh b1 > /dev/null 2>&1; echo "B1_FULL: $(ls $T/O/*_h6.f32 $T/O/*_u6.f32 | wc -l) b1 score files (want 18)"
