#!/bin/bash
# r100check.sh -- #100 check: trellis4.py with codebook mul1_i8 reproduces trellis2.py exactly (2 tensors, an integer
# and a fractional K, r100.sh's settings). Run in r100.sh's code dir (needs trellis2.py there too). Prints IDENTICAL or
# the differing rows.
cd "$(dirname "$(readlink -f "$0")")"
export E3_FACTORS=0.5,0.7,0.85,1.0,1.15,1.3,1.5 E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CD=0
mkdir -p logs; rm -f chk2.jsonl chk4.jsonl
for t in 43:attn_v 10:ssm_out; do l=${t%%:*}; w=${t#*:}
  for v in 2 4; do echo "python3 -W ignore trellis$v.py $l $w mul1_i8 2,2.5 128 chk$v.jsonl > logs/chk$v-$l-$w.log 2>&1"; done
done | xargs -P 4 -I{} bash -c "nice -n 10 {}"
python3 - <<'EOF'
import json
k = lambda o: (o["layer"], o["w"], o["method"], o["bits"])
a = {k(o): o for o in map(json.loads, open("chk2.jsonl"))}; b = {k(o): o for o in map(json.loads, open("chk4.jsonl"))}
bad = [x for x in a if x not in b or (a[x]["rel"], a[x]["scale_f"]) != (b[x]["rel"], b[x]["scale_f"])]
for x in sorted(a): print(x, a[x]["rel"], b.get(x, {}).get("rel"), a[x]["scale_f"], a[x]["sec"], b.get(x, {}).get("sec"))
print("IDENTICAL" if len(a) == len(b) == 4 and not bad else f"DIFFER {bad}")
EOF
