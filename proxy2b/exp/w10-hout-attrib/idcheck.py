#!/usr/bin/env python3
"""idcheck.py BASE_WQ ARM_WQ SUBSET.txt -- a subset arm (mksub.py) must equal its seed-matched base arm outside the
subset: every dequantized weight (ourq.py --save-wq) of a linear NOT in SUBSET.txt bit-identical, and (sanity) the
subset's own linears changed. Prints one JSON line; exit 0 = identical outside the subset, 1 = not (then the arm's
noise is the whole model's, and run.sh falls back to 3 seeds for every subset arm)."""
import glob, json, os, sys
import torch
from safetensors.torch import load_file

bw, aw, sub = sys.argv[1:4]
S = set(l.strip() for l in open(sub) if l.strip())
same, diff_out, diff_in, unchanged_in, worst = 0, [], 0, [], 0.0
fs = sorted(glob.glob(f"{bw}/*.safetensors"))
assert len(fs) == 150, f"{bw}: {len(fs)} weights"
for f in fs:
    key = os.path.basename(f)[:-len(".safetensors")]
    x = load_file(f)["weight"]; y = load_file(f"{aw}/{key}.safetensors")["weight"]
    eq = torch.equal(x, y)
    if key in S:
        if eq: unchanged_in.append(key)
        else: diff_in += 1
    elif eq: same += 1
    else:
        diff_out.append(key); worst = max(worst, float((x.float() - y.float()).norm() / x.float().norm()))
ok = not diff_out
print(json.dumps({"arm": aw, "outside_identical": same, "outside_changed": len(diff_out), "outside_worst_rel": worst,
                  "first_changed": diff_out[:3], "subset_changed": diff_in, "subset_unchanged": unchanged_in[:3],
                  "pass": ok}))
sys.exit(0 if ok else 1)
