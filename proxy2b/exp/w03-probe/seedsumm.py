#!/usr/bin/env python3
"""seedsumm.py NAME=f32[,f32..] ... [--ref NAME] -- per arm: gen-masked mean KL of each seed's evalq.py --out file, the
3-seed mean and per-seed SD; per arm vs --ref (base): gain = mean(ref) - mean(arm), and whether every arm seed beats
every ref seed. One JSON line per arm (missing files: the arm is skipped)."""
import json, os, sys
import numpy as np
a = sys.argv[1:]; ref = a[a.index("--ref") + 1] if "--ref" in a else None
arms = dict(x.split("=", 1) for x in a if "=" in x)
def kl(f):
    r = np.fromfile(f, np.float32).reshape(-1, 4); g = r[:, 2] > 0; return float(r[g, 0].mean())
K = {}
for n, fs in arms.items():
    fs = fs.split(",")
    if all(os.path.exists(f) for f in fs): K[n] = [kl(f) for f in fs]
for n, v in K.items():
    o = {"arm": n, "seeds": v, "mean": float(np.mean(v)), "sd_rel": float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 else None}
    if ref in K and n != ref:
        o["gain"] = float(np.mean(K[ref]) - np.mean(v)); o["ratio"] = float(np.mean(v) / np.mean(K[ref]))
        o["every_beats_ref"] = bool(max(v) < min(K[ref]))
    print(json.dumps(o))
