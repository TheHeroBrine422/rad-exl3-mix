#!/usr/bin/env python3
"""selftest_f1fam.py ROWERR2-edy.jsonl TMPDIR -- planted family KL changes (gdn_in -10%, mlp_in -5%, attn_in +6%,
writers 0%, two seeds with 0.3% noise) as evalq-style .f32 rows; f1fam.py must call e2 right for gdn_in / mlp_in,
e1 right for attn_in (planted against e2) and FAIL overall."""
import os, subprocess, sys
import numpy as np
RW, D = sys.argv[1:3]; os.makedirs(D, exist_ok=True); rng = np.random.default_rng(1)
def put(tag, v):
    x = np.zeros((1000, 4), np.float32); x[:, 0] = v; x[:, 2] = 1; x.tofile(f"{D}/{tag}.f32")
plant = {"gdn_in": -0.10, "mlp_in": -0.05, "attn_in": +0.06, "writers": 0.0}
for K, b in ((3, 0.0986), (2, 0.533)):
    for s, sf in enumerate(("", "_s1")):
        put(f"k{K}base{sf}", b)
        for f, d in plant.items(): put(f"k{K}h-{f}{sf}", b * np.exp(d + rng.normal(0, 0.003)))
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1fam.py"), RW, D],
                     capture_output=True, text=True).stdout
print(out)
want = ["gdn_in", "-> e2"], ["mlp_in", "-> e2"], ["attn_in", "-> e1"]
ok = all(any(f in l and w in l for l in out.splitlines()) for f, w in want) and "F1 per-family: FAIL" in out
print("SELFTEST f1fam", "PASS" if ok else "FAIL")
