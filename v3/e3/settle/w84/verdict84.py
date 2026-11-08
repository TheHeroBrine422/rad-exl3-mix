#!/usr/bin/env python3
"""verdict84.py A ARM [ARM ...] -- ../w84.md's rule over the seed pairs w84.sh wrote (A/<ARM>/pair.json: tools/paired.py,
control -> permuted-order arm on golden-search27): r = kld_b / kld_a per pair, g = exp(mean log r).
ADOPT if every r < 1 (two pairs); NOT ADOPTED if every r >= 1; else AMBIGUOUS (-> K4 pair on the lead's go).
Reported: z per pair, pooled z (sum of diffs / sqrt(sum se^2)), p99 of per-position KL and top-1 agreement on the
scored positions (gen.u8), arm vs control, from the rows files (col 0 KL, col 3 top-1 match)."""
import json, math, sys
import numpy as np
A, arms = sys.argv[1], sys.argv[2:]
G = "/data/swap27/golden-search27"
gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
rows = lambda f: np.fromfile(f, np.float32).reshape(-1, 4)[gen]
rs, ds, ses = [], [], []
for arm in arms:
    d = json.load(open(f"{A}/{arm}/pair.json")); m = json.load(open(f"{A}/{arm}/pair-meta.json"))
    r = d["kld_b"] / d["kld_a"]; rs.append(r); ds.append(d["diff"]); ses.append(d["se"])
    c, p = rows(m["control"]), rows(m["perm"])
    print(f"{arm}: control {m['control'].split('/')[-1]} {d['kld_a']:.6f} -> perm {d['kld_b']:.6f}: r {r:.4f} (z {d['z']:+.2f}, "
          f"se {d['se'] / d['kld_a']:.4f} rel, {d['docs']} docs, {d['positions']} positions); p99 KL {np.percentile(c[:, 0], 99):.4f} -> "
          f"{np.percentile(p[:, 0], 99):.4f}; top-1 {c[:, 3].mean():.5f} -> {p[:, 3].mean():.5f}; noop {m['noop']}, hygiene {m['hygiene']}")
g = math.exp(sum(map(math.log, rs)) / len(rs))
zp = sum(ds) / math.sqrt(sum(s * s for s in ses))
v = "ADOPT" if len(rs) >= 2 and all(r < 1 for r in rs) else "NOT ADOPTED" if all(r >= 1 for r in rs) else "AMBIGUOUS (K4 pair on the lead's go)"
print(f"W84 VERDICT {v}: r = {', '.join(f'{r:.4f}' for r in rs)}; geo {g:.4f}; pooled z {zp:+.2f} (rule: adopt if both r < 1)")
