#!/usr/bin/env python3
"""smoke95.py A [G92] -- w95.sh's report (#126 step 3, score only): per i8 level (k1, k0125) and part of w92's
golden-search27c (x = xcorp, c = calib), KLD / top-1 / p99 and whether every scored row's KL is finite (w84's doc_sums,
as verdict92 --k1). Writes A/smoke95.json; last line "W95 SMOKE OK|FAIL ...". """
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w84"))
from pair84 import doc_sums  # noqa: E402

A = sys.argv[1]; G92 = sys.argv[2] if len(sys.argv) > 2 else "/data/w92"
res, ok, line = {}, True, []
for tag in ("k1", "k0125"):
    out = {}
    for part in "xc":
        r = f"{A}/scores/{tag}-{part}.json.rows"
        if not os.path.exists(r): ok = False; print(f"{tag} {part}: no score"); continue
        s, n, t, R = doc_sums(f"{G92}/golden-search27c-{part}", r)
        o = {"kld": float(s.sum() / n.sum()), "top1": float(t.sum() / n.sum()), "p99": float(np.percentile(R[:, 0], 99)),
             "finite": bool(np.isfinite(R[:, 0]).all()), "positions": int(n.sum()), "docs": len(s), "sum": float(s.sum())}
        out[part] = o; ok &= o["finite"]
        print(f"{tag} {part}: KLD {o['kld']:.5f} top-1 {o['top1']:.4f} p99 {o['p99']:.3f} finite {o['finite']} "
              f"({o['docs']} docs, {o['positions']} pos)")
    if len(out) == 2:
        P = sum(o["sum"] for o in out.values()) / sum(o["positions"] for o in out.values())
        res[tag] = {"parts": out, "pooled_kld": P}; line.append(f"{tag} pooled {P:.5f} (x {out['x']['kld']:.5f}, c {out['c']['kld']:.5f})")
json.dump({"levels": res, "finite": ok}, open(f"{A}/smoke95.json", "w"), indent=1)
print(f"W95 SMOKE {'OK' if ok and len(res) == 2 else 'FAIL'}: " + "; ".join(line))
