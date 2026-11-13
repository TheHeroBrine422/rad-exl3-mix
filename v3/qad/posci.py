#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""posci.py GOLDEN A_ROWS B_ROWS [--json OUT] -- the #165 long-context check: KL by context position bin, B (healed)
vs A (unhealed), paired on the same gen rows. Bins 0-2k, 2-4k, 4-8k, 8k+ (position of the predicting token). Per
bin: pooled KL of each, ratio B / A and its 95% interval (document bootstrap, 2,000 resamples, seed 0; on a chunked
trace slice a document = a chunk), documents better / worse. Rows: radiance's .json.rows (f32 [rows, 4], KL in column
0) or a qad.py tscore-*.npz (per-row KL with positions). A bin with a ratio CI above 1 at 4k+ is flagged."""
import argparse, json

import numpy as np

EDGES = [0, 2048, 4096, 8192, 1 << 40]
NAMES = ["0-2k", "2-4k", "4-8k", "8k+"]


def load(golden, f, m):
    """per golden row: KL (NaN where the file has no row)"""
    R = m["rows"]
    if f.endswith(".npz"):
        z = np.load(f); out = np.full(R, np.nan)
        for d_i, d in enumerate(m["docs"]):
            sel = z["doc"] == d_i
            out[d["row0"] + z["pos"][sel] - d["score_from"]] = z["kl"][sel]
        return out
    return np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("golden"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--json", default=None); ap.add_argument("--label", default="")
    a = ap.parse_args()
    m = json.load(open(f"{a.golden}/kld.json")); gen = np.fromfile(f"{a.golden}/gen.u8", np.uint8).astype(bool)
    pos = np.zeros(m["rows"], np.int64); doc = np.zeros(m["rows"], np.int64)
    for j, d in enumerate(m["docs"]):
        n = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + n)
        pos[s] = d["score_from"] + np.arange(n); doc[s] = j
    A, B = load(a.golden, a.a, m), load(a.golden, a.b, m)
    ok = gen & np.isfinite(A) & np.isfinite(B)
    b = np.digitize(pos, EDGES) - 1; nd = len(m["docs"])
    rep = {"golden": a.golden, "a": a.a, "b": a.b, "label": a.label, "bins": {}, "flag": False}
    for k, name in enumerate(NAMES):
        sel = ok & (b == k)
        if not sel.any(): rep["bins"][name] = None; continue
        sa = np.bincount(doc[sel], A[sel], nd); sb = np.bincount(doc[sel], B[sel], nd); n = np.bincount(doc[sel], None, nd)
        have = np.nonzero(n > 0)[0]
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = have[rng.integers(0, len(have), len(have))]; bs.append(sb[i].sum() / sa[i].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r = {"rows": int(n.sum()), "docs": int(len(have)), "kl_a": float(sa.sum() / n.sum()), "kl_b": float(sb.sum() / n.sum()),
             "ratio": float(sb.sum() / sa.sum()), "ci95": [float(lo), float(hi)],
             "docs_better": int((sb[have] < sa[have]).sum()), "docs_worse": int((sb[have] > sa[have]).sum())}
        if "chunk" in m["docs"][0]:                      # chunked traces: the ratio per source trace as well
            tr = np.array([d["orig"] for d in m["docs"]])
            r["per_trace"] = {str(t): float(sb[tr == t].sum() / sa[tr == t].sum()) for t in sorted(set(tr[have].tolist()))}
        r["flag"] = bool(k >= 2 and lo > 1.0)
        rep["flag"] |= r["flag"]; rep["bins"][name] = r
        print(f"{a.label:12s} {name:>5s} rows {r['rows']:7d} docs {r['docs']:4d}  A {r['kl_a']:.5f}  B {r['kl_b']:.5f}  "
              f"ratio {r['ratio']:.4f} [{lo:.4f}, {hi:.4f}]  better {r['docs_better']} worse {r['docs_worse']}"
              + ("  FLAG" if r["flag"] else "") + (f"  per trace {r['per_trace']}" if "per_trace" in r else ""))
    if a.json: json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
