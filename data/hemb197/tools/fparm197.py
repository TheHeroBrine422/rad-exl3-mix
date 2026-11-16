#!/usr/bin/env python3
"""fparm197.py ROWS_DIR GOLDEN_DIR OUT.json -- #197 lead note (fp8 arm f): the f arms' mean KL over gen rows, and the paired
per-document bootstrap of each f arm against its base's arm (a) (the same statistic as table197.py). Writes OUT.json with
the per-document sums (so the interval can be re-derived). numpy."""
import json, sys
import numpy as np
RD, GD, OUT = sys.argv[1:4]
kld = json.load(open(f"{GD}/kld.json")); n = kld["rows"]
gen = np.fromfile(f"{GD}/gen.u8", np.uint8).astype(bool)
def col0(nm):
    a = np.fromfile(f"{RD}/{nm}.json.rows", np.float32)
    return a.reshape(n, a.size // n)[:, 0].astype(np.float64)
out = {}
for base in ("5.5", "4.0", "2.5"):
    A = col0(base + "-a"); F = col0(base + "-f")
    sa, sb, ns = [], [], []
    for d in kld["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
        sa.append(A[s][mk].sum()); sb.append(F[s][mk].sum()); ns.append(mk.sum())
    sa, sb, ns = map(np.array, (sa, sb, ns)); ka, kb = sa.sum() / ns.sum(), sb.sum() / ns.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ns), len(ns)); bs.append((sb[i].sum() - sa[i].sum()) / ns[i].sum())
    out[base + "-f"] = {"kl_mean_gen": float(kb), "vs": base + "-a", "kl_a": float(ka),
                        "rel_pct": float(100 * (kb - ka) / ka), "ci95_pct": float(100 * 1.96 * np.std(bs) / ka),
                        "perdoc_sum_n": [[float(x), int(y)] for x, y in zip(sb, ns)],
                        "perdoc_sum_n_a": [[float(x), int(y)] for x, y in zip(sa, ns)]}
json.dump({"note": "fp8 E4M3 values (one f32 scale, rtn codes=fp8_e4m3 block=*x*) decoded to bf16 and served as a plain bf16 table on the card: the values embed_lookup_q would produce (bf16(code*scale)); VRAM and tok/s not measured for fp8 (the engine refuses an E4M3 token table in this arch)", "arms": out}, open(OUT, "w"), indent=1)
for k, v in out.items():
    print(f"{k}: {v['kl_mean_gen']:.6f}  vs {v['vs']} ({v['kl_a']:.6f}): {v['rel_pct']:+.2f} % ± {v['ci95_pct']:.2f}")
