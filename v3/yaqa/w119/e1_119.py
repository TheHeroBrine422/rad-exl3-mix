#!/usr/bin/env python3
"""e1_119.py HW GX GC OUT.json -- #154 E1 (RULES.md): per-document re-check of the head's calib / xcorp split on headwm90's
rows (HW/<arm>-{x,c}.json.rows, radiance --kld-out rows [n, 4] f32: KL, ?, ?, top-1), goldens GX / GC (kld.json, gen.u8).
Arithmetic of pair84.doc_sums: per document the sum of KL over the scored (gen) rows."""
import json, sys
import numpy as np

HW, GX, GC, OUT = sys.argv[1:5]
PAIRS = [("v4", "h4-bf16"), ("h4-u5", "h4-s5-3"), ("h4-u6", "h4-s6-3"), ("v8", "h8-bf16")]
B = 2000


def load(g, arm, p):
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    R = np.fromfile(f"{HW}/{arm}-{p}.json.rows", np.float32).reshape(-1, 4).astype(np.float64)
    assert len(R) == m["rows"] == len(gen)
    s, n, pos, src = [], [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); g_ = gen[sl]
        s.append(R[sl, 0][g_].sum()); n.append(g_.sum()); pos.append(R[sl, 0][g_]); src.append(d["source"])
    return np.array(s), np.array(n, float), pos, src


def part(a, b, p):
    g = GX if p == "x" else GC
    sa, n, pa, src = load(g, a, p); sb, _, pb, _ = load(g, b, p)
    dd = sb - sa; N = n.sum(); D = dd.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        i = rng.integers(0, len(n), len(n)); bs.append(dd[i].sum() / n[i].sum())
    m = n > 0
    loo = np.array([(D - dd[j]) / (N - n[j]) for j in range(len(n))])
    dpos = np.concatenate([y - x for x, y in zip(pa, pb)]); ad = np.abs(dpos)
    top = np.sort(ad)[::-1][:max(1, len(ad) // 100)].sum() / ad.sum()
    share_net = np.abs(dd) / abs(D); share_abs = np.abs(dd) / np.abs(dd).sum()
    srcs = {}
    for j, s_ in enumerate(src):
        x = srcs.setdefault(s_, [0.0, 0.0, 0]); x[0] += dd[j]; x[1] += n[j]; x[2] += 1
    rat = sb[m] / sa[m]
    return {"kl_a": sa.sum() / N, "kl_b": sb.sum() / N, "dkl_pos": D / N, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "bs": bs, "docs": int(len(n)), "pos": int(N),
            "doc_dkl_pos_median": float(np.median(dd[m] / n[m])), "doc_ratio_geo": float(np.exp(np.mean(np.log(rat)))),
            "loo_min": float(loo.min()), "loo_max": float(loo.max()), "top1pct_share_absdkl": float(top),
            "max_doc_share_net": float(share_net.max()), "max_doc_share_abs": float(share_abs.max()),
            "docs_over20_net": [(int(j), src[j], float(share_net[j])) for j in np.argsort(-share_net) if share_net[j] > 0.2],
            "dd": dd, "n": n, "D": D, "N": N,
            "by_source": {k: {"dkl": v[0], "pos": v[1], "docs": v[2], "dkl_pos": v[0] / v[1] if v[1] else None,
                              "share_net": v[0] / D} for k, v in sorted(srcs.items(), key=lambda t: -abs(t[1][0]))}}


out = {}; votes = {"a": 0, "b": 0}
for a, b in PAIRS:
    P = {p: part(a, b, p) for p in "xc"}
    x, c = P["x"], P["c"]
    r = c["dkl_pos"] / x["dkl_pos"]
    rb = np.array(c["bs"]) / np.array(x["bs"])     # independent resamples of the two parts
    same = np.sign(c["dkl_pos"]) == np.sign(x["dkl_pos"])
    # (c): a document with > 30% of a part's net |dKL| -> ratio without it
    r_c = None; drop = {}
    for p in "xc":
        q = P[p]; j = int(np.argmax(np.abs(q["dd"]) / abs(q["D"])))
        if abs(q["dd"][j]) / abs(q["D"]) > 0.30: drop[p] = j
    if drop:
        def wo(p):
            q = P[p]; j = drop.get(p)
            return q["dkl_pos"] if j is None else (q["D"] - q["dd"][j]) / (q["N"] - q["n"][j])
        r_c = wo("c") / wo("x")
    rr = r_c if r_c is not None else r
    if 0.5 <= rr <= 2: votes["a"] += 1
    if rr > 2: votes["b"] += 1
    key = f"{a} -> {b}"
    out[key] = {"ratio_c_over_x": r, "ratio_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                "same_sign": bool(same), "dropped_docs": {p: int(j) for p, j in drop.items()}, "ratio_after_c": r_c,
                "doc_median_ratio_c_over_x": c["doc_dkl_pos_median"] / x["doc_dkl_pos_median"],
                **{p: {k: v for k, v in P[p].items() if k not in ("bs", "dd", "n", "D", "N")} for p in "xc"}}
    print(f"{key}: dKL/pos x {x['dkl_pos']:+.3e} [{x['ci95'][0]:+.2e}, {x['ci95'][1]:+.2e}]  c {c['dkl_pos']:+.3e} "
          f"[{c['ci95'][0]:+.2e}, {c['ci95'][1]:+.2e}]  c/x {r:.2f} [{out[key]['ratio_ci95'][0]:.2f}, {out[key]['ratio_ci95'][1]:.2f}]"
          f"{'' if r_c is None else f' after (c) {r_c:.2f}'}  doc-median c/x {out[key]['doc_median_ratio_c_over_x']:.2f}")
    for p in "xc":
        q = P[p]
        print(f"   {p}: kl {q['kl_a']:.5f} -> {q['kl_b']:.5f}; doc median dKL/pos {q['doc_dkl_pos_median']:+.2e}, doc geo ratio "
              f"{q['doc_ratio_geo']:.4f}; LOO [{q['loo_min']:+.2e}, {q['loo_max']:+.2e}]; top-1% pos share {q['top1pct_share_absdkl']:.2f}; "
              f"max doc share net {q['max_doc_share_net']:.2f} abs {q['max_doc_share_abs']:.2f}; >20%: {q['docs_over20_net']}")
        print("      sources: " + "; ".join(f"{k} {v['docs']}d {v['share_net']:+.2f}" for k, v in list(q["by_source"].items())[:6]))
verdict = ("(a) corpus-general: read head results on absolute dKL per position" if votes["a"] >= 3 else
           "(b) calib-composition pricing" if votes["b"] >= 3 else "inconclusive: report both ways")
out["votes"] = votes; out["verdict"] = verdict
print("E1 VERDICT:", verdict, votes)
json.dump(out, open(OUT, "w"), indent=1, default=float)
