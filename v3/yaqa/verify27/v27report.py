#!/usr/bin/env python3
"""v27report.py DIR -- #104 amendment 2 readout (v3/yaqa/SWJ27.md): per arm mean / p99 KLD and top-1 agreement over the
rows the golden's gen.u8 marks (the official score's rows), the pairs (paired.py's per-document bootstrap: seed 0,
2,000 resamples; plus the ratio's 95% interval from the same resamples) and the verdict.
Rows: radiance 1.16 --kld-out, [rows, 4] f32 = {kl, nll_q, nll_p, top1}. Verify arms: DIR/verify/<arm>.json.rows on
DIR/golden-verify27; search arms: DIR/search/ (head-matched controls) plus the existing swap27 / swj27 rows.
--selftest: base4 -> base4swj on golden-search27 must equal paired.py's line in q2's log (no new data)."""
import json, os, sys
import numpy as np

SW, SJ = "/data/swap27", "/data/swj27"
GS = f"{SW}/golden-search27"


def golden(g):
    m = json.load(open(f"{g}/kld.json"))
    gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    return m, gen


def load(path, m):
    s = np.fromfile(path, np.float32).reshape(-1, 4).astype(np.float64)
    assert len(s) == m["rows"], (path, len(s), m["rows"])
    assert not (s == 0).all(1).any(), f"{path}: unscored rows"
    assert np.isin(s[:, 3], (0.0, 1.0)).all(), f"{path}: column 3 is not top-1 (0/1)"
    return s


def arm_stats(s, gen):
    kl = s[gen, 0]
    return {"positions": int(kl.size), "mean": kl.mean(), "p99": float(np.quantile(kl, 0.99)), "top1": s[gen, 3].mean()}


def pair(m, gen, A, B):
    sa, sb, n = [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); mk = gen[sl]
        sa.append(A[sl, 0][mk].sum()); sb.append(B[sl, 0][mk].sum()); n.append(mk.sum())
    sa, sb, n = map(np.array, (sa, sb, n))
    ka, kb = sa.sum() / n.sum(), sb.sum() / n.sum()
    rng = np.random.default_rng(0); bs, rs = [], []
    for _ in range(2000):
        i = rng.integers(0, len(n), len(n)); bs.append((sb[i].sum() - sa[i].sum()) / n[i].sum()); rs.append(sb[i].sum() / sa[i].sum())
    se = float(np.std(bs))
    return {"kld_a": ka, "kld_b": kb, "ratio": kb / ka, "lo": float(np.quantile(rs, 0.025)), "hi": float(np.quantile(rs, 0.975)),
            "diff": kb - ka, "se": se, "z": (kb - ka) / se, "positions": int(n.sum()), "docs": len(n)}


def verdict(r0, r1):
    geo = (r0 * r1) ** 0.5
    if r0 < 1 and r1 < 1 and geo <= 0.90: v = "CONFIRMED (both < 1, geo <= 0.90)"
    elif r0 < 1 and r1 < 1 and geo < 0.98: v = "REAL, SMALLER THAN SEARCH SAID (both < 1, 0.90 < geo < 0.98)"
    else: v = "NOT CONFIRMED (a pair >= 1 or geo >= 0.98)"
    return geo, v


def main():
    if sys.argv[1] == "--selftest":
        m, gen = golden(GS)
        p = pair(m, gen, load(f"{SW}/base4.json.rows", m), load(f"{SJ}/base4swj.json.rows", m))
        print(json.dumps(p)); return
    D = sys.argv[1]
    sets = {"verify": (f"{D}/golden-verify27", lambda a: f"{D}/verify/{a}.json.rows"),
            "search": (GS, lambda a: next(f for f in (f"{D}/search/{a}.json.rows", f"{SJ}/{a}.json.rows", f"{SW}/{a}.json.rows")
                                          if os.path.exists(f)))}
    pairs = {"verify": [("base4v4", "base4swj", "gate"), ("bn4v4", "bn4swj", "gate"),
                        ("base4", "base4swj", "as amendment 1 (head differs)"), ("bn4", "bn4swj", "as amendment 1 (head differs)"),
                        ("base4", "base4v4", "head / embedding V2 3.00 -> 4.00 only"), ("bn4", "bn4v4", "head / embedding only"),
                        ("base", "baseswj", "K2.5 report"), ("s1", "s1swj", "K2.5 report")],
             "search": [("base4v4", "base4swj", "amendment 1 head-matched"), ("bn4v4", "bn4swj", "amendment 1 head-matched"),
                        ("base4", "base4v4", "head / embedding only"), ("bn4", "bn4v4", "head / embedding only")]}
    out = {}
    for name, (g, path) in sets.items():
        m, gen = golden(g); rows = {}
        for a in sorted({x for p in pairs[name] for x in p[:2]}):
            rows[a] = load(path(a), m)
        print(f"## {name}: {os.path.basename(g)}, {len(m['docs'])} docs, {int(gen.sum())} scored positions")
        print(f"{'arm':10} {'mean KLD':>10} {'p99 KLD':>9} {'top-1':>7}")
        for a, s in rows.items():
            st = arm_stats(s, gen); print(f"{a:10} {st['mean']:10.6f} {st['p99']:9.4f} {st['top1']:7.4f}")
        print(f"{'pair':22} {'ratio':>7} {'95% CI':>15} {'z':>6} {'p99 B/A':>8} {'top-1 B-A':>9}  note")
        res = {}
        for a, b, note in pairs[name]:
            p = pair(m, gen, rows[a], rows[b]); sa, sb = arm_stats(rows[a], gen), arm_stats(rows[b], gen)
            p.update(p99_ratio=sb["p99"] / sa["p99"], top1_diff=sb["top1"] - sa["top1"]); res[f"{a}->{b}"] = p
            print(f"{a + ' -> ' + b:22} {p['ratio']:7.4f} [{p['lo']:.4f}, {p['hi']:.4f}] {p['z']:6.1f} {p['p99_ratio']:8.4f} {p['top1_diff']:+9.4f}  {note}")
        out[name] = res
    r0, r1 = out["verify"]["base4v4->base4swj"]["ratio"], out["verify"]["bn4v4->bn4swj"]["ratio"]
    geo, v = verdict(r0, r1)
    print(f"VERDICT #104 amendment 2 (K4, verify, head-matched): r0 {r0:.4f}, r1 {r1:.4f}, geo {geo:.4f} -> {v}")
    json.dump(out, open(f"{D}/report.json", "w"), indent=1)


main()
