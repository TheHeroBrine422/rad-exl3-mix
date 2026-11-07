#!/usr/bin/env python3
"""v75.py A -- w75 #112-02 readout and verdict (../w75.md): C9 on the 27B writers, fixed-code pairs on the swj K2.5
levels. Rows (radiance --kld-out, [rows, 4] f32 = {kl, nll_q, nll_p, top1}): search A/<tag>.json.rows on
golden-search27, verify A/vf/<tag>.json.rows on golden-verify27; tags ctl-<seed>swj (the control, re-scored here) and
<seed>c9<v> (arm v = a / b), seed = base / s1. Per arm and slice: mean / p99 KLD, top-1 over the gen rows; per pair the
ratio, per-document bootstrap SE (2,000 resamples, seed 0, paired.py's arithmetic) and z, p99 ratio, top-1 difference.
Also: the fresh controls vs the stored scores (/data/swj27, /data/verify27/verify): max |row diff|.
Prints markdown; the verdict lines start "C9W27 VERDICT"."""
import json, math, os, sys
import numpy as np

A = sys.argv[1]
GS, GV = "/data/swap27/golden-search27", "/data/verify27/golden-verify27"
SL = {"search": (GS, lambda t: f"{A}/{t}.json.rows", lambda s: f"/data/swj27/{s}swj.json.rows"),
      "verify": (GV, lambda t: f"{A}/vf/{t}.json.rows", lambda s: f"/data/verify27/verify/{s}swj.json.rows")}
SEEDS = ("base", "s1")


def golden(g):
    m = json.load(open(f"{g}/kld.json")); return m, np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)


def rows(p, m):
    if not os.path.exists(p): return None
    s = np.fromfile(p, np.float32).reshape(-1, 4).astype(np.float64)
    assert len(s) == m["rows"], (p, len(s), m["rows"]); return s


def stats(s, gen):
    kl = s[gen, 0]; return {"mean": kl.mean(), "p99": float(np.quantile(kl, 0.99)), "top1": s[gen, 3].mean()}


def pair(m, gen, X, Y):
    sa, sb, n = [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); mk = gen[sl]
        sa.append(X[sl, 0][mk].sum()); sb.append(Y[sl, 0][mk].sum()); n.append(mk.sum())
    sa, sb, n = map(np.array, (sa, sb, n)); ka, kb = sa.sum() / n.sum(), sb.sum() / n.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(n), len(n)); bs.append((sb[i].sum() - sa[i].sum()) / n[i].sum())
    se = float(np.std(bs)); a, b = stats(X, gen), stats(Y, gen)
    return {"r": kb / ka, "z": (kb - ka) / se, "se_rel": se / ka, "p99r": b["p99"] / a["p99"], "dtop1": b["top1"] - a["top1"]}


L, R = ["# w75 #112-02 · C9 on the 27B writers (swj K2.5 levels, fixed codes)", ""], {}
for sl, (g, path, stored) in SL.items():
    m, gen = golden(g)
    L += [f"## {sl}: {os.path.basename(g)}, {len(m['docs'])} docs, {int(gen.sum())} scored positions", "",
          "| arm | mean KLD | p99 KLD | top-1 |", "|---|---|---|---|"]
    S = {}
    for s in SEEDS:
        for t in (f"ctl-{s}swj", f"{s}c9a", f"{s}c9b"):
            x = rows(path(t), m)
            if x is None: continue
            S[t] = x; st = stats(x, gen); L.append(f"| {t} | {st['mean']:.6f} | {st['p99']:.4f} | {st['top1']:.4f} |")
        old = rows(stored(s), m)
        if old is not None and f"ctl-{s}swj" in S:
            L.append(f"| (stored {s}swj vs fresh: max row diff {np.abs(old[:, 0] - S[f'ctl-{s}swj'][:, 0]).max():.2e}) | | | |")
    L += ["", "| pair | ratio | z | rel SE | p99 ratio | top-1 diff |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        for a_, b_ in ((f"ctl-{s}swj", f"{s}c9a"), (f"ctl-{s}swj", f"{s}c9b"), (f"{s}c9a", f"{s}c9b")):
            if a_ in S and b_ in S:
                p = pair(m, gen, S[a_], S[b_]); R[(sl, a_, b_)] = p
                L.append(f"| {a_} → {b_} | {p['r']:.4f} | {p['z']:+.1f} | {p['se_rel']:.4f} | {p['p99r']:.4f} | {p['dtop1']:+.4f} |")
    L.append("")


def verdict(v):
    ps = [R.get(("search", f"ctl-{s}swj", f"{s}c9{v}")) for s in SEEDS]
    pv = [R.get(("verify", f"ctl-{s}swj", f"{s}c9{v}")) for s in SEEDS]
    if None in ps: return None
    g = math.sqrt(ps[0]["r"] * ps[1]["r"])
    if g >= 1.0 or any(p["r"] >= 1 for p in ps): out = "FAIL"
    elif None in pv: out = "PENDING (verify)"
    else:
        ls = sum(math.log(p["r"]) for p in ps) / 2; lv = sum(math.log(p["r"]) for p in pv) / 2
        keep = lv / ls if ls < 0 else float("nan")
        ok = g <= 0.99 and all(p["r"] < 1 and p["z"] <= -2 for p in ps) and all(p["r"] < 1 for p in pv) and keep >= 2 / 3
        out = ("PASS -> adopt C9 on the writers as a per-level step after swj" if ok else "PARTIAL") + f" (verify keeps {keep:.2f})"
    fl = [f"{sl} {s}: p99 ratio {p['p99r']:.4f}" for sl, pp in (("search", ps), ("verify", pv)) for s, p in zip(SEEDS, pp) if p and p["p99r"] > 1.01]
    fl += [f"{sl} {s}: top-1 {p['dtop1'] * 100:+.2f} pt" for sl, pp in (("search", ps), ("verify", pv)) for s, p in zip(SEEDS, pp) if p and p["dtop1"] < -0.001]
    return f"C9W27 VERDICT arm {v}: {out}; search r {ps[0]['r']:.4f} / {ps[1]['r']:.4f} (z {ps[0]['z']:+.1f} / {ps[1]['z']:+.1f}), g {g:.4f}" \
        + ("; verify r " + " / ".join(f"{p['r']:.4f}" for p in pv) if None not in pv else "") + (f"; FLAGS: {', '.join(fl)}" if fl else "; no flags")


for v in ("a", "b"):
    x = verdict(v)
    if x: L.append(x)
bb = [R.get(("search", f"{s}c9a", f"{s}c9b")) for s in SEEDS]; bv = [R.get(("verify", f"{s}c9a", f"{s}c9b")) for s in SEEDS]
if None not in bb and None not in bv:
    gb, gv = math.sqrt(bb[0]["r"] * bb[1]["r"]), math.sqrt(bv[0]["r"] * bv[1]["r"])
    pref = gb <= 0.995 and all(p["r"] < 1 for p in bb) and gv < 1
    L.append(f"C9W27 VERDICT B vs A: search geo {gb:.4f} ({bb[0]['r']:.4f} / {bb[1]['r']:.4f}), verify geo {gv:.4f} -> {'B preferred' if pref else 'A kept'}")
print("\n".join(L))
