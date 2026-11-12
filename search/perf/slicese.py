#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""slicese.py ROWS.npz OUTDIR [--max-dlog 0.12] -- paired-SE study of mix-vs-mix KL ratios on stored 27B score rows (w123, #161).

Input: rowpack.py's npz (scored positions per document, per report). A "part" is one golden (e.g. golden-search27d-x);
a pair is two reports of different models (container paths) on the same part. For every pair with |log r| <= --max-dlog:

  pairs.jsonl   per pair x part: r = sum B / sum A (pooled over scored positions, the board's statistic), its document
                bootstrap SE (log scale, 2000 draws) and the delta-method SE sd(u)/sqrt(n), u_d = B_d/mean(B) - A_d/mean(A);
                the same for every per-document cap c (each document cut to its first c scored positions) with the
                prefill tokens that cap costs; r_cap20 (no document above 20% of A's KL, readout.py's water-filling);
                doc-geo (mean per-document log ratio); the largest document share; sd(u) per 1,000 documents.
  scale.json    SE vs number of documents (subsamples without replacement, 400 draws per n) against sd(u)/sqrt(n).
  cross.jsonl   the same pair scored on two different slices: (log r1 - log r2) / sqrt(SE1^2 + SE2^2) -- if the
                document bootstrap SE is right this z is ~N(0, 1).
Everything is on log r (ratios near 1: log r ~ r - 1)."""
import argparse, collections, itertools, json, os

import numpy as np

CAPS = [256, 512, 1024, 2048, 4096, 8192, 0]       # 0 = whole document


def load(npz):
    Z = np.load(npz)
    G = collections.defaultdict(lambda: {"kl": {}, "t1": {}})
    for k in Z.files:
        g, rest = k.split("/", 1)
        if rest.startswith("kl/"): G[g]["kl"][rest[3:]] = Z[k]
        elif rest.startswith("t1/"): G[g]["t1"][rest[3:]] = Z[k]
        else: G[g][rest] = Z[k]
    for g, v in G.items():
        v["off"] = np.concatenate([[0], np.cumsum(v["doc_n"])])
    return dict(G)


def doc_sums(v, kl, cap=0):
    """per-document KL sums and prefill tokens with each document cut at its first `cap` scored positions"""
    off, n = v["off"], v["doc_n"]
    if cap == 0:
        s = np.add.reduceat(kl.astype(np.float64), off[:-1]) if len(kl) else np.zeros(0)
        s[n == 0] = 0
        return s, v["doc_tok"].astype(np.float64), n.astype(np.float64)
    idx = np.arange(len(kl)) - np.repeat(off[:-1], n)          # position index inside its document
    keep = idx < cap
    s = np.bincount(np.repeat(np.arange(len(n)), n)[keep], kl[keep].astype(np.float64), minlength=len(n))
    # tokens prefilled: up to the row of the last kept scored position (+ score_from offset ~ doc_tok - doc_rows)
    last = np.minimum(n, cap) - 1
    lastrow = np.where(n > 0, v["gpos"][np.clip(off[:-1] + last, 0, len(v["gpos"]) - 1)], 0)
    tok = np.minimum(v["doc_tok"], lastrow + 2 + (v["doc_tok"] - 1 - v["doc_rows"]))
    return s, tok.astype(np.float64), np.minimum(n, cap).astype(np.float64)


def capped_weights(a, cap=0.2):
    w = np.ones(len(a)); tot = a.sum()
    if tot <= 0 or len(a) * cap < 1: return w
    for _ in range(len(a)):
        sh = w * a / (w * a).sum(); over = sh > cap + 1e-12
        if not over.any(): break
        rest = (w * a)[~over].sum(); T = rest / (1 - over.sum() * cap); w[over] = cap * T / a[over]
    return w


def se_delta(a, b):
    n = len(a); u = b / b.mean() - a / a.mean()
    return float(u.std(ddof=1) / np.sqrt(n)), float(u.std(ddof=1))


def se_boot(a, b, w=None, nb=2000, seed=0):
    w = np.ones(len(a)) if w is None else w
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (nb, len(a)))
    v = np.log((w[idx] * b[idx]).sum(1) / (w[idx] * a[idx]).sum(1))
    return float(v.std())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("out"); ap.add_argument("--max-dlog", type=float, default=0.12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    G = load(a.npz); IX = json.load(open(a.npz[:-4] + ".index.json"))
    model = {(g, t): r["model"] for g, v in IX.items() for t, r in v["reports"].items()}
    rows, scale, logr = [], [], {}
    for g, v in sorted(G.items()):
        tags = sorted(v["kl"])
        for ta, tb in itertools.combinations(tags, 2):
            if model[(g, ta)] == model[(g, tb)]: continue                # the same container scored twice
            A, tokA, _ = doc_sums(v, v["kl"][ta]); B, _, _ = doc_sums(v, v["kl"][tb])
            ok = v["doc_n"] > 0; A, B = A[ok], B[ok]
            lr = float(np.log(B.sum() / A.sum()))
            if abs(lr) > a.max_dlog: continue
            se_d, sdu = se_delta(A, B)
            w = capped_weights(A); lr20 = float(np.log((w * B).sum() / (w * A).sum()))
            pa = A > 0
            rec = {"part": g, "a": ta, "b": tb, "model_a": model[(g, ta)], "model_b": model[(g, tb)], "docs": int(ok.sum()), "positions": int(v["doc_n"].sum()),
                   "tokens": int(v["doc_tok"].sum()), "kl_a": float(A.sum() / v["doc_n"].sum()),
                   "logr": lr, "se_boot": se_boot(A, B), "se_delta": se_d, "sd_u": sdu,
                   "logr_cap20": lr20, "se_cap20": se_boot(A, B, w), "max_share": float((A / A.sum()).max()),
                   "doc_geo": float(np.mean(np.log(B[pa] / A[pa]))),
                   "se_doc_geo": float(np.std(np.log(B[pa] / A[pa]), ddof=1) / np.sqrt(pa.sum())),
                   "t1_diff": float(v["t1"][tb].mean() - v["t1"][ta].mean()), "caps": {}}
            for c in CAPS[:-1]:
                Ac, tok, npos = doc_sums(v, v["kl"][ta], c); Bc, _, _ = doc_sums(v, v["kl"][tb], c)
                Ac, Bc, tok, npos = Ac[ok], Bc[ok], tok[ok], npos[ok]
                sdc, sduc = se_delta(Ac, Bc)
                rec["caps"][str(c)] = {"logr": float(np.log(Bc.sum() / Ac.sum())), "se_delta": sdc, "se_boot": se_boot(Ac, Bc, nb=500),
                                       "positions": int(npos.sum()), "tokens": int(tok.sum()), "max_share": float((Ac / Ac.sum()).max())}
            rows.append(rec); logr[(g, model[(g, ta)], model[(g, tb)])] = (lr, rec["se_boot"])
            # SE vs n by subsampling documents without replacement
            if len(scale) < 600:
                rng = np.random.default_rng(1); N = len(A); out = []
                for n in sorted({max(8, N // 8), N // 4, N // 2, (3 * N) // 4}):
                    vals = []
                    for _ in range(400):
                        i = rng.choice(N, n, replace=False); vals.append(np.log(B[i].sum() / A[i].sum()))
                    fpc = np.sqrt((N - n) / (N - 1))
                    out.append({"n": int(n), "sd_sub": float(np.std(vals)), "pred": float(sdu / np.sqrt(n) * fpc)})
                scale.append({"part": g, "a": ta, "b": tb, "N": N, "pts": out})
    with open(f"{a.out}/pairs.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(scale, open(f"{a.out}/scale.json", "w"))
    # cross-slice: the same two containers on two different goldens of the same part kind (x / c; single-part
    # calib slices such as golden-search27 / golden-verify27 count as c)
    ents = collections.defaultdict(dict)
    for (g, ma, mb), val in logr.items():
        ents[("x" if g.endswith("-x") else "c", ma, mb)][g] = val
    cross = []
    for (kind, ma, mb), d in ents.items():
        for g1, g2 in itertools.combinations(sorted(d), 2):
            (l1, s1), (l2, s2) = d[g1], d[g2]
            cross.append({"part": kind, "a": ma, "b": mb, "g1": g1, "g2": g2, "logr1": l1, "logr2": l2, "se1": s1, "se2": s2,
                          "z": (l1 - l2) / np.hypot(s1, s2)})
    with open(f"{a.out}/cross.jsonl", "w") as f:
        for r in cross: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} pair-parts, {len(cross)} cross-slice pairs")


if __name__ == "__main__":
    main()
