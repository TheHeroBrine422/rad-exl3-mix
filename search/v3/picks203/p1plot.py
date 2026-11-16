# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p1plot.py CANDS.json COORD_ROOT OUT.png [STATUS.txt] -- phase-1 progress chart (RULES-picks am. 2026-11-16).

One panel per size: every candidate scored so far on the phase-1 slices (s195c2-c5), as its size-adjusted pooled KLD
vs the size's reference, in %, with a paired per-document bootstrap 95 % interval (1000 draws, seed 203). Reference =
the size's live search incumbent once scored, else the best candidate so far (panel says which).
Pairing uses the documents both models have (slices finished by both). Filled = all 4 slices, hollow = partial.
Colour = why the candidate is in the set. b = the driver's current b (provisional; the frozen measured b replaces it
once the incumbents are scored)."""
import json, math, os, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cp, root, outp = sys.argv[1:4]
stp = sys.argv[4] if len(sys.argv) > 4 else None
SL = os.environ.get("P1_SLICES", "s195c2,s195c3,s195c4,s195c5").split(",")   # env overrides: tests
PFX = os.environ.get("P1_PREFIX", "p1-")
C = json.load(open(cp))
bpw, docs = {}, {}
for l in open(f"{root}/results.jsonl"):
    if f'"{PFX}' not in l: continue
    r = json.loads(l)
    if r.get("fail") or r.get("stopped") or r.get("slice") not in SL or not r["tag"].startswith(PFX): continue
    t = r["tag"][len(PFX):]; bpw[t] = r["bytes"] * 8 / 26895998464 if "bytes" in r else None
    p = f"{root}/docs/{r['tag']}-{r['slice']}.json"
    if os.path.exists(p): 
        m = json.load(open(p))["m"]
        if any(not x or len(x) < 2 or x[0] is None for x in m): continue
        docs.setdefault(t, {})[r["slice"]] = np.array([[x[0], x[1]] for x in m], dtype=float)
B = {}
if stp and os.path.exists(stp):
    m = re.search(r"\| b ([^|]*)", open(stp).read().strip().split("\n")[-1])
    if m: B = {float(a): float(b) for a, b in (x.split(":") for x in m.group(1).split())}
COL = {"incumbent": "#d62728", "pool": "#ff7f0e", "confirmed": "#1f77b4", "reselect-s186e": "#2ca02c", "narrow-miss": "#9467bd"}
rng = np.random.default_rng(203)


def pair(a, b):
    """common slices -> per-doc (klA, klB) arrays"""
    s = [x for x in SL if x in docs.get(a, {}) and x in docs.get(b, {})]
    if not s: return None
    A = np.concatenate([docs[a][x][:, 0] for x in s]); Bb = np.concatenate([docs[b][x][:, 0] for x in s])
    return A, Bb, len(s)


sizes = sorted(C, key=float); n = len(sizes); nc = 4; nr = math.ceil(n / nc)
fig, axs = plt.subplots(nr, nc, figsize=(16, 3.2 * nr), squeeze=False)
ct = {c["tag"] for v in C.values() for c in v}; tot = len(ct); full = sum(1 for t in ct if len(docs.get(t, {})) == len(SL)); part = sum(1 for t in ct if 0 < len(docs.get(t, {})) < len(SL))
for k, s in enumerate(sizes):
    ax = axs[k // nc][k % nc]; S = float(s); cands = C[s]
    tags = [c["tag"] for c in cands if c["tag"] in docs]
    inc = next((c["tag"] for c in cands if "incumbent" in c["why"]), None)
    b = B.get(S, 0.0)
    adj = lambda t: b * ((bpw.get(t) or S) - S)
    if inc in docs: ref, rlab = inc, "vs search incumbent"
    else:
        best = None
        for t in tags:
            v = (-len(docs[t]), math.log(sum(docs[t][x][:, 0].sum() for x in docs[t]) / sum(docs[t][x][:, 1].sum() for x in docs[t])) + adj(t))
            if best is None or v < best[0]: best = (v, t)
        ref, rlab = (best[1], "vs best so far") if best else (None, "nothing scored yet")
    rows = []
    for c in cands:
        t = c["tag"]
        if t not in docs: continue
        why = next((w for w in ("incumbent", "pool", "confirmed", "reselect-s186e", "narrow-miss") if w in c["why"]), c["why"][0])
        if ref is None or t == ref:
            rows.append((0.0, 0.0, 0.0, t, why, len(docs[t]))); continue
        pr = pair(t, ref)
        if pr is None: continue
        A, R, _ = pr; d = adj(t) - adj(ref)
        est = math.log(A.sum() / R.sum()) + d
        idx = rng.integers(0, len(A), (1000, len(A)))
        bs = np.log(A[idx].sum(1) / R[idx].sum(1)) + d
        rows.append((est, *np.percentile(bs, [2.5, 97.5]), t, why, len(docs[t])))
    rows.sort(key=lambda r: r[0])
    for i, (e, lo, hi, t, why, ns) in enumerate(rows):
        c = COL.get(why, "gray"); y = len(rows) - 1 - i
        ax.plot([100 * (math.exp(lo) - 1), 100 * (math.exp(hi) - 1)], [y, y], color=c, lw=1.2, alpha=0.7)
        ax.plot(100 * (math.exp(e) - 1), y, "o", ms=5, mfc=c if ns == len(SL) else "white", mec=c, mew=1.3)
        lab = t.split("-", 2)[2] + ("" if ns == len(SL) else f" ({ns}/4)")
        ax.text(1.0, y, lab, transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", ha="left", color="#333")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks([]); ax.set_ylim(-0.7, max(len(rows), 1) - 0.3)
    ax.set_title(f"{S:g} bpw — {len(rows)}/{len(cands)} scored, {rlab}" + (f", b {b:.2f}" if b else ""), fontsize=8.5)
    ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3)
    if k // nc == nr - 1 or k + nc >= n: ax.set_xlabel("size-adjusted KLD vs reference (%), 95 % CI", fontsize=7.5)
for k in range(n, nr * nc): axs[k // nc][k % nc].axis("off")
h = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in COL.items()]
h += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="gray", label="partial (slices done/4)")]
fig.legend(handles=h, loc="upper center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle(f"Phase 1 (s195c2–c5, ~1.8M rows): {full} complete + {part} partial of {tot} candidates — lower is better", fontsize=11, y=1.015)
fig.tight_layout(rect=(0, 0, 0.93, 0.98)); fig.savefig(outp + ".part.png", dpi=110, bbox_inches="tight"); os.replace(outp + ".part.png", outp)
print(outp, full, part, tot)
