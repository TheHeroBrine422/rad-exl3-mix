#!/usr/bin/env python3
"""r94breport.py [DIR] -- the #94 rerun on w43's exllamav3-faithful proxy (settle.py), w45. DIR (default
data/e3/sub1/r94b) holds smoke.jsonl (+ smoke16.jsonl) and out/<cb>/s<seed>/<layer>-<tensor>-K<K>.jsonl.

Step 1 (sanity, as #94's rule): on the GPU smoke's own tiles (cbsmoke.py: rng 11 Gaussian, sd 1, no scale) the
settle core must give int8 / mul1 within +-0.05 of the GPU's x1.024 (K7) and x1.347 (K8).
Step 2 (#94's rule, unchanged; README "#94"): ratio = rel(mul1_i8) / rel(mul1) per tensor and K, each arm at its own
exllamav3 g_scale (searched with its own codebook, m = 1, refit on: settle.py's primary "rel"), median and p90 over
#94's 10 tensors on sample 0. An i8 level K7.25 / 7.5 / 7.75 joins the ladder if median <= 1.10 and p90 <= 1.20.
Reported beside it (no decision): sample 1, best of m in {0.9, 1, 1.1} per arm, before the refit, the exact exllamav3
codebooks (mul1x / mul1x_i8), each arm's g_scale."""
import glob, json, math, os, sys
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
D = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/e3/sub1/r94b"
T = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
     (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
KS = [6.5, 7.0, 7.25, 7.5, 7.75, 8.0]
GPU = {7.0: 1.024, 8.0: 1.347}

# ---- step 1
S = []
for f in ("smoke.jsonl", "smoke16.jsonl"):
    if os.path.exists(f"{D}/{f}"): S += [json.loads(l) for l in open(f"{D}/{f}")]
print(f"# #94 rerun report ({os.path.relpath(D, ROOT)})\n\n## Step 1: the GPU smoke's tiles (4,096 Gaussian, rng 11)\n")
print("int8 / mul1 MSE ratio; scale = tile sd fed to the trellis (the smoke: 1)\n")
scales = sorted({o["scale"] for o in S})
for st in ("settle", "exl3"):
    print(f"**{st} codebooks** ({', '.join(next((o['cb'] for o in S if o['set'] == st), []))})\n")
    print("| K | search | " + " | ".join(f"scale {c:g}" for c in scales) + " |")
    print("|---|---|" + "---|" * len(scales))
    for K in sorted({o["K"] for o in S}):
        for se in ("free", "tb1", "free16"):
            row = {o["scale"]: o["ratio"] for o in S if o["set"] == st and o["K"] == K and o["search"] == se}
            if row: print(f"| {K:g} | {se} | " + " | ".join(f"{row[c]:.3f}" if c in row else "-" for c in scales) + " |")
    print()
sane = {}
for K, g in GPU.items():
    v = [o["ratio"] for o in S if o["set"] == "settle" and o["K"] == K and o["scale"] == 1 and o["search"] in ("free", "tb1")]
    sane[K] = bool(v) and all(abs(x - g) <= 0.05 for x in v)
    print(f"Sanity K{K:g}: settle core at scale 1 {', '.join(f'{x:.3f}' for x in v) or '-'} vs GPU {g:.3f} -> "
          + ("ok" if sane[K] else "OFF"))
SANE = all(sane.values())

# ---- step 2
R = {}
for f in glob.glob(f"{D}/out/*/s*/*.jsonl"):
    for l in open(f):
        o = json.loads(l); R[(o["cb"], o["seed"], o["layer"], o["w"], float(o["bits"]))] = o


def ratios(a, b, seed, key="rel", how="m1"):
    out = {}
    for K in KS:
        r = []
        for l, w in T:
            x, y = R.get((b, seed, l, w, K)), R.get((a, seed, l, w, K))
            if not (x and y): continue
            if how == "m1": r.append(x[key]["1.0"] / y[key]["1.0"])
            else: r.append(min(x[key].values()) / min(y[key].values()))
        out[K] = r
    return out


def table(title, rr, rule=False):
    print(f"\n**{title}**\n\n| K | median | p90 | min | max | n |" + (" rule |" if rule else ""))
    print("|---|---|---|---|---|---|" + ("---|" if rule else ""))
    for K in KS:
        r = rr[K]
        if not r: print(f"| {K:g} | - | - | - | - | 0 |" + (" |" if rule else "")); continue
        md, p9 = float(np.median(r)), float(np.percentile(r, 90))
        tag = ""
        if rule and K in (7.25, 7.5, 7.75):
            tag = (" joins |" if md <= 1.10 and p9 <= 1.20 else " no |") if len(r) == len(T) else " (incomplete) |"
        elif rule: tag = " |"
        print(f"| {K:g} | {md:.3f} | {p9:.3f} | {min(r):.3f} | {max(r):.3f} | {len(r)} |" + tag)


print("\n## Step 2: #94's 10 tensors on settle.py's path (proxy v2, 128 rows = one output-Hadamard block)")
main = ratios("mul1", "mul1_i8", 0)
table("Primary: sample 0, mul1_i8 / mul1, each at its own g_scale (m = 1), after the sv refit", main, rule=True)
table("Sample 1 (replication)", ratios("mul1", "mul1_i8", 1))
table("Sample 0, best of m in {0.9, 1, 1.1} per arm", ratios("mul1", "mul1_i8", 0, how="best"))
table("Sample 0, before the refit (rel_norefit, m = 1)", ratios("mul1", "mul1_i8", 0, key="rel_norefit"))
table("Sample 0, exact exllamav3 codebooks (mul1x_i8 / mul1x)", ratios("mul1x", "mul1x_i8", 0))

print("\nper tensor, sample 0 (mul1_i8 / mul1, m = 1):\n\n```")
print("tensor            " + " ".join(f"{K:>6g}" for K in KS))
for l, w in T:
    cells = []
    for K in KS:
        x, y = R.get(("mul1_i8", 0, l, w, K)), R.get(("mul1", 0, l, w, K))
        cells.append(f"{x['rel']['1.0'] / y['rel']['1.0']:6.3f}" if x and y else "     -")
    print(f"L{l:<2d} {w:13s} " + " ".join(cells))
print("```\n\ng_scale (exllamav3 units) picked by each arm's own search, median over tensors, sample 0:\n")
print("| K | mul1 | mul1_i8 | mul1 rel (m = 1) median |")
print("|---|---|---|---|")
for K in KS:
    g = {cb: [R[(cb, 0, l, w, K)]["search"]["g_scale"] for l, w in T if (cb, 0, l, w, K) in R] for cb in ("mul1", "mul1_i8")}
    rel = [R[("mul1", 0, l, w, K)]["rel"]["1.0"] for l, w in T if ("mul1", 0, l, w, K) in R]
    print(f"| {K:g} | " + " | ".join(f"{np.median(v):.3f}" if v else "-" for v in g.values())
          + (f" | {np.median(rel):.2e} |" if rel else " | - |"))

def kequiv(l, w, rel):
    """the mul1 K with this rel on tensor (l, w), sample 0, m = 1: linear in log rel between the measured K
    (linear extrapolation from the end segments)"""
    ks = [K for K in KS if ("mul1", 0, l, w, K) in R]
    if len(ks) < 2: return None
    y = [math.log(R[("mul1", 0, l, w, K)]["rel"]["1.0"]) for K in ks]; t = math.log(rel)
    i = 0 if t > y[0] else next((j for j in range(len(ks) - 1) if y[j + 1] <= t), len(ks) - 2)
    return ks[i] + (t - y[i]) * (ks[i + 1] - ks[i]) / (y[i + 1] - y[i])


print("\nWhat an i8 level is worth: the mul1 K with the same proxy error (per tensor, sample 0, m = 1; median, range):\n")
print("| i8 K | mul1-equivalent K | bits lost to int8 rounding | i8 rel / i8 rel at K7 |")
print("|---|---|---|---|")
for K in KS:
    e, g = [], []
    for l, w in T:
        x, x7 = R.get(("mul1_i8", 0, l, w, K)), R.get(("mul1_i8", 0, l, w, 7.0))
        if x:
            k = kequiv(l, w, x["rel"]["1.0"])
            if k is not None: e.append(k)
            if x7: g.append(x["rel"]["1.0"] / x7["rel"]["1.0"])
    if e: print(f"| {K:g} | {np.median(e):.2f} ({min(e):.2f} .. {max(e):.2f}) | {K - np.median(e):.2f} | "
                + (f"{np.median(g):.3f} |" if g else "- |"))

print("\n## Verdict (#94's rule, sample 0)\n")
print(f"Sanity (step 1): {'ok' if SANE else 'FAILED'}")
for K in (7.25, 7.5, 7.75):
    r = main[K]
    if len(r) < len(T): print(f"- K{K:g}: incomplete (n = {len(r)})"); continue
    md, p9 = float(np.median(r)), float(np.percentile(r, 90))
    j = md <= 1.10 and p9 <= 1.20
    print(f"- K{K:g}: median {md:.3f} (<= 1.10), p90 {p9:.3f} (<= 1.20) -> " + ("joins the i8 ladder" if j else "does not join")
          + ("" if SANE else " [sanity failed: not trusted]"))
