#!/usr/bin/env python3
"""agg.py DATADIR [W211_RES_DIR] -- tables of the w235 HEPH-vs-ours comparison (markdown on stdout).
DATADIR holds heph-<tag>/*.jsonl (heph_check.py) and ours-<tag>/*.jsonl (ours_metrics.py); TAG env (default main) picks the pair.
Ours' full-H rel at K = 1..4 comes from w211's base lines (best of factors {1, 1.15, 1.3}); ours_metrics' own K = 1..3 lines cross-check it.
Rate: ours = K + 16/Kd (fp16 row scale); HEPH = K + Huffman exponent cost (+ OFFS = 0.022 bpw random-access row offsets, the paper's number).
Rate-matched ratio: ours' error at HEPH's rate by log-linear interpolation between its integer-K points (error ~ 2^-2K).
"""
import glob, json, math, os, sys
import numpy as np

DATA = sys.argv[1]; W211 = sys.argv[2] if len(sys.argv) > 2 else "/workspace/rad-exl3-mix/data/ds41-quant/w211/res"
TAG = os.environ.get("TAG", "main"); OFFS = float(os.environ.get("OFFS", "0.022"))
experts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/ds41-quant/w211/experts.json")))
CLS = {}
for L, v in experts.items():
    for c in ("hot", "medium", "cold"):
        for x in v[c]: CLS[(int(L), x["e"])] = c


def rows(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l)
            if "K" in r: out.append(r)
    return out


heph = rows(f"{DATA}/heph-{TAG}/*.jsonl"); ours = rows(f"{DATA}/ours-{TAG}/*.jsonl")
rng = np.random.default_rng(0)


def w211_base(layer, e, kind, K):
    for l in open(f"{W211}/res-s0-{kind}-L{layer}-e{e}-base.jsonl"):
        r = json.loads(l)
        if r.get("K") == K and r.get("slice", 0) == 0: return min(r["allf"][f]["own"] for f in ("1.0", "1.15", "1.3"))
    return None


def geo(x): return float(np.exp(np.mean(np.log(x))))
def boot(x, n=2000):
    x = np.log(np.asarray(x)); b = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.exp(np.percentile(b, 2.5))), float(np.exp(np.percentile(b, 97.5)))
def fmt(x):
    g = geo(x); lo, hi = boot(x) if len(x) > 2 else (g, g)
    return f"{g:.3f} [{lo:.3f}, {hi:.3f}]"


def interp_ours(curve, x):
    """curve {K: value} at integer K; x = bits (K-equivalent); log-linear, extrapolating from the end segments."""
    ks = sorted(curve); k0 = min(max(int(math.floor(x)), ks[0]), ks[-1] - 1)
    a, b = math.log(curve[k0]), math.log(curve[k0 + 1])
    return math.exp(a + (x - k0) * (b - a))


mats = sorted({(r["layer"], r["e"], r["kind"]) for r in heph})
Ocurve = {}   # (matrix) -> {K: rel}
for m in mats:
    Ocurve[m] = {K: w211_base(m[0], m[1], m[2], K) for K in (1, 2, 3, 4)}
Ometr = {}    # (matrix, K) -> ours_metrics line
for r in ours: Ometr[((r["layer"], r["e"], r["kind"]), r["K"])] = r
Hl = {}       # (matrix, variant, K, L, seed) -> line
for r in heph: Hl[((r["layer"], r["e"], r["kind"]), r["variant"], r["K"], r["L"], r["seed"], 0)] = r   # one row subset per tag (TAG=allrows for the all-rows runs)

print(f"# w235 tables (tag {TAG}; {len(mats)} matrices with HEPH results, {len(ours)} ours_metrics lines)\n")
# --- cross-check
d = [abs(r["rel"] / Ocurve[m][int(K)] - 1) for (m, K), r in Ometr.items() if m in Ocurve and float(K).is_integer() and Ocurve[m].get(int(K))]
if d: print(f"**Cross-check** ours_metrics rel vs w211 base (min over F3): {len(d)} lines, max |rel diff| {max(d):.2e}\n")


def table(variant="std", L=12, seed=1, rows_sel=0, offs=OFFS, title=""):
    print(f"### {title or f'HEPH {variant} L={L} seed={seed}'}  (rate = K + Huffman exponents + {offs})\n")
    print("| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for K in (1.0, 2.0, 3.0):
        rm, same, rate_h = {}, {}, []
        for m in mats:
            h = Hl.get((m, variant, K, L, seed, rows_sel))
            if h is None or Ocurve[m][1] is None: continue
            rh = K + h["exp"]["huff_bpw"] + offs; rate_h.append(rh)
            o_at = interp_ours(Ocurve[m], rh - 16 / h["Kd"])
            rm[m] = h["rel"] / o_at; same[m] = h["rel"] / Ocurve[m][int(K)]
        if not rm: continue
        sel = lambda f: [v for m, v in rm.items() if f(m)]
        cl = lambda c: fmt(sel(lambda m: CLS[(m[0], m[1])] == c)) if sel(lambda m: CLS[(m[0], m[1])] == c) else "-"
        kd = lambda k: fmt(sel(lambda m: m[2] == k)) if sel(lambda m: m[2] == k) else "-"
        wins = sum(v < 1 for v in rm.values())
        print(f"| {K:g} | {np.mean(rate_h):.3f} | {K + 0.003:.3f} | {fmt(list(same.values()))} | **{fmt(list(rm.values()))}** | {cl('hot')} | {cl('medium')} | {cl('cold')} | {kd('gate_up')} | {kd('down')} | {wins}/{len(rm)} |")
        res[K] = geo(list(rm.values()))
    print()
    return res


main_res = table("std", 12, 1, 0)
table("std", 12, 1, 0, offs=0.0, title="HEPH std L=12 without the 0.022 random-access offsets (Huffman exponents only)")
table("std", 16, 1, 0, title="HEPH std L=16 (K = 2 only is run)")
table("std", 12, 2, 0, title="HEPH std L=12, codebook seed 2")
table("sw", 12, 1, 0, title="HEPH sw (diag H x s_g^2 weights) L=12")
table("h1", 12, 1, 0, title="HEPH h1 (no calibration, h = 1) L=12")
if main_res:
    ks = sorted(main_res)
    print("**Crossover (rate-matched geo ratio = 1, log-linear in K):** " + ", ".join(f"K{k:g}={main_res[k]:.3f}" for k in ks))
    for a, b in zip(ks, ks[1:]):
        if (main_res[a] - 1) * (main_res[b] - 1) < 0:
            x = a + (b - a) * math.log(main_res[a]) / (math.log(main_res[a]) - math.log(main_res[b])); print(f"  crossing between K{a:g} and K{b:g} at K ~ {x:.2f}")
    print()

# --- other metrics (diag H, MSE) at equal K and rate-matched, needs ours_metrics K = 1..4
for met in ("rel_diag", "rel_mse"):
    print(f"### {met}: HEPH std L=12 / ours (same K; rate-matched needs ours K = 1..4)\n")
    print("| K | same-K ratio geo [CI] | rate-matched geo [CI] |"); print("|---|---|---|")
    for K in (1.0, 2.0, 3.0):
        same, rm = [], []
        for m in mats:
            h = Hl.get((m, "std", K, 12, 1, 0)); o = Ometr.get((m, K))
            if h is None or o is None: continue
            same.append(h[met] / o[met])
            cur = {k: Ometr[(m, float(k))][met] for k in (1, 2, 3, 4) if (m, float(k)) in Ometr}
            if len(cur) == 4:
                rm.append(h[met] / interp_ours(cur, K + h["exp"]["huff_bpw"] + OFFS - 16 / h["Kd"]))
        print(f"| {K:g} | {fmt(same) if same else '-'} | {fmt(rm) if rm else '-'} |")
    print()

# --- bit-exactness, on-grid, exponent cost, timings
print("### Exactness / on-grid / exponent cost / encode time (HEPH std L=12 seed 1)\n")
print("| K | % bit-exact (mean, min..max) | on-grid all | infeasible seqs | enc ms per row gate_up / down |"); print("|---|---|---|---|---|")
for K in (1.0, 2.0, 3.0):
    hs = [(m, Hl[(m, "std", K, 12, 1, 0)]) for m in mats if (m, "std", K, 12, 1, 0) in Hl]
    if not hs: continue
    fe = [h["frac_exact"] * 100 for _, h in hs]
    t = lambda k: np.mean([h["sec_per_row"] * 1000 for m, h in hs if m[2] == k]) if any(m[2] == k for m, _ in hs) else float("nan")
    print(f"| {K:g} | {np.mean(fe):.1f} ({min(fe):.1f}..{max(fe):.1f}) | {all(h['on_grid'] for _, h in hs)} | {sum(h['bad_seq'] for _, h in hs)} | {t('gate_up'):.1f} / {t('down'):.1f} |")
print()
ex = [(m, Hl[(m, "std", 3.0, 12, 1, 0)]["exp"]) for m in mats if (m, "std", 3.0, 12, 1, 0) in Hl]
if ex:
    print("Exponent stream (all rows of each matrix; Huffman code per matrix; first byte of each row raw):\n")
    print("| kind | huff bpw mean [min..max] | entropy bits/delta | frac delta=0 | frac |delta|<=1 | first-byte bpw |"); print("|---|---|---|---|---|---|")
    for k in ("gate_up", "down"):
        e = [x for m, x in ex if m[2] == k]
        if e: print(f"| {k} | {np.mean([x['huff_bpw'] for x in e]):.4f} [{min(x['huff_bpw'] for x in e):.4f}..{max(x['huff_bpw'] for x in e):.4f}] | {np.mean([x['delta_entropy'] for x in e]):.3f} | {np.mean([x['frac_delta0'] for x in e]):.3f} | {np.mean([x['frac_delta_pm1'] for x in e]):.4f} | {np.mean([x['first_byte_bpw'] for x in e]):.4f} |")
    print()
print("### Per matrix (std L=12 seed 1; rel under the expert's full H; ours = w211 base best-of-3; ratio rate-matched)\n")
print("| matrix | class | K | HEPH rel | ours rel (same K) | ours at HEPH rate | ratio |"); print("|---|---|---|---|---|---|---|")
for m in mats:
    for K in (1.0, 2.0, 3.0):
        h = Hl.get((m, "std", K, 12, 1, 0))
        if h is None: continue
        o_at = interp_ours(Ocurve[m], K + h["exp"]["huff_bpw"] + OFFS - 16 / h["Kd"])
        print(f"| L{m[0]} e{m[1]} {m[2]} | {CLS[(m[0], m[1])]} | {K:g} | {h['rel']:.5f} | {Ocurve[m][int(K)]:.5f} | {o_at:.5f} | {h['rel'] / o_at:.3f} |")
