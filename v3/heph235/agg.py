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
    """curve {K: value} (any K points); x = bits (K-equivalent); log-linear between neighbouring points, end segments extrapolated."""
    ks = sorted(curve); i = 0
    while i < len(ks) - 2 and x > ks[i + 1]: i += 1
    k0, k1 = ks[i], ks[i + 1]
    a, b = math.log(curve[k0]), math.log(curve[k1])
    return math.exp(a + (x - k0) / (k1 - k0) * (b - a))


def eq_bits(curve, val):
    """K-equivalent bits at which ours' error (log-linear curve through its points, end segments extrapolated) equals val."""
    ks = sorted(curve); lv = math.log(val)
    pts = [(k, math.log(curve[k])) for k in ks]
    i = 0
    while i < len(pts) - 2 and lv < pts[i + 1][1]: i += 1
    (k0, a), (k1, b) = pts[i], pts[i + 1]
    return k0 + (lv - a) / (b - a) * (k1 - k0)


mats = sorted({(r["layer"], r["e"], r["kind"]) for r in heph})
Ocurve = {}   # (matrix) -> {K: rel}
for m in mats:
    Ocurve[m] = {float(K): w211_base(m[0], m[1], m[2], K) for K in (1, 2, 3, 4)}
Ometr = {}    # (matrix, K) -> ours_metrics line
for r in ours: Ometr[((r["layer"], r["e"], r["kind"]), float(r["K"]))] = r
OcurveF = {m: {**Ocurve[m], **{k: Ometr[(m, k)]["rel"] for k in (1.5, 2.5) if (m, k) in Ometr}} for m in mats}   # + measured fractional points
Hl = {}       # (matrix, variant, K, L, seed) -> line
for r in heph: Hl[((r["layer"], r["e"], r["kind"]), r["variant"], r["K"], r["L"], r["seed"], 0)] = r   # one row subset per tag (TAG=allrows for the all-rows runs)

print(f"# w235 tables (tag {TAG}; {len(mats)} matrices with HEPH results, {len(ours)} ours_metrics lines)\n")
# --- cross-check
d = [abs(r["rel"] / Ocurve[m][K] - 1) for (m, K), r in Ometr.items() if m in Ocurve and K in (1.0, 2.0, 3.0, 4.0)]
if d: print(f"**Cross-check** ours_metrics rel vs w211 base (min over F3): {len(d)} lines, max |rel diff| {max(d):.2e}\n")


def table(variant="std", L=12, seed=1, rows_sel=0, offs=OFFS, title="", Ks=(1.0, 2.0, 3.0)):
    print(f"### {title or f'HEPH {variant} L={L} seed={seed}'}  (rate = K + Huffman exponents + {offs})\n")
    print("| K | HEPH rate (mean) | ours rate | geo ratio HEPH/ours same K | geo ratio rate-matched [95% CI over matrices] | hot | medium | cold | gate_up | down | # matrices HEPH wins (rate-matched) | bits ours saves at equal error (mean) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    curves = OcurveF if any(k % 1 for k in Ks) else Ocurve      # integer-K tables interpolate ours between its integer points only (TB, conservative for ours)
    for K in Ks:
        rm, same, rate_h, saved = {}, {}, [], []
        for m in mats:
            h = Hl.get((m, variant, K, L, seed, rows_sel))
            if h is None or Ocurve[m][1.0] is None or (K not in curves[m]): continue
            rh = K + h["exp"]["huff_bpw"] + offs; rate_h.append(rh)
            o_at = interp_ours(curves[m], rh - 16 / h["Kd"])
            rm[m] = h["rel"] / o_at; same[m] = h["rel"] / curves[m][K]
            saved.append(rh - 16 / h["Kd"] - eq_bits(curves[m], h["rel"]))
        if not rm: continue
        sel = lambda f: [v for m, v in rm.items() if f(m)]
        cl = lambda c: fmt(sel(lambda m: CLS[(m[0], m[1])] == c)) if sel(lambda m: CLS[(m[0], m[1])] == c) else "-"
        kd = lambda k: fmt(sel(lambda m: m[2] == k)) if sel(lambda m: m[2] == k) else "-"
        wins = sum(v < 1 for v in rm.values())
        print(f"| {K:g} | {np.mean(rate_h):.3f} | {K + 0.003:.3f} | {fmt(list(same.values()))} | **{fmt(list(rm.values()))}** | {cl('hot')} | {cl('medium')} | {cl('cold')} | {kd('gate_up')} | {kd('down')} | {wins}/{len(rm)} | {np.mean(saved):+.2f} |")
        res[K] = geo(list(rm.values()))
    print()
    return res


main_res = table("std", 12, 1, 0)
table("std", 12, 1, 0, offs=0.0, title="HEPH std L=12 without the 0.022 random-access offsets (Huffman exponents only)")
table("std", 16, 1, 0, title="HEPH std L=16 (K = 2 only is run)")
table("std", 12, 2, 0, title="HEPH std L=12, codebook seed 2")
table("std", 12, 1, 0, title="HEPH std L=12, fractional K (ours measured at K = 1.5 / 2.5, non-tail-biting legacy pattern search; interpolation uses all of ours' measured points)", Ks=(1.5, 2.5))
table("sw", 12, 1, 0, title="HEPH sw (diag H x s_g^2 weights) L=12", Ks=(1.5, 2.0, 2.5, 3.0))
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

# --- how much of the error is off-diagonal: rel(full H) / rel(diag H), geo mean over matrices
print("### rel(full H) / rel(diag H) per method (geo mean over matrices; < 1: error sits in directions the full H weights less than its diagonal says)\n")
print("| K | HEPH gate_up | ours gate_up | HEPH down | ours down |"); print("|---|---|---|---|---|")
for K in (1.0, 2.0, 3.0):
    cells = []
    for k in ("gate_up", "down"):
        hv = [Hl[(m, "std", K, 12, 1, 0)]["rel"] / Hl[(m, "std", K, 12, 1, 0)]["rel_diag"] for m in mats if m[2] == k and (m, "std", K, 12, 1, 0) in Hl]
        ov = [Ometr[(m, K)]["rel"] / Ometr[(m, K)]["rel_diag"] for m in mats if m[2] == k and (m, K) in Ometr]
        cells += [f"{geo(hv):.3f}" if hv else "-", f"{geo(ov):.3f}" if ov else "-"]
    print(f"| {K:g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
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
        print(f"| L{m[0]} e{m[1]} {m[2]} | {CLS[(m[0], m[1])]} | {K:g} | {h['rel']:.5f} | {Ocurve[m][K]:.5f} | {o_at:.5f} | {h['rel'] / o_at:.3f} |")

# --- interpolation check: ours measured at K = 1.5 / 2.5 vs the log-linear interpolation of its integer points
chk = {k: [] for k in (1.5, 2.5)}
for m in mats:
    for k in (1.5, 2.5):
        if (m, k) in Ometr:
            chk[k].append(Ometr[(m, k)]["rel"] / interp_ours(Ocurve[m], k))
if any(chk.values()):
    print("\n### Interpolation check: ours measured (fractional K, non-TB) / log-linear interpolation of ours' integer K points\n")
    for k, v in chk.items():
        if v: print(f"- K = {k:g}: geo {fmt(v)} over {len(v)} matrices")
# --- all-rows HEPH vs the 1/16 subset
ALL = {(r["layer"], r["e"], r["kind"], r["variant"], r["K"], r["L"], r["seed"]): r for r in rows(f"{DATA}/heph-allrows/*.jsonl")}
if ALL:
    print("\n### HEPH on ALL rows vs the 288 / 320-row subset (std L=12 seed 1; all-rows rel / subset rel per matrix, geo [CI]); all-rows frac_exact, time\n")
    print("| K | n | rel all / subset | gate_up | down | % bit-exact (all rows) | enc s per matrix gate_up / down |"); print("|---|---|---|---|---|---|---|")
    for K in (1.0, 2.0, 3.0):
        pr = [(m, ALL[(*m, "std", K, 12, 1)], Hl[(m, "std", K, 12, 1, 0)]) for m in mats if (*m, "std", K, 12, 1) in ALL and (m, "std", K, 12, 1, 0) in Hl]
        if not pr: continue
        rr = lambda f: [a["rel"] / b["rel"] for m, a, b in pr if f(m)]
        tm = lambda k: np.mean([a["enc_sec"] for m, a, b in pr if m[2] == k]) if any(m[2] == k for m, _, _ in pr) else float("nan")
        print(f"| {K:g} | {len(pr)} | {fmt(rr(lambda m: True))} | {fmt(rr(lambda m: m[2] == 'gate_up'))} | {fmt(rr(lambda m: m[2] == 'down'))} | {np.mean([a['frac_exact'] * 100 for _, a, _ in pr]):.1f} | {tm('gate_up'):.0f} / {tm('down'):.0f} |")
    # all-rows rate-matched ratio vs ours subset (unpaired rows; ours' absolute rel is good to ~+-4% per gate_up matrix, w211)
    print("\nAll-rows HEPH vs ours (ours on the subset; rows unpaired), rate-matched geo ratio [CI]:\n")
    for K in (1.0, 2.0, 3.0):
        v = []
        for m in mats:
            a = ALL.get((*m, "std", K, 12, 1))
            if a and Ocurve[m][1.0]: v.append(a["rel"] / interp_ours(Ocurve[m], K + a["exp"]["huff_bpw"] + OFFS - 16 / a["Kd"]))
        if v: print(f"- K = {K:g}: {fmt(v)} (n = {len(v)})")
