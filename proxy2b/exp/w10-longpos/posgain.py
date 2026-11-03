#!/usr/bin/env python3
"""posgain.py GLONG LONGDIR SHORTDIR[,DIR..] [--arms edy,eB,...] [--boot 2000] -- w10-longpos (ranked.md I4): does the
0.8B diagonal-H_out gain hold at positions > 2k? Score files: LONGDIR/<tag>.glong.f32 (evalq.py --out on GLONG, the
long search-pool golden) and <tag>.f32 on g08-search1 (SHORTDIR list); tags k<K><arm><sfx>, sfx "" / _s<n>.

Per arm, K and seed s, seed-paired with base (same rotation seed; both quants fixed, only the golden changes):
  bands by context position (sf + row): 0-2k / 2-8k / 8-32k / 32-64k; "short" = 0-2k, "long" = >= 2k.
  (a) within-doc (primary, position not category): docs with >= MINROWS gen rows in both short and long; per doc
      l = ln(sum KL_arm / sum KL_base) over its short rows and over its long rows; g_short / g_long = -mean over docs
      and seeds; D = g_long - 2/3 g_short, D3 = g_long - 1/3 g_short.
  (b) pooled (what the score sees, agentic traces included): l over all gen rows of a band; gain per band; long gain
      vs the g08-search1 gain (the documents the 0.8B results so far were measured on, capped at 2k).
SE: documents bootstrapped (all seeds of a replicate share its documents) and the seed spread (SD/sqrt(n)), added in
quadrature. Verdict per README's draft rule; --need-seeds prints "K arm seed" lines when the rule says INCONCLUSIVE.
Family arms (h-<sub>, w10-hout-attrib) get the per-band table only (family x position, L5), no verdict."""
import json, math, os, sys
import numpy as np

a = sys.argv[1:]
def opt(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
    return d
ARMS = opt("--arms", "edy,eB,h-gdn_in,h-attn_in,h-mlp_in,h-writers,h-d0,h-d1,h-d2").split(",")
NB = int(opt("--boot", 2000)); MINROWS = int(opt("--minrows", 500)); MAXSEED = int(opt("--maxseed", 5))
need = "--need-seeds" in a
if need: a.remove("--need-seeds")
gl, ldir, sdirs = a[0], a[1], a[2].split(",")
m = json.load(open(f"{gl}/kld.json")); R = m["rows"]
gen = np.fromfile(f"{gl}/gen.u8", np.uint8).astype(bool); assert len(gen) == R
pos = np.zeros(R, np.int64); doc = np.zeros(R, np.int64)
for j, d in enumerate(m["docs"]):
    n = d["tokens"] - 1 - d["score_from"]; pos[d["row0"]:d["row0"] + n] = d["score_from"] + np.arange(n); doc[d["row0"]:d["row0"] + n] = j
EDG = [0, 2048, 8192, 32768, 65536]; BN = ["0-2k", "2-8k", "8-32k", "32-64k"]
band = np.digitize(pos, EDG) - 1
ND = len(m["docs"])
sfx = lambda s: "" if s == 0 else f"_s{s}"

def rows(tag):
    f = f"{ldir}/{tag}.glong.f32"
    if not os.path.exists(f): return None
    r = np.fromfile(f, np.float32).reshape(-1, 4); assert len(r) == R, (f, len(r), R)
    return r[:, 0].astype(np.float64)
def short_kl(tag):
    for d in sdirs:
        f = f"{d}/{tag}.f32"
        if os.path.exists(f):
            r = np.fromfile(f, np.float32).reshape(-1, 4); g = r[:, 2] > 0; return float(r[g, 0].astype(np.float64).mean())

# per-doc x band sums of gen KL: S[doc, band]; "long" = bands 1..3 summed
def sums(x):
    S = np.zeros((ND, len(BN)))
    np.add.at(S, (doc[gen], band[gen]), x[gen]); return S
cnt = np.zeros((ND, len(BN)), np.int64); np.add.at(cnt, (doc[gen], band[gen]), 1)
el = np.where((cnt[:, 0] >= MINROWS) & (cnt[:, 1:].sum(1) >= MINROWS))[0]     # within-doc eligible
rng = np.random.default_rng(0)
B_all = rng.integers(0, ND, (NB, ND)); B_el = rng.integers(0, len(el), (NB, len(el)))

def analyse(K, arm):
    seeds, SA, SB = [], [], []
    for s in range(8):
        x, b = rows(f"k{K}{arm}{sfx(s)}"), rows(f"k{K}base{sfx(s)}")
        if x is None or b is None: continue
        seeds.append(s); SA.append(sums(x)); SB.append(sums(b))
    if not seeds: return None
    SA, SB = np.array(SA), np.array(SB)                    # [seed, doc, band]
    def pooled(ix):                                        # gains per band + long, from docs ix: [seed, 5]
        a_, b_ = SA[:, ix].sum(1), SB[:, ix].sum(1)        # [seed, band]
        with np.errstate(divide="ignore", invalid="ignore"):
            l = np.log(a_ / b_); ll = np.log(a_[:, 1:].sum(1) / b_[:, 1:].sum(1))
        return -np.concatenate([l, ll[:, None]], 1)
    def within(ix):                                        # [seed, (g_short, g_long, D)]
        a_, b_ = SA[:, el[ix]], SB[:, el[ix]]
        ls = np.log(a_[:, :, 0] / b_[:, :, 0]); lL = np.log(a_[:, :, 1:].sum(2) / b_[:, :, 1:].sum(2))
        gs, gL = -ls.mean(1), -lL.mean(1); return np.stack([gs, gL, gL - 2 / 3 * gs, gL - gs / 3], 1)
    P, W = pooled(np.arange(ND)), within(np.arange(len(el)))
    Pb = np.array([pooled(ix).mean(0) for ix in B_all]); Wb = np.array([within(ix).mean(0) for ix in B_el])
    n = len(seeds); sd = lambda v: v.std(0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(v.shape[1])
    se_P = np.sqrt(np.nanstd(Pb, 0) ** 2 + sd(P) ** 2); se_W = np.sqrt(Wb.std(0) ** 2 + sd(W) ** 2)
    o = {"K": K, "arm": arm, "seeds": seeds, "pooled": {k: [float(v), float(e)] for k, v, e in zip(BN + ["long"], P.mean(0), se_P)},
         "within": {k: [float(v), float(e)] for k, v, e in zip(("g_short", "g_long", "D", "D3"), W.mean(0), se_W)},
         "within_docs": int(len(el))}
    gs = []                                                 # g08-search1 gain, the same seeds
    for s in seeds:
        x, b = short_kl(f"k{K}{arm}{sfx(s)}"), short_kl(f"k{K}base{sfx(s)}")
        if x is not None and b is not None: gs.append(-math.log(x / b))
    if gs: o["search1"] = [float(np.mean(gs)), float(np.std(gs, ddof=1) / math.sqrt(len(gs))) if len(gs) > 1 else 0.0]
    if "search1" in o:
        g1, e1 = o["search1"]; gL, eL = o["pooled"]["long"]
        o["pooled_long_vs_search1"] = gL / g1 if g1 else None
        o["third_test_z"] = (gL - g1 / 3) / math.sqrt(eL ** 2 + (e1 / 3) ** 2) if (eL or e1) else None
    return o

def verdict(o):
    """HOLDS / PARTIAL / DECAYS / INCONCLUSIVE for one K (README draft rule)"""
    D, eD = o["within"]["D"]; D3, eD3 = o["within"]["D3"]; gL, eL = o["within"]["g_long"]
    r1, z3 = o.get("pooled_long_vs_search1"), o.get("third_test_z")
    if D3 < -2 * eD3 or (r1 is not None and r1 < 1 / 3 and (z3 or 0) <= -2): return "DECAYS"
    if D >= -eD and gL >= 2 * eL and (r1 or 0) >= 2 / 3: return "HOLDS"
    if D < -2 * eD: return "PARTIAL"
    return "INCONCLUSIVE"
def combine(vs):
    for v in ("DECAYS", "INCONCLUSIVE", "PARTIAL"):
        if v in vs: return v
    return "HOLDS"

res = {}
for K in (3, 2):
    for arm in ARMS:
        o = analyse(K, arm)
        if o: res[(K, arm)] = o
V = {}
for arm in ("edy", "eB"):
    vs = {K: verdict(res[(K, arm)]) for K in (3, 2) if (K, arm) in res}
    if len(vs) == 2:
        V[arm] = combine(list(vs.values()))
        V[arm + "_byK"] = vs
if need:     # primary arm edy: inconclusive -> two more seeds (pre-registered), at most MAXSEED
    if V.get("edy") == "INCONCLUSIVE":
        for K in (3, 2):
            n = len(res[(K, "edy")]["seeds"])
            for s in range(n, min(n + 2, MAXSEED)): print(K, "edy", s)
    sys.exit(0)
if V.get("edy") == "INCONCLUSIVE" and all(len(res[(K, "edy")]["seeds"]) >= MAXSEED for K in (3, 2)):   # after the extra seeds: point estimates
    pe = lambda o: "DECAYS" if o["within"]["D3"][0] < 0 else "HOLDS" if o["within"]["D"][0] >= 0 and \
        (o.get("pooled_long_vs_search1") or 0) >= 2 / 3 else "PARTIAL"
    V["edy"] = combine([pe(res[(K, "edy")]) for K in (3, 2)]) + " (point estimates after the extra seeds)"

base = {}
for K in (3, 2):
    b = rows(f"k{K}base")
    if b is not None: S = sums(b).sum(0); base[K] = (S / np.maximum(cnt.sum(0), 1)).tolist()
print(f"golden {gl}: {ND} docs, gen rows by band {dict(zip(BN, cnt.sum(0).tolist()))}; within-doc set {len(el)} docs")
for K, v in base.items(): print(f"base K{K} KL by band: " + " ".join(f"{n} {x:.4f}" for n, x in zip(BN, v)))
print(f"{'arm':10s} K seeds | pooled gain by band  " + " ".join(f"{n:>13s}" for n in BN + ["long"]) + " | search1 | within: g_short g_long D")
for (K, arm), o in res.items():
    p = " ".join(f"{o['pooled'][n][0]:+.3f}±{o['pooled'][n][1]:.3f}" for n in BN + ["long"])
    s1 = f"{o['search1'][0]:+.3f}" if "search1" in o else "  -   "
    w = " ".join(f"{o['within'][n][0]:+.3f}±{o['within'][n][1]:.3f}" for n in ("g_short", "g_long", "D"))
    print(f"{arm:10s} {K} {len(o['seeds'])}     | {p} | {s1} | {w}")
fs = {}
for (K, arm), o in res.items():     # family x position: a subset arm's share of edy's gain, short vs long (pooled bands)
    e = res.get((K, "edy"))
    if not arm.startswith("h-") or not e: continue
    sh = {b: o["pooled"][b][0] / e["pooled"][b][0] for b in ("0-2k", "long")}
    fs[f"k{K}{arm}"] = sh; print(f"share of edy's gain {arm:10s} K{K}: 0-2k {sh['0-2k']:.2f}  >=2k {sh['long']:.2f}")
for k, v in V.items(): print(f"VERDICT {k}: {v}")
print(json.dumps({"verdicts": V, "base_by_band": base, "family_shares": fs, "arms": {f"k{K}{arm}": o for (K, arm), o in res.items()}}))
