#!/usr/bin/env python3
"""predict.py OUT.json STATS.npz... [--proxy proxy.jsonl] -- A6 / D6 / B6 from stats.py output (blk.py's algebra).

Per linear (d = h sv^2):
  A6  R_pred: predicted two-sided / one-sided cost in the h metric (block-16 LDL, sigma_out 0.01); R_asym: the GM/AM
      bound; R_pred_ident: the identity-H_out arm (feedback designed on sv^2, scored in h).
      With measured errors (stats e/<tag>, tags k<K>{base,bdiag,bident}[_s1,_s2]): R_meas = mean_seeds sum h e(bdiag) /
      mean_seeds sum h e(base) (the h-weighted proxy: tr(E^T H_in E diag h) exactly); Spearman(R_pred, R_meas).
      Loss weight L_t = mean_seeds sum h e(base) / tr(H_in) (Fisher scale, stats.py); the estimate without quants
      (27B): L_est = eps_t ref_t sum(h sv^2) / sum(sv^2) / tr(H_in) (eps_t: --proxy rel error, else 1; ref_t =
      sum_c w^T H w), checked on the 0.8B.
  D6  across-128-block spread 1/2 log2(AM/GM) of the predicted two-sided block costs (the rate a K per output slab
      could save after within-block feedback), of the one-sided ones, and of measured bdiag block costs; with K2 and
      K3 donors (same seed): h-proxy of a half/half slab split, K3 on the blocks with the highest predicted cost,
      vs the same split at random block choice, vs uniform K2.5 (geometric mean of the K2 and K3 proxies).
  B6  MLP intermediate permutations (gate rows, up rows, down columns together): natural / random / balanced (snake
      deal by the L-weighted log d of gate and up) / sorted; the two-sided gate+up cost relative to natural, as an
      equivalent rate (bits); the down input side is downperm.py's.
"""
import argparse, collections, json, math, os, re, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blk

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("stats", nargs="+"); ap.add_argument("--proxy")
ap.add_argument("--perms-out", help="write B6's permutations (L<l>/balanced|sorted|random0) for downperm.py / permq.py")
ap.add_argument("--kl-gain", default="3:-0.081,2:-0.054", help="measured KL gain of bdiag vs base per K (3 seeds, #33-16)")
a = ap.parse_args()
Z = {}
for f in a.stats:
    z = np.load(f); Z.update({k: z[k] for k in z.files})
CFG = json.loads(str(Z.pop("_cfg"))) if "_cfg" in Z else {}
keys = sorted({k.split("/")[0] for k in Z}, key=lambda k: (int(k.split(".")[3]), k))
fam = lambda k: ".".join(k.split(".")[-2:])
eps = {}
if a.proxy:
    for l in open(a.proxy):
        d = json.loads(l); eps[d["key"]] = d["proxy_err"]
KLG = {int(x.split(":")[0]): float(x.split(":")[1]) for x in a.kl_gain.split(",")}
tags = sorted({k.split("/e/")[1] for k in Z if "/e/" in k})
KS = sorted({int(re.match(r"k(\d)", t).group(1)) for t in tags}, reverse=True)
def seeds(K, arm): return [t for t in tags if re.fullmatch(rf"k{K}{arm}(_s\d)?", t)]

rows = {}
for k in keys:
    h, sv2 = Z[f"{k}/h"], Z[f"{k}/sv2"]; d = np.maximum(h, 1e-30 * h.max()) * sv2
    two, one = blk.block_costs(d); ti, _ = blk.block_costs(sv2, d)
    am = d.reshape(-1, blk.HB).mean(1)
    r = {"key": k, "fam": fam(k), "n": len(d), "R_pred": two.sum() / one.sum(), "R_asym": float((blk.gm_am(d) * am).sum() / am.sum()),
         "R_pred_ident": ti.sum() / one.sum(), "d6_bits_two": blk.bits_spread(two), "d6_bits_one": blk.bits_spread(one),
         "hsv_cv": float(d.std() / d.mean()), "sum_hsv2": float(d.sum()), "sum_sv2": float(sv2.sum())}
    if f"{k}/h_even" in Z:                   # split-half cross-fit: design on one half, score in the other (both ways)
        de, do = (np.maximum(Z[f"{k}/{x}"], 1e-30 * Z[f"{k}/{x}"].max()) * sv2 for x in ("h_even", "h_odd"))
        r["R_cross"] = float(np.mean([(lambda c: c[0].sum() / c[1].sum())(blk.block_costs(x, y)) for x, y in ((de, do), (do, de))]))
        r["cos_halves"] = float((Z[f"{k}/h_even"] * Z[f"{k}/h_odd"]).sum() / np.linalg.norm(Z[f"{k}/h_even"]) / np.linalg.norm(Z[f"{k}/h_odd"]))
    if f"{k}/ref" in Z:
        ref = Z[f"{k}/ref"]; r["ref"] = float(ref.sum()); tr = float(Z[f"{k}/trH"])
        r["L_est"] = eps.get(k, 1.0) * r["ref"] * d.sum() / sv2.sum() / tr
    for K in KS:
        S = {arm: [float((h * Z[f"{k}/e/{t}"]).sum()) for t in seeds(K, arm)] for arm in ("base", "bdiag", "bident")}
        U = {arm: [float(Z[f"{k}/e/{t}"].sum()) for t in seeds(K, arm)] for arm in ("base", "bdiag")}
        if not S["base"] or not S["bdiag"]: continue
        r[f"L{K}"] = np.mean(S["base"]) / tr; r[f"R_meas{K}"] = np.mean(S["bdiag"]) / np.mean(S["base"])
        r[f"R_meas{K}_seeds"] = [b / c for b, c in zip(S["bdiag"], S["base"])]
        r[f"R_onesided{K}"] = np.mean(U["bdiag"]) / np.mean(U["base"])       # the proxy knap4 reads (F1)
        r[f"eps{K}"] = np.mean(U["base"]) / r["ref"]
        if S["bident"]: r[f"R_meas_ident{K}"] = np.mean(S["bident"]) / np.mean(S["base"])
        mb = np.mean([(h * Z[f"{k}/e/{t}"]).reshape(-1, blk.HB).sum(1) for t in seeds(K, "bdiag")], 0)
        r[f"d6_bits_meas{K}"] = blk.bits_spread(mb)
        sv = np.mean([np.corrcoef(np.log(sv2 + 1e-30), np.log(Z[f"{k}/e/{t}"] + 1e-30))[0, 1] for t in seeds(K, "base")])
        r[f"corr_logsv2_loge{K}"] = float(sv)
    rows[k] = r

out = {"tensors": rows, "summary": {}}
def P(*x): print(*x, flush=True)
def wmean(v, w): v, w = np.asarray(v, float), np.asarray(w, float); return float((v * w).sum() / w.sum())
fams = sorted({r["fam"] for r in rows.values()})
# ---------------------------------------------------------------- A6
P("== A6: predicted vs measured two-sided gain per tensor (h-weighted proxy) ==")
for K in KS:
    rr = [r for r in rows.values() if f"R_meas{K}" in r]
    if not rr: continue
    pr, me = [r["R_pred"] for r in rr], [r[f"R_meas{K}"] for r in rr]
    rho = spearmanr(pr, me).correlation; L = [r[f"L{K}"] for r in rr]
    b = np.polyfit(np.log(pr), np.log(me), 1)
    s = {"spearman": rho, "spearman_asym": spearmanr([r["R_asym"] for r in rr], me).correlation,
         "loglog_slope": b[0], "loglog_icpt": b[1],
         "agg_pred_gain": 1 - wmean(pr, L), "agg_meas_proxy_gain": 1 - wmean(me, L), "kl_gain": KLG.get(K),
         "agg_onesided_ratio": wmean([r[f"R_onesided{K}"] for r in rr], L),
         "spearman_Lest_L": spearmanr([r["L_est"] for r in rr], L).correlation,
         "agg_pred_gain_Lest": 1 - wmean(pr, [r["L_est"] for r in rr]),
         "median_corr_logsv2_loge": float(np.median([r[f"corr_logsv2_loge{K}"] for r in rr]))}
    seedrho = [spearmanr(pr, [r[f"R_meas{K}_seeds"][i] for r in rr]).correlation for i in range(len(rr[0][f"R_meas{K}_seeds"]))]
    s["spearman_per_seed"] = seedrho
    # seed-to-seed agreement of the measured ratio itself: the ceiling any predictor can reach
    sr = np.array([r[f"R_meas{K}_seeds"] for r in rr]); n = sr.shape[1]
    s["spearman_seed_vs_seed"] = [spearmanr(sr[:, i], sr[:, j]).correlation for i in range(n) for j in range(i + 1, n)]
    if all(f"R_meas_ident{K}" in r for r in rr):
        s["ident_spearman"] = spearmanr([r["R_pred_ident"] for r in rr], [r[f"R_meas_ident{K}"] for r in rr]).correlation
        s["ident_agg_pred"] = wmean([r["R_pred_ident"] for r in rr], L); s["ident_agg_meas"] = wmean([r[f"R_meas_ident{K}"] for r in rr], L)
    out["summary"][f"A6_K{K}"] = s
    P(f"K{K}: Spearman(R_pred, R_meas) {rho:.3f} (GM/AM bound {s['spearman_asym']:.3f}; per seed "
      f"{', '.join(f'{x:.3f}' for x in seedrho)}; measured seed vs seed {', '.join(f'{x:.3f}' for x in s['spearman_seed_vs_seed'])})")
    P(f"     log R_meas = {b[1]:+.3f} + {b[0]:.3f} log R_pred; L-weighted gain: predicted {s['agg_pred_gain']:.1%}, measured proxy "
      f"{s['agg_meas_proxy_gain']:.1%}, measured KL {KLG.get(K, float('nan')):+.1%}; one-sided proxy ratio bdiag/base "
      f"{s['agg_onesided_ratio']:.3f}")
    P(f"     L_est vs L: Spearman {s['spearman_Lest_L']:.3f}, gain with L_est {s['agg_pred_gain_Lest']:.1%}; "
      f"median corr(log sv2, log e_c) {s['median_corr_logsv2_loge']:.2f}")
    if "ident_spearman" in s:
        P(f"     identity H_out: predicted ratio {s['ident_agg_pred']:.3f}, measured {s['ident_agg_meas']:.3f}, Spearman {s['ident_spearman']:.3f}")
    P(f"     {'family':26s} {'n':>3s} {'R_pred':>7s} {'R_meas':>7s} {'1sided':>7s} {'rho':>6s} {'L share':>7s}")
    Lt = sum(L)
    for f in fams:
        g = [r for r in rr if r["fam"] == f]
        if not g: continue
        rf = spearmanr([r["R_pred"] for r in g], [r[f"R_meas{K}"] for r in g]).correlation if len(g) > 3 else float("nan")
        P(f"     {f:26s} {len(g):3d} {np.median([r['R_pred'] for r in g]):7.3f} {np.median([r[f'R_meas{K}'] for r in g]):7.3f} "
          f"{np.median([r[f'R_onesided{K}'] for r in g]):7.3f} {rf:6.2f} {sum(r[f'L{K}'] for r in g) / Lt:7.1%}")
if not KS:          # no quants (27B): predictions only, L_est-weighted when there is a ref
    rr = list(rows.values()); W = [r.get("L_est", 1.0) for r in rr]
    out["summary"]["A6_pred"] = {"agg_pred_gain_Lest": 1 - wmean([r["R_pred"] for r in rr], W),
                                 "median_R_pred": float(np.median([r["R_pred"] for r in rr]))}
    if "R_cross" in rr[0]:
        out["summary"]["A6_pred"]["agg_cross_gain_Lest"] = 1 - wmean([r["R_cross"] for r in rr], W)
        out["summary"]["A6_pred"]["median_cos_halves"] = float(np.median([r["cos_halves"] for r in rr]))
    P(f"A6 (no measurements): L_est-weighted predicted gain {out['summary']['A6_pred']['agg_pred_gain_Lest']:.1%} (self: design and "
      f"score on the full h, an upper bound at small n)" + (f"; split-half cross-fit {out['summary']['A6_pred']['agg_cross_gain_Lest']:.1%} "
      f"(a lower bound: a half-size design), halves median cos {out['summary']['A6_pred']['median_cos_halves']:.4f}" if "R_cross" in rr[0] else ""))
    for f in fams:
        g = [r for r in rr if r["fam"] == f]
        P(f"     {f:26s} {len(g):3d} R_pred median {np.median([r['R_pred'] for r in g]):.3f}  GM/AM {np.median([r['R_asym'] for r in g]):.3f}  CV(h sv2) {np.median([r['hsv_cv'] for r in g]):.2f}")

# ---------------------------------------------------------------- D6
P("\n== D6: across-block spread (bits a K per 128-row output slab could save) ==")
P(f"     {'family':26s} {'two(pred)':>9s} {'one(pred)':>9s} " + " ".join(f"{'meas K' + str(K):>9s}" for K in KS))
for f in fams:
    g = [r for r in rows.values() if r["fam"] == f]
    P(f"     {f:26s} {np.median([r['d6_bits_two'] for r in g]):9.3f} {np.median([r['d6_bits_one'] for r in g]):9.3f} "
      + " ".join(f"{np.median([r[f'd6_bits_meas{K}'] for r in g]):9.3f}" for K in KS if f"d6_bits_meas{K}" in g[0]))
LK = lambda r: r.get(f"L{KS[0]}") if KS else r.get("L_est", 1.0)     # loss weight: measured, else estimated
W = [LK(r) for r in rows.values()]
out["summary"]["D6"] = {"Lweighted_bits_two": wmean([r["d6_bits_two"] for r in rows.values()], W),
                        "gateup_median_bits_two": float(np.median([r["d6_bits_two"] for r in rows.values() if r["fam"] in ("mlp.gate_proj", "mlp.up_proj")]))}
P(f"     L-weighted mean (two-sided, predicted): {out['summary']['D6']['Lweighted_bits_two']:.3f} bit; gate/up median "
  f"{out['summary']['D6']['gateup_median_bits_two']:.3f} bit")
# the part a K per natural section gets (separate tensors / a per-part rate in the stacked kernel, no slab format):
# GDN in_proj_qkv = q | k | v (3 x 2048 at 0.8B, contiguous); attention q_proj = per head [query 256 | gate 256]
QK = CFG.get("linear_key_head_dim", 128) * CFG.get("linear_num_key_heads", 16)
HD = CFG.get("head_dim", 256)
def sections(k, n):
    if k.endswith("in_proj_qkv"): return np.repeat(np.arange(3), [QK, QK, n - 2 * QK])
    if k.endswith("self_attn.q_proj"): return (np.arange(n) // HD) % 2
    return None
for k, r in rows.items():
    sec = sections(k, r["n"])
    if sec is None: continue
    d = np.maximum(Z[f"{k}/h"], 1e-30 * Z[f"{k}/h"].max()) * Z[f"{k}/sv2"]
    two, _ = blk.block_costs(d)
    if k.endswith("in_proj_qkv"):
        c = np.array([two[sec[::blk.HB] == s].sum() for s in range(3)]); w = np.bincount(sec[::blk.HB], minlength=3)
    else:                                   # strided parts: reorder the channels so every part is contiguous (own tensor)
        p = np.argsort(sec, kind="stable"); tw, _ = blk.block_costs(d[p]); s2 = sec[p][::blk.HB]
        c = np.array([tw[s2 == s].sum() for s in range(2)]); w = np.bincount(s2)
    # free K per part, parts weighted by their size: saved rate = log2-mean form of 1/2 log2(AM/GM) with weights
    pc = c / w; ww = w / w.sum()
    r["d6_bits_sections"] = 0.5 * math.log2((ww * pc).sum() / math.exp((ww * np.log(pc)).sum()))
for f in ("linear_attn.in_proj_qkv", "self_attn.q_proj"):
    g = [r["d6_bits_sections"] for r in rows.values() if r["fam"] == f and "d6_bits_sections" in r]
    if g: P(f"     {f}: a K per section (qkv: q|k|v; q_proj: query|gate) saves median {np.median(g):.3f} bit "
            f"(range {min(g):.3f}..{max(g):.3f})")
if 2 in KS and 3 in KS:
    P("     donors K2 / K3 bdiag (same seed), half the blocks at K3 per tensor (bytes of K2.5):")
    tot = collections.defaultdict(float); fs = collections.defaultdict(lambda: collections.defaultdict(float))
    for k, r in rows.items():
        h = Z[f"{k}/h"]; nb = r["n"] // blk.HB
        if nb < 2: continue
        c, _ = blk.block_costs(np.maximum(h, 1e-30 * h.max()) * Z[f"{k}/sv2"]); top = np.argsort(-c)[:nb // 2]
        for s2, s3 in zip(seeds(2, "bdiag"), seeds(3, "bdiag")):
            b2 = (h * Z[f"{k}/e/{s2}"]).reshape(-1, blk.HB).sum(1); b3 = (h * Z[f"{k}/e/{s3}"]).reshape(-1, blk.HB).sum(1)
            b2, b3 = b2 / float(Z[f"{k}/trH"]), b3 / float(Z[f"{k}/trH"])      # Fisher scale: tensors add up
            m = np.zeros(nb, bool); m[top] = True; f2 = (nb - nb // 2) / nb
            v = {"sorted": b3[m].sum() + b2[~m].sum(), "random": (1 - f2) * b3.sum() + f2 * b2.sum(),
                 "K2.5": math.sqrt(b2.sum() * b3.sum()), "K2": b2.sum(), "K3": b3.sum()}
            for a_, x in v.items(): tot[a_] += x; fs[r["fam"]][a_] += x
    out["summary"]["D6_donors"] = dict(tot)
    P(f"     all: sorted / random {tot['sorted'] / tot['random']:.3f}, sorted / K2.5(est) {tot['sorted'] / tot['K2.5']:.3f}, "
      f"random / K2.5(est) {tot['random'] / tot['K2.5']:.3f}")
    for f in fams:
        v = fs[f]
        if v: P(f"     {f:26s} sorted/random {v['sorted'] / v['random']:.3f}  sorted/K2.5 {v['sorted'] / v['K2.5']:.3f}")

# ---------------------------------------------------------------- B6
P("\n== B6: MLP intermediate permutations (gate+up two-sided cost vs natural order) ==")
b6 = []; PERMS = {}
for l in sorted({int(k.split(".")[3]) for k in keys}):
    kg, ku = (f"model.language_model.layers.{l}.mlp.{x}_proj" for x in ("gate", "up"))
    if kg not in rows: continue
    dg, du = (np.maximum(Z[f"{x}/h"], 1e-30 * Z[f"{x}/h"].max()) * Z[f"{x}/sv2"] for x in (kg, ku))
    if ku not in rows: continue
    Lg, Lu = LK(rows[kg]), LK(rows[ku])
    wg, wu = Lg / (Lg + Lu), Lu / (Lg + Lu)
    def cost(p):         # L-weighted two-sided cost relative to one-sided (perm-invariant), and the slab GM version
        tg, og = blk.block_costs(dg[p]); tu, ou = blk.block_costs(du[p])
        c = wg * tg / og.sum() + wu * tu / ou.sum()
        return c.sum(), len(c) * math.exp(np.log(c).mean())
    key = wg * np.log(dg) + wu * np.log(du)
    n = len(dg); res = {"layer": l, "w_gate": wg}
    for nm, p in [("natural", np.arange(n)), ("balanced", blk.perm_balanced(key)), ("sorted", blk.perm_sorted(key))] + \
                 [(f"random{s}", blk.perm_random(n, s)) for s in range(4)]:
        res[nm] = cost(p)
        if nm in ("balanced", "sorted", "random0"): PERMS[f"L{l}/{nm}"] = p.astype(np.int64)
    rs = [res.pop(f"random{s}") for s in range(4)]
    res["random"] = tuple(np.mean([x[i] for x in rs]) for i in range(2))
    res["glob_gmam"] = wg * math.exp(np.log(dg).mean()) / dg.mean() + wu * math.exp(np.log(du).mean()) / du.mean()
    b6.append(res)
bits = lambda a_, b_: 0.5 * math.log2(a_ / b_)
if b6:
    P(f"     {'layer':>5s} {'natural':>8s} {'random':>8s} {'balanced':>8s} {'sorted':>8s} {'GM/AM':>6s}  bits bal  bits slab(sorted)  bits slab(nat)")
    for r in b6:
        P(f"     {r['layer']:5d} {r['natural'][0]:8.3f} {r['random'][0]:8.3f} {r['balanced'][0]:8.3f} {r['sorted'][0]:8.3f} "
          f"{r['glob_gmam']:6.3f}  {bits(r['natural'][0], r['balanced'][0]):+8.3f}  {bits(r['natural'][0], r['sorted'][1]):+16.3f}  "
          f"{bits(r['natural'][0], r['natural'][1]):+13.3f}")
    LW = [LK(rows[f"model.language_model.layers.{r['layer']}.mlp.gate_proj"]) + LK(rows[f"model.language_model.layers.{r['layer']}.mlp.up_proj"]) for r in b6]
    out["summary"]["B6"] = {"bits_balanced": wmean([bits(r["natural"][0], r["balanced"][0]) for r in b6], LW),
                            "bits_random": wmean([bits(r["natural"][0], r["random"][0]) for r in b6], LW),
                            "bits_sorted_slab": wmean([bits(r["natural"][0], r["sorted"][1]) for r in b6], LW),
                            "bits_natural_slab": wmean([bits(r["natural"][0], r["natural"][1]) for r in b6], LW),
                            "layers": b6}
    s = out["summary"]["B6"]
    P(f"     L-weighted: balanced {s['bits_balanced']:+.3f} bit, random {s['bits_random']:+.3f}, sorted + slab K "
      f"{s['bits_sorted_slab']:+.3f}, natural + slab K {s['bits_natural_slab']:+.3f} (positive = saves rate vs natural two-sided)")
if a.perms_out: np.savez(a.perms_out, **PERMS)
json.dump(out, open(a.out, "w"), indent=1, default=float)
