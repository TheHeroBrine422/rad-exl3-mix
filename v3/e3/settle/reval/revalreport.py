#!/usr/bin/env python3
"""revalreport.py DATADIR [--m 1.0] [--sub1json OUT] -- w46's re-validation tables and verdicts (reval.md), each
rule applied with its ORIGINAL text (v3/e3/sub1/README.md "#90-02", "#100" + amendment, "#90"; actblk/README.md
"#106"), on settle.py's numbers: every *.jsonl under DATADIR (out/<mode>/<cb>/s<seed>/<layer>-<tensor>-K<k>.jsonl from
both boxes). Primary statistic: rel after the per-output-channel refit at m = 1 (exllamav3's own g_scale search, the
quantizer's behaviour; --m picks another multiplier, "best" = per job the best of the multipliers run).
Samples: seed s = sign seeds s + row block s (w43's convention). Per tensor, the samples are pooled by the geometric
mean of the ratios; "weighted" = the whole-tensor parameter count (ffn_gate_up = gate + up) as in the original
reports."""
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--m", default="1.0"); ap.add_argument("--sub1json", default=None)
a = ap.parse_args()

G, O = {}, {}                                              # gain: (cb, layer, w, K, seed) -> line; order: (layer, w, K, seed)
for p in glob.glob(os.path.join(a.data, "**", "*.jsonl"), recursive=True):
    for l in open(p):
        try: o = json.loads(l)
        except ValueError: continue
        if o.get("exp") == "gain": G[(o["cb"], o["layer"], o["w"], float(o["bits"]), o["seed"])] = o
        elif o.get("exp") == "order": O[(o["layer"], o["w"], float(o["bits"]), o["seed"])] = o
gm = lambda v: float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")
size = lambda o: o["N"] * o["K"] * (2 if o["w"] == "ffn_gate_up" else 1)


def rel(cb, l, w, K, s, m=None):
    o = G.get((cb, l, w, K, s))
    if not o: return None
    m = m or a.m
    if m == "best": return min(o["rel"].values())
    return o["rel"].get(m) if m in o["rel"] else o["rel"].get(str(float(m)))


def ratio(cba, cbb, l, w, K, seeds, m=None):
    """seed-pooled geometric mean of rel(cba) / rel(cbb), and the per-seed ratios"""
    r = {}
    for s in seeds:
        x, y = rel(cba, l, w, K, s, m), rel(cbb, l, w, K, s, m)
        if x is not None and y is not None: r[s] = x / y
    return (gm(list(r.values())) if r else None), r


T10 = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
       (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
T10b = [(26, "ffn_down"), (39, "ffn_down"), (6, "ssm_qkv"), (46, "ssm_z"), (39, "attn_output"), (6, "ffn_gate_up"),
        (26, "ssm_out"), (46, "ffn_gate_up"), (58, "ssm_qkv"), (39, "attn_qg")]
T20 = T10 + T10b
wt = {}
for (cb, l, w, K, s), o in G.items(): wt[(l, w)] = size(o)
fmtK = lambda K: f"{K:g}"
K9002 = {0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5}     # #90-02's own K grid (i4run.sh)


def item_9002(T, seeds, label, m=None, verbose=True, cba="int4grid", cbb="mul1_i8", Kset=None):
    """#90-02's table + rule on tensors T, samples seeds -> top K of the passing run"""
    Ks = sorted({k[3] for k in G if k[0] == cba and (k[1], k[2]) in T and k[4] in seeds} &
                {k[3] for k in G if k[0] == cbb and (k[1], k[2]) in T and k[4] in seeds} & (Kset or K9002))
    res, rows = {}, []
    for K in Ks:
        r = {t: ratio(cba, cbb, t[0], t[1], K, seeds, m)[0] for t in T}
        r = {t: v for t, v in r.items() if v is not None}
        if not r: continue
        wm = sum(r[t] * wt[t] for t in r) / sum(wt[t] for t in r)
        tmax = max(r, key=r.get); over = [t for t in r if r[t] > 1.03]
        ok = wm <= 1.01 and not over
        res[K] = (ok, len(r) == len(T))
        rows.append(f"| {fmtK(K)} | {wm:.4f} | {np.median(list(r.values())):.4f} | {r[tmax]:.4f} (L{tmax[0]} {tmax[1]}) | "
                    f"{len(over) or '-'} | {len(r)} | {'yes' if ok else 'no'} |")
    top = None
    for K in Ks:
        if K not in res or not res[K][0]: break
        top = K
    late = [K for K in Ks if res.get(K, (False,))[0] and (top is None or K > top)]
    complete = all(c for _, c in res.values())
    if verbose:
        print(f"\n#### {label}\n\n| K | weighted mean i4/i8 | median | max (tensor) | tensors > 1.03 | n | pass |\n|---|---|---|---|---|---|---|")
        print("\n".join(rows))
        print(f"\nRule #90-02: passing run from K{fmtK(Ks[0]) if Ks else '?'} up to "
              + (f"**K{fmtK(top)}**" if top is not None else "none")
              + (f" (isolated passes above it: {', '.join('K' + fmtK(K) for K in late)})" if late else "")
              + ("" if complete else "  [INCOMPLETE]"))
    return top, late, complete


def item_100(T, seeds, cb="iqrh", ref="mul1_i8", label="", rule=True, m=None):
    Ks = sorted({k[3] for k in G if k[0] == cb and (k[1], k[2]) in T and k[4] in seeds})
    print(f"\n#### {label}\n\n| K | n | weighted ratio | mean ± SE | min | max (tensor) | {ref} slope (log2/bit) | equal-quality bpw cost |\n|---|---|---|---|---|---|---|---|")
    W = {}; worst = {}
    allK = sorted({k[3] for k in G if k[0] == ref and (k[1], k[2]) in T})
    for K in Ks:
        r = {t: ratio(cb, ref, t[0], t[1], K, seeds, m)[0] for t in T}
        r = {t: v for t, v in r.items() if v is not None}
        if not r: continue
        wm = sum(r[t] * wt[t] for t in r) / sum(wt[t] for t in r)
        v = np.array(list(r.values())); tmax = max(r, key=r.get)
        # ref's own local slope of log2 rel per bit: central difference over the K grid's neighbours (param-weighted)
        i = allK.index(K) if K in allK else -1; sl = float("nan")
        if 0 <= i:
            lo, hi = allK[max(i - 1, 0)], allK[min(i + 1, len(allK) - 1)]
            num = den = 0.0
            for t in r:
                x = [rel(ref, t[0], t[1], lo, s, m) for s in seeds]; y = [rel(ref, t[0], t[1], hi, s, m) for s in seeds]
                x = [q for q in x if q]; y = [q for q in y if q]
                if x and y and hi > lo: num += wt[t] * (math.log2(gm(x)) - math.log2(gm(y))) / (hi - lo); den += wt[t]
            sl = num / den if den else float("nan")
        W[K] = wm; worst[K] = (r[tmax], tmax, len(r))
        print(f"| {fmtK(K)} | {len(r)} | {wm:.4f} | {v.mean():.4f} ± {v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0:.4f} | "
              f"{v.min():.4f} | {r[tmax]:.4f} ({tmax[0]}:{tmax[1]}) | {sl:.2f} | {math.log2(wm) / sl if sl == sl and sl else float('nan'):+.4f} |")
    if not rule: return
    in16 = [K for K in W if 1 <= K <= 6]; in25 = [K for K in W if 2 <= K <= 5]
    adopt = all(W[K] <= 1.005 and worst[K][0] <= 1.02 for K in in16) and len(in16) >= 6
    reject = any(W[K] > 1.05 for K in in25)
    v = "ADOPT" if adopt else "REJECT" if reject else "NO AUTOMATIC CALL (amendment: the user decides from this table + measured speed)"
    print(f"\nRule #100 (+ amendment): weighted ratio <= 1.005 at every K in 1-6 and no tensor > 1.02 -> adopt; > 1.05 at any "
          f"K in 2-5 -> reject; else no automatic call. Max in K2-5: {max((W[K] for K in in25), default=float('nan')):.4f}. "
          f"**Verdict: {v}**")


def item_106(T, seeds, label, rule=True):
    fams = {"attn_output": "o_proj", "ssm_out": "out_proj"}
    print(f"\n#### {label}\n")
    for fam, nm in fams.items():
        TT = [t for t in T if t[1] == fam]
        if not TT: continue
        print(f"\n{nm} ({fam}): ratio = rel(arm) / rel(natural), geometric mean over tensors x seeds\n")
        print("| K | n | head | free (bound) | rev (sign check) | worst tensor head | tensors > 1.00 | p90/p10 (metric) |\n|---|---|---|---|---|---|---|---|")
        passK = []; anyworse = False
        for K in sorted({k[2] for k in O}):
            per = defaultdict(list); pt = defaultdict(list); sp = []
            for (l, w) in TT:
                for s in seeds:
                    o = O.get((l, w, K, s))
                    if not o: continue
                    for arm in ("head", "free", "rev"): per[arm].append(o["rel"][arm] / o["rel"]["natural"])
                    pt[(l, w)].append(o["rel"]["head"] / o["rel"]["natural"]); sp.append(o["spread_p90p10"])
            if not per["head"]: continue
            tp = {t: gm(v) for t, v in pt.items()}; tw = max(tp, key=tp.get); worse = [t for t in tp if tp[t] > 1.0]
            g = gm(per["head"]); inrule = K in (2.0, 3.0, 4.0)          # the rule's K; others (i4k) reported only
            if inrule:
                anyworse |= bool(worse)
                if g <= 0.98: passK.append(K)
            print(f"| {fmtK(K)}{'' if inrule or not rule else ' (reported)'} | {len(per['head'])} | {g:.4f} | {gm(per['free']):.4f} | {gm(per['rev']):.4f} | "
                  f"{tp[tw]:.4f} (L{tw[0]}) | {len(worse) or '-'} | {np.median(sp):.2f} |")
        if rule:
            ok = len(passK) >= 2 and not anyworse
            print(f"\nRule #106 ({nm}): head <= 0.98 at >= 2 of K2/3/4 ({', '.join('K' + fmtK(K) for K in passK) or 'none'}) "
                  f"and no rule tensor > 1.00 ({'none' if not anyworse else 'some'}) -> **{'ADOPT' if ok else 'NOT ADOPTED'}**")
    if not rule:
        print("\nper tensor (seed 0): head / natural at K2 / K3 / K4, metric p90/p10")
        for (l, w) in sorted({(k[0], k[1]) for k in O if (k[0], k[1]) in T}, key=lambda t: (t[1], t[0])):
            v = [O.get((l, w, K, 0)) for K in (2.0, 3.0, 4.0)]
            print(f"  L{l:<2d} {w:12s} " + " ".join(f"{o['rel']['head'] / o['rel']['natural']:.4f}" if o else "   -  " for o in v)
                  + (f"  {v[0]['spread_p90p10']:.2f}" if v[0] else ""))


T90 = [(l, w) for l in (2, 10, 30, 50, 62) for w in ("ssm_qkv", "ssm_z", "ssm_out", "ffn_gate_up", "ffn_down")] + \
      [(l, w) for l in (3, 19, 43, 63) for w in ("attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down")]
K90 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]


def item_90(cb, seeds, label, old=None):
    """rel(K) / rel(K1) per family (geometric mean over its tensors, samples pooled), #90's rule 1 (monotone at every
    1/8 step within 1%), and the old proxy's curve beside it"""
    fam = defaultdict(lambda: defaultdict(list)); mono = []
    for (l, w) in T90:
        for s in seeds:
            r1 = rel(cb, l, w, 1.0, s)
            if not r1: continue
            prev = None
            for K in K90 + [1.0]:
                r = rel(cb, l, w, K, s)
                if r is None: prev = None; continue
                if K < 1: fam[w][K].append(r / r1)
                if prev is not None and r > 1.01 * prev: mono.append((l, w, s, K, r / prev))
                prev = r
    out = {w: {str(K): gm(v) for K, v in d.items()} for w, d in fam.items()}
    n = {w: len(d.get(0.5, [])) for w, d in fam.items()}
    print(f"\n#### {label}\n\n| family | n | " + " | ".join(f"K{fmtK(K)}" for K in K90) + (" | old K0.5 | old/new K0.125 |" if old else " |"))
    print("|---|---|" + "---|" * len(K90) + ("---|---|" if old else ""))
    for w in sorted(out):
        o = old["ratios"].get(w) if old else None
        print(f"| {w} | {n[w]} | " + " | ".join(f"{out[w].get(str(K), float('nan')):.3f}" for K in K90)
              + (f" | {o['0.5']:.3f} | {o['0.125'] / out[w]['0.125']:.3f} |" if o and "0.125" in out[w] else " |"))
    print(f"\n#90 rule 1 (rel falls at every 1/8 step K0.125 -> 1 within 1%): "
          + ("**holds** for every tensor and sample" if not mono else f"**{len(mono)} violations**: " +
             ", ".join(f"L{l} {w} s{s} K{fmtK(K)} x{r:.3f}" for l, w, s, K, r in mono[:12])))
    return out


print(f"# w46 re-validation report (settle.py proxy v2), statistic: rel after refit at m = {a.m}; "
      f"{len(G)} gain lines, {len(O)} order lines\n")
print("## 1 · #90-02 i4 vs i8")
item_9002(T10, [0], "#90-02's 10 tensors, sample 0 (the original run's design)")
item_9002(T10, [0, 1], "#90-02's 10 tensors, samples 0 + 1")
item_9002(T10b, [0, 1], "#90-03's 10 new tensors, samples 0 + 1")
top, late, comp = item_9002(T20, [0, 1], "**pooled: 20 tensors, samples 0 + 1 (the verdict, #90-03's pooling rule)**")
item_9002(T20, [0, 1, 2], "20 tensors, samples 0 + 1 + 2 (third sample, when present)")
item_9002(T20, [0, 1], "pooled, best of m in {0.9, 1, 1.1} per job (robustness)", m="best")
item_9002(T10, [0], "exact fork codebooks (int4x vs mul1x_i8), #90-02's 10 tensors, sample 0", cba="int4x", cbb="mul1x_i8")
item_9002(T20, [0, 1], "ladder 1/8 steps addendum: the K grid incl. K1.125-1.875 (same rule, finer K)",
          Kset=set(K9002) | {1.125, 1.25, 1.375, 1.625, 1.75, 1.875})

print("\n## 2 · #100 iqrh vs mul1_i8")
item_100(T20, [0], label="20 tensors, sample 0 (the original run's design)")
item_100(T20, [0, 1], label="**20 tensors, samples 0 + 1 (the verdict)**")
item_100(T20, [0, 1, 2], label="samples 0 + 1 + 2 (when present)")
item_100(T20, [0, 1], label="best of m in {0.9, 1, 1.1} per job (robustness)", m="best")
item_100(T20, [0, 1], cb="iqrh", ref="mul1", label="side: iqrh vs mul1 (f16-path codebook), no rule", rule=False)
item_100(T20, [0, 1], cb="mul1_i8", ref="mul1", label="side: mul1_i8 vs mul1, no rule", rule=False)
item_100(T20, [0, 1], cb="iqrh:0xd0474b:0x2545f491", ref="mul1_i8", label="side: tuned A = 0xd0474b vs mul1_i8, no rule", rule=False)

print("\n## 4 · #106 head-permutation act order")
R106 = [(19, "attn_output"), (43, "attn_output"), (63, "attn_output"), (10, "ssm_out"), (30, "ssm_out"), (50, "ssm_out")]
item_106(R106, [0, 1], "rule tensors, samples 0 + 1")
X = sorted({(k[0], k[1]) for k in O})
item_106(X, [0], "every tensor run, sample 0 (extension, reported)", rule=False)

print("\n## 3 · #90 sub-1 curves rel(K) / rel(K1)")
oldp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/e3/sub1/sub1ratios.json")
old = json.load(open(oldp)) if os.path.exists(oldp) else None
new = item_90("int4grid", [0, 1], "int4grid (the ladder's sub-1 codebook), samples 0 + 1; old = trellis2 i8c (sub1ratios.json)", old)
item_90("cubic0.02_i8", [0], "cubic0.02_i8 (the old run's codebook), sample 0: same codebook, new proxy", old)
if a.sub1json and new:
    json.dump({"ratios": new, "source": "w46 settle.py proxy v2, int4grid, m = 1, samples 0 + 1, geometric mean per family"},
              open(a.sub1json, "w"), indent=1)
    print(f"\nwrote {a.sub1json}")
