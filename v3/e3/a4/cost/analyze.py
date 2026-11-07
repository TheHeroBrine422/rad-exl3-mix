#!/usr/bin/env python3
"""analyze.py DIR -- #46-03 step 1 readout (rules: v3/e3/a4/README.md amendment 1) from DIR/results.jsonl (cmp.py lines,
one per arm vs its level's all-a8g128 baseline). Writes DIR/table.jsonl (one line per family x band x K x group: the
paired dKLD, SE, ratio, p99 / top-1 of both) and prints the tables, the additivity check (gate_up + down, all bands, vs
the sum of its eight parts: pass within +-25%), the K-interaction check (dK1 / dK3, dK2 / dK3 per arm; "additive and
K-independent" if each dKLD-weighted geometric mean is within +-25% of 1 and no family with d > 2 SE differs by more
than x1.5) and one 'A4COST VERDICT' line."""
import collections, json, math, os, re, sys

D = sys.argv[1]
FAMS = ["gate_up", "down", "attn_in", "gdn_in", "attn_out", "ssm_out"]
BANDS = ["0-15", "16-31", "32-47", "48-63"]
R = {}
for l in open(f"{D}/results.jsonl"):
    if l.strip().startswith("{"):
        r = json.loads(l); R[r["tag"]] = r          # a rerun's line wins
cell = {}
for t, r in R.items():
    m = re.fullmatch(r"K(\d)-(\w+)-(\d+-\d+)-g(\d+)", t)
    if m: cell[(m[2], m[3], int(m[1]), int(m[4]))] = r
with open(f"{D}/table.jsonl.part", "w") as f:
    for (fa, bd, K, g), r in sorted(cell.items(), key=lambda kv: (kv[0][3], kv[0][2], FAMS.index(kv[0][0]), kv[0][1])):
        f.write(json.dumps(dict(family=fa, band=bd, K=K, group=g, diff=r["diff"], se=r["se"], z=r["z"], ratio=r["ratio"],
                                kld_a8=r["kld_a"], kld_a4=r["kld_b"], p99_a8=r["p99_a"], p99_a4=r["p99_b"],
                                top1_a8=r["top1_a"], top1_a4=r["top1_b"])) + "\n")
os.replace(f"{D}/table.jsonl.part", f"{D}/table.jsonl")
Ks = sorted({k[2] for k in cell}); Gs = sorted({k[3] for k in cell})
print("# #46-03 A4 cost per family x band (paired dKLD vs the same level all-a8g128, golden-search27)\n")
for t in ("K2-f16", "v2-300"):
    if t in R: print(f"control {t}: KLD {R[t]['kld_b']:.5f} vs {R[t]['base']} {R[t]['kld_a']:.5f} (ratio {R[t]['ratio']:.4f}, z {R[t]['z']:+.1f})")
for K in Ks:
    b = next((r for k, r in cell.items() if k[2] == K), None)
    if b: print(f"K{K} a8g128 baseline: KLD {b['kld_a']:.5f}, p99 {b['p99_a']:.4f}, top-1 {b['top1_a']:.4f}")
verdict = []
for g in Gs:
    print(f"\n## a4g{g}: dKLD x1e4 (z) -- ratio\n")
    print("| family | band | " + " | ".join(f"K{K}" for K in Ks) + " |"); print("|---|---|" + "---|" * len(Ks))
    for fa in FAMS:
        for bd in BANDS:
            xs = [cell.get((fa, bd, K, g)) for K in Ks]
            if not any(xs): continue
            print(f"| {fa} | {bd} | " + " | ".join(f"{1e4 * x['diff']:+.2f} ({x['z']:+.1f}) x{x['ratio']:.3f}" if x else "-" for x in xs) + " |")
    print("\nper family, all bands (sum of dKLD x1e4):")
    for fa in FAMS:
        s = {K: [cell[(fa, bd, K, g)] for bd in BANDS if (fa, bd, K, g) in cell] for K in Ks}
        print(f"  {fa:9s} " + "  ".join(f"K{K} {1e4 * sum(x['diff'] for x in v):+7.2f} ({len(v)}/4)" for K, v in s.items()))
    # additivity: gate_up + down, all bands
    for K in Ks:
        c = R.get(f"K{K}-gu+dn-all-g{g}")
        parts = [cell.get((fa, bd, K, g)) for fa in ("gate_up", "down") for bd in BANDS]
        if c and all(parts):
            s = sum(p["diff"] for p in parts); se = math.sqrt(sum(p["se"] ** 2 for p in parts))
            q = c["diff"] / s if s > 0 else float("nan"); ok = abs(q - 1) <= 0.25
            print(f"\nadditivity K{K} g{g}: combined {1e4 * c['diff']:+.2f} (se {1e4 * c['se']:.2f}) vs sum of 8 parts "
                  f"{1e4 * s:+.2f} (se {1e4 * se:.2f}): x{q:.3f} -> {'PASS' if ok else 'FAIL (interaction factor ' + format(q, '.3f') + ')'}")
            verdict.append(f"add K{K} g{g} x{q:.2f} {'pass' if ok else 'FAIL'}")
    # K-interaction vs K3
    if 3 in Ks:
        for K in [k for k in Ks if k != 3]:
            num = den = 0.0; bad = []
            for fa in FAMS:
                for bd in BANDS:
                    x, y = cell.get((fa, bd, K, g)), cell.get((fa, bd, 3, g))
                    if not (x and y) or x["diff"] <= 0 or y["diff"] <= 0: continue
                    w = y["diff"]; num += w * math.log(x["diff"] / y["diff"]); den += w
                    if (x["z"] > 2 or y["z"] > 2) and not (1 / 1.5 <= x["diff"] / y["diff"] <= 1.5): bad.append(f"{fa}/{bd} x{x['diff'] / y['diff']:.2f}")
            if den > 0:
                gm = math.exp(num / den); ok = abs(gm - 1) <= 0.25 and not bad
                print(f"\nK-interaction g{g}: dK{K} / dK3 weighted geo mean x{gm:.3f}; arms off by > x1.5 at z > 2: {bad or 'none'} "
                      f"-> {'K-independent' if ok else 'K-DEPENDENT'}")
                verdict.append(f"dK{K}/dK3 g{g} x{gm:.2f} {'indep' if ok else 'DEP'}")
n = len(cell); tot = len(FAMS) * len(BANDS) * len(Ks) * len(Gs)
for g in Gs:
    for K in Ks:
        s = sum(cell[(fa, bd, K, g)]["diff"] for fa in FAMS for bd in BANDS if (fa, bd, K, g) in cell)
        verdict.append(f"K{K} g{g} sum {1e4 * s:+.1f}e-4")
print(f"\nA4COST VERDICT: {n} of {tot} cells; " + "; ".join(verdict))
