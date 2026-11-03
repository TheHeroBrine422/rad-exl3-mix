#!/usr/bin/env python3
"""f1check.py ROWERR2.jsonl -- F1's CPU check (ranked.md): on the 0.8B, does the one-sided proxy knap4 prices measured
levels by (KN_LEVELS: proxy_err) move with KL when the quants gain an H_out? Per K, per tensor: e1 and e2 ratio of
bdiag (base settings + 8K diagonal H_out) to base, geometric mean over rotation seeds 0 / 1 / 2 (paired by seed), and
of bident (H_out = I, seed 0) and edy (Σdy² diagonal, #33-18) when present; against the whole-model KL ratios of #33-16 (LOG.md). Rule (ranked.md F1): if the
one-sided ratio is > 1.05 on tensors where KL fell, the fit must switch to an H-aware price."""
import collections, json, sys
import numpy as np
R = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    r = json.loads(l); R[r["arm"][2:]][r["key"]] = r
KL = {"k3base": (0.0989, 0.0994, 0.0974), "k3bdiag": (0.0910, 0.0906, 0.0902), "k2base": (0.5354, 0.5411, 0.5225),
      "k2bdiag": (0.4928, 0.5107, 0.5093), "k3bident": (0.1037,), "k2bident": (0.5466,),     # LOG #33-16
      "k3edy": (0.09009, 0.08984, 0.08986), "k2edy": (0.4984, 0.5046, 0.4958)}                  # result-fact7 (#33-18)
for K in (3, 2):
    kb = np.mean(KL[f"k{K}base"])
    for arm in [x for x in ("bdiag", "edy", "bident") if f"k{K}{x}" in R]:
        sfx = [""] if arm == "bident" else ["", "_s1", "_s2"]
        kl = np.mean(KL[f"k{K}{arm}"]) / (kb if arm != "bident" else KL[f"k{K}base"][0])
        keys = sorted(R[f"k{K}base"])
        print(f"K{K} {arm} ({len(sfx)} seed pair{'s' if len(sfx) > 1 else ''}): whole-model KL ratio {kl:.3f}")
        for m, lab in (("e1", "one-sided"), ("e2", "two-sided diag")):
            lr = np.mean([[np.log(R[f"k{K}{arm}{s}"][k][m] / R[f"k{K}base{s}"][k][m]) for k in keys] for s in sfx], 0)
            by = collections.defaultdict(list)
            for k, v in zip(keys, lr): by[k.split(".", 4)[4].split(".")[-1]].append(v)
            print(f"   {lab:15}: geomean {np.exp(lr.mean()):.3f}  median {np.exp(np.median(lr)):.3f}  "
                  f"tensors > 1.05: {np.mean(lr > np.log(1.05)):.0%}  < 0.95: {np.mean(lr < np.log(0.95)):.0%}")
            print("      by type: " + "  ".join(f"{t} {np.exp(np.mean(v)):.3f}" for t, v in sorted(by.items())))
        if arm != "bident":
            e2 = np.exp(np.mean([[np.log(R[f"k{K}{arm}{s}"][k]["e2"] / R[f"k{K}base{s}"][k]["e2"]) for k in keys] for s in sfx]))
            print(f"   family factor KN_HF that maps the two-sided geomean onto the KL ratio: {kl / e2:.3f}")
