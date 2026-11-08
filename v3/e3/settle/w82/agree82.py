#!/usr/bin/env python3
"""agree82.py BTDIR [ORDER.json] -- #117 1(a) readout (../w82.md): Spearman agreement of the readers' 128-block trace
rankings with the leave-own-layer-out consensus, the rule's three conditions, pairwise agreement within / across
families and depth bands, spreads; writes the consensus order (all 64 layers, stored heaviest LAST) to ORDER.json."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1]
rows = []                                                    # (layer, tensor, params, w[40], raw[40])
for f in sorted(glob.glob(f"{D}/*.json"), key=lambda p: int(os.path.basename(p)[:-5])):
    d = json.load(open(f))
    for t, v in d["t"].items(): rows.append((d["layer"], t, v["N"] * v["Kd"], np.array(v["w"]), np.array(v["raw"])))
L = np.array([r[0] for r in rows]); T = np.array([r[1] for r in rows]); P = np.array([r[2] for r in rows], float)
W = np.log(np.stack([r[3] for r in rows])); R = np.log(np.stack([r[4] for r in rows]))
FAM = ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ffn_gate", "ffn_up"]
sp = lambda a, b: spearmanr(a, b)[0]
p9 = lambda x: float(np.percentile(np.exp(x), 90) / np.percentile(np.exp(x), 10))


def cons(X, mask): return (P[mask, None] * X[mask]).sum(0) / P[mask].sum()


def report(X, name):
    rho = np.array([sp(X[i], cons(X, L != L[i])) for i in range(len(rows))])
    med, frac = float(np.median(rho)), float((rho >= 0.5).mean())
    fm = {f: float(np.median(rho[T == f])) for f in FAM}
    ok = med >= 0.70 and frac >= 0.80 and min(fm.values()) >= 0.50
    print(f"## {name}: {len(rows)} tensors; median rho_t {med:.3f} (rule >= 0.70), share rho_t >= 0.5 {frac:.3f} "
          f"(>= 0.80), family medians min {min(fm.values()):.3f} (>= 0.50) -> {'AGREE' if ok else 'DO NOT AGREE'}")
    print("family | n | median rho_t | p10 rho_t | median spread p90/p10 | max spread")
    for f in FAM:
        m = T == f; sps = [p9(X[i]) for i in np.where(m)[0]]
        print(f"{f} | {m.sum()} | {fm[f]:.3f} | {np.percentile(rho[m], 10):.3f} | {np.median(sps):.2f} | {max(sps):.2f}")
    print("depth band (layers) | median rho_t | median spread")
    for a in range(0, 64, 16):
        m = (L >= a) & (L < a + 16)
        print(f"{a}-{a + 15} | {np.median(rho[m]):.3f} | {np.median([p9(X[i]) for i in np.where(m)[0]]):.2f}")
    # pairwise: family x family median Spearman over tensor pairs from different layers
    print("pairwise median rho (tensor pairs from different layers), rows / cols: " + " ".join(FAM))
    C = np.array([[sp(X[i], X[j]) for j in range(len(rows))] for i in range(len(rows))])
    for f in FAM:
        line = []
        for g in FAM:
            m = (T[:, None] == f) & (T[None, :] == g) & (L[:, None] != L[None, :])
            line.append(f"{np.median(C[m]):.2f}")
        print(f"{f} | " + " | ".join(line))
    print("lowest rho_t:", ", ".join(f"{L[i]}:{T[i]} {rho[i]:.2f}" for i in np.argsort(rho)[:10]))
    return rho, ok


rw, okw = report(W, "w (LDLQ metric trace, deciding)")
print()
rr, _ = report(R, "raw H_in trace (reported)")
c = cons(W, np.ones(len(rows), bool))
order = [int(b) for b in np.argsort(c)]                     # stored order: ascending consensus, heaviest last
share = np.exp(c) / np.exp(c).sum()
print(f"\nconsensus (all layers), heaviest first: {order[::-1]}")
print(f"consensus block share p90/p10 {np.percentile(share, 90) / np.percentile(share, 10):.2f}, max/min {share.max() / share.min():.2f}")
if len(sys.argv) > 2:
    json.dump({"stored_order": order, "score": c.tolist(), "agree": bool(okw),
               "median_rho": float(np.median(rw))}, open(sys.argv[2], "w"))
