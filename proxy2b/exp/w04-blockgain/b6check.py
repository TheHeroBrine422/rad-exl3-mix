#!/usr/bin/env python3
"""b6check.py NAT.npz PERM.npz PERMS.npz DOWNPERM.json K [NAME] -- B6 on real quants: the h-weighted proxy
sum_c h_c e_c / tr(H) of every MLP tensor (stats.py), permuted quants (k<K>bperm[_s1,_s2], stats of the permuted
model) vs natural (k<K>bdiag[_s1,_s2]), mean over seeds, against the prediction (gate / up: blk.block_costs of the
permuted vs natural h sv^2; down: downperm.py's input-side ratio for NAME)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blk
nat, per, pf, dpf, K = sys.argv[1:6]; name = sys.argv[6] if len(sys.argv) > 6 else "balanced"
N, Pq, PZ, DP = np.load(nat), np.load(per), np.load(pf), json.load(open(dpf))
def hp(Z, k, arm):
    t = [x.split("/e/")[1] for x in Z.files if x.startswith(k + "/e/") and x.split("/e/")[1].startswith(f"k{K}{arm}")]
    return np.mean([(Z[f"{k}/h"] * Z[f"{k}/e/{s}"]).sum() for s in t]) / float(Z[f"{k}/trH"]), len(t)
rows = []; tot = {"meas_n": 0.0, "meas_p": 0.0, "pred_p": 0.0}
for l in sorted(int(x.split("/")[0][1:]) for x in PZ.files if x.endswith("/" + name)):
    p = PZ[f"L{l}/{name}"]
    for x in ("gate", "up", "down"):
        k = f"model.language_model.layers.{l}.mlp.{x}_proj"
        (mn, nn), (mp, npp) = hp(N, k, "bdiag"), hp(Pq, k, "bperm")
        if x == "down": pr = DP[str(l)][name]
        else:
            d = np.maximum(N[f"{k}/h"], 1e-30 * N[f"{k}/h"].max()) * N[f"{k}/sv2"]
            pr = blk.block_costs(d[p])[0].sum() / blk.block_costs(d)[0].sum()
        rows.append((l, x, mp / mn, pr, nn, npp)); tot["meas_n"] += mn; tot["meas_p"] += mp; tot["pred_p"] += mn * pr
print(f"K{K} {name}: layer family  measured perm/natural  predicted  (seeds nat / perm)")
for l, x, m, pr, a, b in rows: print(f"  {l:3d} {x:5s} {m:8.4f} {pr:8.4f}   ({a}/{b})")
for x in ("gate", "up", "down"):
    r = [(m, pr) for _, xx, m, pr, *_ in rows if xx == x]
    print(f"  {x}: median measured {np.median([m for m, _ in r]):.4f}, predicted {np.median([p_ for _, p_ in r]):.4f}, "
          f"Spearman {__import__('scipy.stats').stats.spearmanr(*zip(*r)).correlation:.3f}")
print(f"  MLP total (Fisher-scale h proxy): measured {tot['meas_p'] / tot['meas_n']:.4f}, predicted {tot['pred_p'] / tot['meas_n']:.4f}")
