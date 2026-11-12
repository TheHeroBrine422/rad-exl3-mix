import json, numpy as np, sys
sys.path.insert(0, "code"); import slicese as S
G = S.load("rows.npz"); v = G["golden-search27"]
tags = sorted(t for t in v["kl"] if t.startswith("w123b2:s27-"))
A, tok, n = S.doc_sums(v, v["kl"]["w123b2:s27-base"])
print("docs", len(A), "tokens", int(tok.sum()), "positions", int(n.sum()), "base KL", A.sum() / n.sum())
U = {}
for t in tags:
    if t == "w123b2:s27-base": continue
    B, _, _ = S.doc_sums(v, v["kl"][t]); lr = np.log(B.sum() / A.sum()); se, sdu = S.se_delta(A, B)
    U[t] = B / B.mean() - A / A.mean()
    g = np.log(B / A)
    print(f"{t:14s} dlog {100*lr:+6.2f}%  se {100*S.se_boot(A, B):.2f}%  sd_u {sdu:.3f}  sd_u/|dlog| {sdu/abs(lr):5.1f}  docs better {np.mean(B < A):.2f}  doc-geo {100*g.mean():+.2f}")
ks = [k for k in U if k not in ("w123b2:s27-all",)]
C = np.corrcoef([U[k] for k in ks])
iu = np.triu_indices(len(ks), 1)
print("cross-move corr of per-doc u (excluding 'all'): mean", C[iu].mean().round(3), "median", np.median(C[iu]).round(3), "range", C[iu].min().round(2), C[iu].max().round(2))
# disjoint layer-group moves only
lay = [k for k in ks if k[11:] in ("L00-07", "L08-15", "L16-23", "L24-31", "L32-39", "L40-47", "L48-55", "L56-63")]
C2 = np.corrcoef([U[k] for k in lay]); iu2 = np.triu_indices(len(lay), 1)
print("disjoint 8-layer groups:", len(lay), "corr mean", C2[iu2].mean().round(3), "range", C2[iu2].min().round(2), C2[iu2].max().round(2))
# additivity: sum of the 8 group dlogs vs 'all'
B, _, _ = S.doc_sums(v, v["kl"]["w123b2:s27-all"])
print("sum of group dlogs", round(100 * sum(np.log(S.doc_sums(v, v['kl'][k])[0].sum() / A.sum()) for k in lay), 2), "vs all", round(100 * np.log(B.sum() / A.sum()), 2))
