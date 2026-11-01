#!/usr/bin/env python3
"""hhalf.py DIR -- split-half agreement of collect.py's full-token H_out: even sequences (DIR/hout-even-snapN) vs odd
(total - even), at every snapshot N. Per linear: relative Frobenius difference of the trace-normalised halves, diagonal
correlation, and the overlap of the top-16 eigenvector subspaces (1 = same). Halves that disagree a lot = undersampled."""
import glob, os, re, sys, torch
from safetensors import safe_open
D = sys.argv[1]
def load(p):
    with safe_open(p, "pt") as f: return f.get_tensor("hout").float()
def top(M, k=16, it=30):   # top-k eigenvectors by subspace iteration (no full eigh of a 17408² matrix)
    Q = torch.linalg.qr(torch.randn(M.shape[0], k, generator=torch.Generator().manual_seed(0)))[0]
    for _ in range(it): Q = torch.linalg.qr(M @ Q)[0]
    return Q
for ed in sorted(glob.glob(f"{D}/hout-even-snap*"), key=lambda x: int(re.findall(r"\d+$", x)[0])):
    n = re.findall(r"\d+$", ed)[0]; td = f"{D}/hout-snap{n}"; fro, dg, ov = [], [], []
    for p in sorted(glob.glob(f"{ed}/*.safetensors")):
        T = load(f"{td}/{os.path.basename(p)}"); E = load(p); O = T - E
        E, O = E / E.trace(), O / O.trace()
        fro.append(float((E - O).norm() / ((E + O) / 2).norm()))
        dg.append(float(torch.corrcoef(torch.stack([E.diagonal(), O.diagonal()]))[0, 1]))
        ue, uo = top(E), top(O)
        ov.append(float((ue.T @ uo).pow(2).sum() / 16))
    m = lambda v: sum(v) / len(v)
    print(f"split-half at {n} seqs ({int(n) // 2} per half): rel diff mean {m(fro):.3f} max {max(fro):.3f}; "
          f"diag corr mean {m(dg):.3f} min {min(dg):.3f}; top-16 subspace overlap mean {m(ov):.3f} min {min(ov):.3f}")
