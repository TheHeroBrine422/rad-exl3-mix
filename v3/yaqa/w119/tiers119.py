#!/usr/bin/env python3
"""tiers119.py HEADSENS.npz CKPT OUT.json -- E3 (a) (RULES.md): pick ONE fixed head tiering before any quant, with w118's
high-rate model of two-sided head LDLQ (hpiv*.py: output Hadamard 128, factor from the qad-gen Fisher x row mean square,
global mean-1, + sigma_out 0.1, block-16 LDL), error scored in the fit-gen metric.

Layout of a tiering (n1, n2, ...): tiers by qad-gen Fisher rank (tier 1 = the top n1 rows), stored tier after tier;
inside a tier the rows are taken in vocab-id order and laid out as runs of 16 ids strided across the tier's blocks (R16
inside the tier). Per tier and K: error = (sum of the tier's block errors) x 2^-2K x i8 penalty (K7 x1.27, #94); a bf16
tier costs 16 bits and has error 0. For each budget (avg bits over the 248,320 rows) the best K per tier over the grid is
found by enumeration; the tiering with the lowest geometric mean of model error over the budgets wins."""
import itertools, json, sys
import numpy as np, torch
from safetensors import safe_open

HS, CK, OUT = sys.argv[1:4]
SIG = 0.1; KG = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 16.0]; PEN = {7.0: 1.27}
BUDGETS = [3.25, 4.0, 5.0, 6.0]
CANDS = [(2048, 16384, 65536), (512, 4096, 32768), (8192, 65536)]
torch.set_num_threads(4)
z = np.load(HS); Fq = torch.from_numpy(z["fisher_qad_gen"].astype(np.float64)); Ff = torch.from_numpy(z["fisher_fit_gen"].astype(np.float64))
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight")
V = W.shape[0]; sv2 = torch.zeros(V, dtype=torch.float64)
for i in range(0, V, 16384): sv2[i:i + 16384] = W[i:i + 16384].double().pow(2).mean(1)
del W
H = torch.tensor([[1.0]], dtype=torch.float64)
while H.shape[0] < 128: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H /= 128 ** 0.5


def blk_ldl(M):
    L = torch.linalg.cholesky(M); g = M.shape[0]; b = 16; m = 128 // b
    Lb = L.view(g, m, b, m, b); Dd = torch.stack([Lb[:, j, :, j, :] for j in range(m)], 1); Dinv = torch.linalg.inv(Dd)
    A = L.clone().view(g, m, b, m, b)
    for j in range(m): A[:, :, :, j, :] = A[:, :, :, j, :] @ Dinv[:, j][:, None]
    return A.view(g, 128, 128)


def block_err(order):
    """two-sided model error per 128-row block (unit rounding noise), fit-gen metric; one-sided for reference"""
    r = order.view(-1, 128)
    Mq = H @ torch.diag_embed(Fq[r] * sv2[r]) @ H
    Mq = Mq / Mq.diagonal(dim1=1, dim2=2).mean() + SIG * torch.eye(128, dtype=torch.float64)
    Ai = torch.linalg.inv(blk_ldl(Mq)); Me = H @ torch.diag_embed(Ff[r] * sv2[r]) @ H
    return (Ai @ Me @ Ai.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(1), Me.diagonal(dim1=1, dim2=2).sum(1)


def r16(ids):
    """ids (a tier, any order) -> stored order: vocab-id order, 16-id runs strided across the tier's blocks"""
    s = torch.sort(ids).values; n = len(s); assert n % 128 == 0
    nrun = n // 16; return s.view(8, nrun // 8, 16).transpose(0, 1).reshape(-1)


rank = torch.argsort(-Fq)
def c(k): return 0.0 if k == 16.0 else 2.0 ** (-2 * k) * PEN.get(k, 1.0)
_, one = block_err(rank); base1 = float(one.sum())          # one-sided, any order (reference unit: uniform one-sided = 1)
res = {"model": "w118 hpiv, sigma_out 0.1, fit-gen metric", "unit": "error / one-sided uniform at the same K-scale (2^-2K)",
       "grid": KG, "budgets": BUDGETS, "cands": {}}
u = block_err(r16(torch.arange(V)))[0]; res["U_R16_uniform_rel"] = float(u.sum()) / base1
for cand in CANDS:
    bnd = (0,) + cand + (V,); E, n = [], []
    for a, b in zip(bnd, bnd[1:]):
        e, _ = block_err(r16(rank[a:b])); E.append(float(e.sum())); n.append(b - a)
    best = {}
    for B in BUDGETS:
        top = None
        for ks in itertools.product(KG, repeat=len(n)):
            bits = sum(k * m for k, m in zip(ks, n)) / V
            if bits > B + 1e-9: continue
            err = sum(e * c(k) for e, k in zip(E, ks))
            if top is None or err < top[0]: top = (err, ks, bits)
        uni = float(u.sum()) * c(B) if B in KG else float(u.sum()) * 2.0 ** (-2 * B)
        best[str(B)] = {"err": top[0] / base1, "Ks": top[1], "bits": top[2], "vs_U_same_bits": top[0] / uni}
    gm = float(np.exp(np.mean([np.log(v["err"]) for v in best.values()])))
    res["cands"][str(cand)] = {"tier_err_rel": [e / base1 for e in E], "rows": n, "best": best, "geo_err": gm}
    print(cand, f"geo {gm:.4e}", {B: (v["Ks"], round(v["bits"], 3), round(v["vs_U_same_bits"], 3)) for B, v in best.items()}, flush=True)
win = min(res["cands"], key=lambda k: res["cands"][k]["geo_err"]); res["choice"] = win
print("choice", win)
json.dump(res, open(OUT, "w"), indent=1)
