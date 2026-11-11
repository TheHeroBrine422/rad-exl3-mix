#!/usr/bin/env python3
"""model101x.py BNDIR CKPT HESS OUT.jsonl [LAYERS] -- w101 #130, lead note 11-09 9:35 PM CST: the composable model's
predicted P16 gain for more families (model numbers only, not a rule decision).

Output side (stab101.py's model, full bn1920 h, in-sample, damped metric = perm.py's pivot sum): attn k_proj / v_proj,
mlp gate_proj / up_proj (and qkv / z / q_proj again for reference). Ratio = c_P16 / c_N per tensor.
MLP: permuting the intermediate dimension by one pi (gate / up rows, down columns) is an exactly equivalent model;
down's columns then share different 128-wide input Hadamard blocks. Input-side model for down (LAYERS only): published
calib H_in + 0.005 mean(diag) I, columns permuted by pi, signed 128-Hadamard per input block (rng 2), high-rate LDLQ
cost = sum of the LDL pivots (Cholesky of the reversed matrix); ratio = pivsum_P16 / pivsum_N."""
import glob, json, os, sys

import numpy as np
import scipy.linalg
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perm101 import gperm  # noqa: E402

BN, CK, HS, OUT = sys.argv[1:5]
LAYERS = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
torch.set_num_threads(1)
wmap = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
FAMS = ("self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "self_attn.q_proj")
keys = sorted(os.path.basename(f)[:-12] for f in glob.glob(f"{BN}/*.safetensors"))
keys = [k for k in keys if k.split(".", 4)[-1] in FAMS]

Hd = np.array([[1.0]])
while Hd.shape[0] < 128: Hd = np.block([[Hd, Hd], [Hd, -Hd]])
S = Hd / np.sqrt(128) * np.random.default_rng(1).choice([-1.0, 1.0], 128)   # stab101.py's S


def mets(v):
    """stab101.py verbatim: per stored block S diag(v_b / mean v) S^T, reversed; [nb, 128, 128]"""
    v = v / v.mean(); nb = len(v) // 128
    M = np.einsum("ij,bj,kj->bik", S, v.reshape(nb, 128), S)
    return M[:, ::-1, ::-1].copy()


def factor(Mr):
    C = np.linalg.cholesky(Mr + 0.01 * np.eye(128))
    return np.linalg.inv(C / np.diagonal(C, axis1=1, axis2=2)[:, None, :])


def pivsum(Li, Mr):
    """damped-metric cost = stab101 cost + floor = perm.py's pivot sum"""
    return float(np.einsum("bij,bjk,bik->", Li, Mr, Li)) + 0.01 * float((Li ** 2).sum())


def wr(o):
    with open(OUT, "a") as fo: fo.write(json.dumps(o) + "\n")
    print(json.dumps(o), flush=True)


for k in keys:
    if k in done: continue
    with safe_open(f"{BN}/{k}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double().numpy()
    with safe_open(f"{CK}/{wmap[k + '.weight']}", "pt") as f: r2 = f.get_tensor(k + ".weight").float().pow(2).mean(1).double().numpy()
    v = h * r2; R = len(v); o = {"key": k, "R": R}
    for arm, p in (("N", np.arange(R)), ("P16", gperm(R, 16).numpy())):
        Mr = mets(v[p]); o[arm] = pivsum(factor(Mr), Mr)
    o["ratio"] = o["P16"] / o["N"]; wr(o)

S2 = Hd / np.sqrt(128) * np.random.default_rng(2).choice([-1.0, 1.0], 128)


def load_h(path):
    """published packed upper triangle -> full symmetric float64 (row-wise fill, no index arrays)"""
    with safe_open(path, "pt") as f:
        n = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").numpy()
    H = np.empty((n, n)); o = 0
    for i in range(n): H[i, i:] = up[o:o + n - i]; H[i:, i] = up[o:o + n - i]; o += n - i
    return H


def rows_t(X, p):
    """T applied to the rows of X: rows permuted by p, then the signed Hadamard per 128-row block (in place)"""
    Y = X[p]
    for b in range(0, Y.shape[0], 128): Y[b:b + 128] = S2 @ Y[b:b + 128]
    return Y


for L in LAYERS:
    k = f"model.language_model.layers.{L}.mlp.down_proj"
    if k + ":in" in done: continue
    o = {"key": k + ":in"}
    for arm in ("N", "P16"):
        H = load_h(f"{HS}/{k}.input.safetensors"); n = H.shape[0]; o["n"] = n
        H[np.diag_indices(n)] += 0.005 * np.diag(H).mean()
        p = gperm(n, 16).numpy() if arm == "P16" else np.arange(n)
        Y = rows_t(H, p); del H                       # T H
        Z = rows_t(Y.T, p); del Y                         # T (T H)^T = T H T^T (H symmetric)
        Z = np.ascontiguousarray(Z[::-1, ::-1])       # perm.py's order: Cholesky of the reversed matrix
        C = scipy.linalg.cholesky(Z, lower=True, overwrite_a=True, check_finite=False)
        o[arm] = float((np.diag(C) ** 2).sum()); del Z, C
    o["ratio"] = o["P16"] / o["N"]; wr(o)
