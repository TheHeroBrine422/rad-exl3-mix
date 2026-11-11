#!/usr/bin/env python3
"""stab101.py SNAPS BNDIR CKPT OUT.jsonl -- w101 #130-02 (../w101.md): is the head-level structure of h stable enough
for the P16 interleave? Composable perm.py / perm2.py's high-rate model, factor fitted on one half of the Sigma dy^2
blocks and scored on the other.

SNAPS/hdiag-snapN-dy/<key>.safetensors: the 27B collection's cumulative means (N = 128 .. 1920); differenced into 15
blocks of 128 sequences, each scaled to the same per-tensor total (bn), E = mean of blocks 0, 2, .., 14, O = blocks
1, 3, .., 13, F = all 15. Gate G-b: F / mean(F) == bn1920 / mean(bn1920) (BNDIR) to 1e-5 on every key.
Per stored 128-row block: M_X = S diag(v_X / mean v_X) S^T (S = perm2.py's signed 128-Hadamard, rng 1), v = h r^2,
r^2 = row mean square of the bf16 weight. Factor from M_A + 0.01 I in perm.py's order (Cholesky of the reversed
matrix, unit-lower L', D'); cost c(A -> B) = sum_blocks tr(L'^-1 rev(M_B) L'^-T) (undamped metric B).
Arms: N (natural), P16 (perm2 gperm(R, 16)), B16 (balanced on A, reported only). One JSON line per key; "<arm>:<A>fl"
= 0.01 tr(L'^-1 L'^-T) (add it for the damped metric: <arm>:FF + <arm>:Ffl over 128 nb = perm2.py's pivsum)."""
import glob, json, os, sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perm101 import gperm, balanced  # noqa: E402

SN, BN, CK, OUT = sys.argv[1:5]
torch.set_num_threads(1)
ns = list(range(128, 1921, 128))
keys = sorted(os.path.basename(f)[:-12] for f in glob.glob(f"{SN}/hdiag-snap1920-dy/*.safetensors"))
wmap = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()

Hd = np.array([[1.0]])
while Hd.shape[0] < 128: Hd = np.block([[Hd, Hd], [Hd, -Hd]])
S = Hd / np.sqrt(128) * np.random.default_rng(1).choice([-1.0, 1.0], 128)


def mets(v):
    """per stored block: S diag(v_b / mean v) S^T, reversed (perm.py's order); [nb, 128, 128]"""
    v = v / v.mean(); nb = len(v) // 128
    M = np.einsum("ij,bj,kj->bik", S, v.reshape(nb, 128), S)
    return M[:, ::-1, ::-1].copy()


def factor(Mr):
    """unit-lower L' of Mr + 0.01 I (Cholesky C, L' = C diag(C)^-1); returns L'^-1 [nb, 128, 128]"""
    C = np.linalg.cholesky(Mr + 0.01 * np.eye(128))
    L = C / np.diagonal(C, axis1=1, axis2=2)[:, None, :]
    return np.linalg.inv(L)


def cost(Li, MrB):
    return float(np.einsum("bij,bjk,bik->", Li, MrB, Li))


def floor(Li):
    """0.01 tr(L'^-1 L'^-T): cost(Li, M_B) + floor(Li) = the damped-metric cost (perm.py's pivot sum when B = A)"""
    return 0.01 * float((Li ** 2).sum())


def get(n, k):
    with safe_open(f"{SN}/hdiag-snap{n}-dy/{k}.safetensors", "pt") as f: return f.get_tensor("hout_diag").double().numpy()


def rank(x): r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r


with open(OUT, "a") as fo:
    for k in keys:
        if k in done: continue
        blocks, prev, pn = [], None, 0
        for n in ns:
            cur = get(n, k); b = cur * n - (prev * pn if prev is not None else 0); m = n - pn
            blocks.append(b / max(b.sum(), 1e-300) * m); prev, pn = cur, n
        h = {"E": sum(blocks[0::2]) / 8, "O": sum(blocks[1::2]) / 7, "F": sum(blocks) / 15}
        with safe_open(f"{BN}/{k}.safetensors", "pt") as f: bn = f.get_tensor("hout_diag").double().numpy()
        gate = float(np.abs(h["F"] / h["F"].mean() / (bn / bn.mean()) - 1).max())
        assert all(np.isfinite(x).all() and (x > 0).all() for x in h.values()), k
        with safe_open(f"{CK}/{wmap[k + '.weight']}", "pt") as f: r2 = f.get_tensor(k + ".weight").float().pow(2).mean(1).double().numpy()
        R = len(r2); p16 = gperm(R, 16).numpy(); nat = np.arange(R)
        v = {x: h[x] * r2 for x in h}
        o = {"key": k, "R": R, "gate_rel": gate}
        for A, B in (("E", "O"), ("O", "E"), ("F", "F")):
            perms = {"N": nat, "P16": p16, "B16": balanced(v[A]).numpy()}
            for arm, p in perms.items():
                MA, MB = mets(v[A][p]), mets(v[B][p]); Li = factor(MA)
                o[f"{arm}:{A}{A}"] = cost(Li, MA); o[f"{arm}:{A}{B}"] = cost(Li, MB); o[f"{arm}:{A}fl"] = floor(Li)
        hb = {x: h[x].reshape(-1, 128).mean(1) for x in ("E", "O")}
        o["rho_head"] = float(np.corrcoef(rank(hb["E"]), rank(hb["O"]))[0, 1])
        vb = {x: v[x].reshape(-1, 128).mean(1) for x in ("E", "O")}
        o["rho_head_v"] = float(np.corrcoef(rank(vb["E"]), rank(vb["O"]))[0, 1])
        fo.write(json.dumps(o) + "\n"); fo.flush()
        print(k, f"gate {gate:.1e}", {a: round(o[f'{a}:EO'] / o['N:EO'], 4) for a in ("P16", "B16")},
              {a: round(o[f'{a}:EE'] / o['N:EE'], 4) for a in ("P16", "B16")}, flush=True)
