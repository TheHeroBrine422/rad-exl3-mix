#!/usr/bin/env python3
"""e3.py LAYER TENSOR METHODS BITS [OUT.jsonl] -- E3: quality per bit of integer-grid weight codes (CPU).

Relative Hessian proxy error of one Qwen3.8-27B linear quantized several ways, in hproxy's metric
(err = tr(dW H dW^T), ref = tr(W H W^T); rel = err / ref), so it lines up with our EXL3 ours-K rows
(hproxy-ours.jsonl). H = the challenge's calib input Hessian (sum of x x^T over all of calib).

TENSOR: hproxy's names: ssm_qkv ssm_z ssm_out attn_qg attn_k attn_v attn_output ffn_gate_up ffn_down.
METHODS (comma list); every method is per-output-channel scaled unless it says g128:
  rtn         round to nearest, no rotation
  rtn_rot     round to nearest after the input rotation (128-block Hadamard x random signs along K)
  gptq_rot    GPTQ/LDLQ on the rotated weight (1% damping, columns in natural order)
  gptq_rot2   as gptq_rot, plus a 128-block output rotation along N (EXL3-style, undone in the epilogue)
  gptq_g128   gptq_rot with an exact fp16 scale per 128 inputs (+0.125 bpw; not free in an int GEMM)
  gptq_g128i8 gptq_g128 whose decoded values are re-rounded to an int8 grid with one scale per row: what an int8
              WMMA would actually multiply (the group scales baked into the 8-bit values)
  lloyd_rot   GPTQ onto a 2^b-level Lloyd-Max codebook (one per tensor, on the scale-normalised rotated weights),
              decoded to int8 per row (LUT -> int8)
Grids: 'uni' levels are midrise (odd integers: 2-bit {-3,-1,1,3}) except b = 4 and 8, which are midtread (-8..7,
-128..127) so that the 4-bit codes are int4 and the 8-bit codes are int8 as stored (iu4 / iu8 operands).
One JSON line per (method, bits): rel, bpw (incl. scales), seconds.
"""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

CK = os.environ.get("E3_CKPT", "/workspace/models/qwen38-27b-bf16")
HD = os.environ.get("E3_HESS", "/workspace/e3data/challenge/hessians/calib")
P = "model.language_model.layers."
torch.set_grad_enabled(False)
OUT = None
ROWS = int(os.environ.get("E3_ROWS", "1024"))   # 0: every row

ROUTE = {
    "ssm_qkv": (["linear_attn.in_proj_qkv"], "input"),
    "ssm_z": (["linear_attn.in_proj_z"], "input"),
    "ssm_out": (["linear_attn.out_proj"], "linear_attn.out_proj.input"),
    "attn_qg": (["self_attn.q_proj"], "input"),
    "attn_k": (["self_attn.k_proj"], "input"),
    "attn_v": (["self_attn.v_proj"], "input"),
    "attn_output": (["self_attn.o_proj"], "self_attn.o_proj.input"),
    "ffn_gate_up": (["mlp.gate_proj", "mlp.up_proj"], "post_attention.input"),
    "ffn_down": (["mlp.down_proj"], "mlp.down_proj.input"),
}


def load_w(layer, names):
    idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    out = []
    for n in names:
        k = f"{P}{layer}.{n}.weight"
        with safe_open(f"{CK}/{idx[k]}", "pt") as f:
            out.append(f.get_tensor(k).float())
    return torch.cat(out, 0).double()


def load_h(layer, hn):
    with safe_open(f"{HD}/{P}{layer}.{hn}.safetensors", "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64)
    iu = torch.triu_indices(k, k)
    H[iu[0], iu[1]] = up
    H = H + H.T - torch.diag(H.diagonal())
    return H


def had_blocks(n, seed):
    """128-block Hadamard x random signs as a function on the last dim (orthogonal)."""
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h /= math.sqrt(128)
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).double()
    def f(x):   # x [..., n] -> rotated
        sh = x.shape
        return ((x * s).reshape(*sh[:-1], n // 128, 128) @ h.T).reshape(sh)
    return f


def grid(b):
    if b in (4, 8): return torch.arange(-(2 ** (b - 1)), 2 ** (b - 1), dtype=torch.float64), 2 ** (b - 1)
    lv = torch.arange(-(2 ** b - 1), 2 ** b, 2, dtype=torch.float64)
    return lv, 2 ** b - 1


def nearest(x, lv):
    """x [...], sorted levels lv -> nearest level values."""
    mid = (lv[1:] + lv[:-1]) / 2
    return lv[torch.bucketize(x, mid)]


def clip_scales(W, d, lv, amax, steps=12):
    """per-row scale = c * max|w| / amax with c searched to minimise the diag(H)-weighted RTN error"""
    m = W.abs().amax(1, keepdim=True).clamp(min=1e-12)
    best, bs = None, None
    for c in torch.linspace(0.55, 1.0, steps):
        s = c * m / amax
        e = ((nearest(W / s, lv) * s - W) ** 2 * d).sum(1)
        if best is None: best, bs = e, s.clone()
        else:
            t = e < best; best = torch.where(t, e, best); bs = torch.where(t[:, None], s, bs)
    return bs


def gptq(W, H, qfn, block=128, damp=0.01):
    """GPTQ (LDLQ) in natural column order; qfn(col_index_start, block_of_columns [N, b]) -> per column via callback
    qfn(j, w_col) -> quantized column. Returns Q."""
    W = W.clone(); N, K = W.shape
    H = H.clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    for a in range(0, K, block):
        b = min(a + block, K)
        W1 = W[:, a:b].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); U1 = U[a:b, a:b]
        for i in range(b - a):
            w = W1[:, i]; q = qfn(a + i, w); Q1[:, i] = q
            e = (w - q) / U1[i, i]
            W1[:, i:] -= e[:, None] * U1[i, i:][None, :]
            E1[:, i] = e
        Q[:, a:b] = Q1
        W[:, b:] -= E1 @ U[a:b, b:]
    return Q


def rel(W, Q, H):
    D = W - Q
    return float(torch.einsum("ij,jk,ik->", D, H, D) / torch.einsum("ij,jk,ik->", W, H, W))


def lloyd(x, n, iters=40):
    x = x.flatten()
    if x.numel() > 2_000_000: x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    x, _ = torch.sort(x)
    c = torch.quantile(x, (torch.arange(n, dtype=torch.float64) + 0.5) / n)
    for _ in range(iters):
        idx = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        s = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, x)
        k = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, torch.ones_like(x))
        c = torch.where(k > 0, s / k.clamp(min=1), c)
    return torch.sort(c)[0]


def run(layer, tensor, methods, bits):
    names, hn = ROUTE[tensor]
    W0 = load_w(layer, names); H0 = load_h(layer, hn)
    if ROWS and ROWS < W0.shape[0]:   # rows are independent given H: a random subset estimates rel without bias
        idx = torch.randperm(W0.shape[0], generator=torch.Generator().manual_seed(7 + layer))[:ROWS].sort()[0]
        W0 = W0[idx]
    N, K = W0.shape
    rk = had_blocks(K, 1000 + layer); rn = had_blocks(N, 2000 + layer)
    Wr = rk(W0)                                  # W R^T along K
    Hr = rk(rk(H0).T).T                          # R H R^T
    Wr2 = rn(Wr.T).T                             # plus the output rotation along N (error is invariant to it)
    d0, dr = H0.diagonal()[None, :], Hr.diagonal()[None, :]
    out = []
    for b in bits:
        lv, amax = grid(b)
        for m in methods:
            t0 = time.time(); bpw = b + 16 / K
            if m == "rtn":
                s = clip_scales(W0, d0, lv, amax); r = rel(W0, nearest(W0 / s, lv) * s, H0)
            elif m == "rtn_rot":
                s = clip_scales(Wr, dr, lv, amax); r = rel(Wr, nearest(Wr / s, lv) * s, Hr)
            elif m in ("gptq_rot", "gptq_rot2"):
                Wx = Wr if m == "gptq_rot" else Wr2
                s = clip_scales(Wx, dr, lv, amax)[:, 0]
                Q = gptq(Wx, Hr, lambda j, w: nearest(w / s, lv) * s); r = rel(Wx, Q, Hr)
            elif m in ("gptq_g128", "gptq_g128i8"):
                G = 128; S = torch.zeros(N, K // G, dtype=torch.float64)
                for g in range(K // G):
                    S[:, g] = clip_scales(Wr[:, g * G:(g + 1) * G], dr[:, g * G:(g + 1) * G], lv, amax)[:, 0]
                S = S.half().double()   # stored as fp16
                Q = gptq(Wr, Hr, lambda j, w: nearest(w / S[:, j // G], lv) * S[:, j // G])
                if m == "gptq_g128i8":
                    s8 = Q.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
                    Q = torch.round(Q / s8).clamp(-127, 127) * s8
                r = rel(Wr, Q, Hr); bpw = b + 16 / G + 16 / K
            elif m == "lloyd_rot":
                s = Wr.abs().amax(1, keepdim=True).clamp(min=1e-12)
                cb = lloyd(Wr / s, 2 ** b)
                cb8 = torch.round(cb / cb.abs().max() * 127) / 127 * cb.abs().max()   # LUT -> int8 per row scale
                # rescale per row: search a scale multiplier like clip_scales, with the codebook as the grid
                ss = clip_scales(Wr / s, dr, cb8, 1.0) * s
                Q = gptq(Wr, Hr, lambda j, w: nearest(w / ss[:, 0], cb8) * ss[:, 0]); r = rel(Wr, Q, Hr)
                bpw = b + 16 / K + 2 ** b * 8 / (N * K)
            else:
                raise SystemExit(f"unknown method {m}")
            o = {"layer": layer, "w": tensor, "method": m, "bits": b, "bpw": round(bpw, 4), "rel": r,
                 "N": N, "K": K, "rows_sampled": ROWS, "sec": round(time.time() - t0, 1)}
            print(json.dumps(o), flush=True); out.append(o)
            if OUT:
                with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")
    return out


if __name__ == "__main__":
    layer, tensor = int(sys.argv[1]), sys.argv[2]
    methods = sys.argv[3].split(","); bits = [int(x) for x in sys.argv[4].split(",")]
    OUT = sys.argv[5] if len(sys.argv) > 5 else None
    run(layer, tensor, methods, bits)
