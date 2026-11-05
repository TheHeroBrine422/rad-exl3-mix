#!/usr/bin/env python3
"""rowslab2.py LAYER TENSOR OUT.json -- #96 stage 1: row-slab K ceiling with output-side row importance (CPU).

rowslab.py (E3c) priced a slab of S output rows by its input-side Hessian norm ref_s = sum_{r in s} W_r H_in W_r^T.
Here each row also carries its output-side weight h_r = the block-normalized Σdy² diagonal of the 27B collection
(hout-bn, one file per linear, tensor "hout_diag"; a stacked gate_up is gate rows then up rows, as e3.load_w stacks
them). High-rate model: slab s at b_s bits costs w_s 2^(-2 b_s); the continuous-bit ceiling of moving bits between
slabs at the same total error is 0.5 log2(AM(w)/GM(w)). Weights reported per S in (128, 256, 1024):
  in    w_s = sum_s ref_r                          (E3c's number, no output side)
  out   w_s = sum_s h_r ref_r                      (#96's rule number: README "#96")
  rot   w_s = sum_{128-blocks b in s} (sum_b h)(sum_b ref) / 128
        (EXL3 also rotates the output side in 128-row Hadamard blocks, which spreads a block's error evenly over its
        rows; the same model with that mixing. Equals "out" when h or ref is flat inside each block.)
Also h_cv (coefficient of variation of h over the rows) and gain_h_S128 (slabs priced by Σ h alone).
Env E3_CKPT, E3_HESS, RS_HOUT (the hout-bn folder). The Hessian is unpacked in float32 row by row (~4 GB for
ffn_down instead of e3.load_h's ~11 GB)."""
import json, math, os, sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3
torch.set_grad_enabled(False)

HOUT = os.environ.get("RS_HOUT", "/data/work/hout27/hout-bn")


def load_h32(layer, hn):
    with safe_open(f"{e3.HD}/{e3.P}{layer}.{hn}.safetensors", "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").float()
    H = torch.zeros(k, k, dtype=torch.float32); o = 0
    for i in range(k):                                   # packed upper triangle, row-major
        H[i, i:] = up[o:o + k - i]; o += k - i
    del up
    d = H.diagonal().clone(); H += H.T.clone(); H.diagonal().copy_(d)
    return H


def gain(w):
    w = np.clip(w, 1e-300, None)
    return 0.5 * math.log2(w.mean() / math.exp(np.log(w).mean()))


def main(layer, tensor, out):
    names, hn = e3.ROUTE[tensor]
    W = e3.load_w(layer, names).float(); H = load_h32(layer, hn)
    ref = ((W @ H) * W).sum(1).double().clamp(min=1e-30).numpy()
    N, Kd = W.shape; del W, H
    h = []
    for n in names:
        with safe_open(f"{HOUT}/{e3.P}{layer}.{n}.safetensors", "pt") as f: h.append(f.get_tensor("hout_diag").double())
    h = torch.cat(h).numpy(); assert h.shape[0] == N, (h.shape, N)
    h = np.clip(h, 0, None)
    o = {"layer": layer, "w": tensor, "N": N, "K": Kd, "params": N * Kd, "h_cv": float(h.std() / h.mean())}
    hb, rb = h.reshape(-1, 128).sum(1), ref.reshape(-1, 128).sum(1)
    for S in (128, 256, 1024):
        if N % S: continue
        o[f"in_S{S}"] = gain(ref.reshape(-1, S).sum(1))
        o[f"out_S{S}"] = gain((h * ref).reshape(-1, S).sum(1))
        o[f"rot_S{S}"] = gain((hb * rb / 128).reshape(-1, S // 128).sum(1))
    o["gain_h_S128"] = gain(hb)
    print(json.dumps(o), flush=True)
    with open(out + ".part", "w") as fh: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
