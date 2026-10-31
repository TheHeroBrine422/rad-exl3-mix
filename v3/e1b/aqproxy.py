#!/usr/bin/env python3
"""aqproxy.py DUMPDIR MAP IDX[,IDX...] OUT.jsonl -- per-tensor activation-quantization cost proxy (CPU).

For each dumped linear (the E1b hook's EXL3_AQ_DUMP: raw GEMM inputs x, its suh scales), every config in CONFIGS:
  xr = f16(H128(f16(x * suh)) / sqrt(128))       the kernel's rotation (Sylvester 128-point Hadamard, per block)
  q  = the hook's quantizer on xr                symmetric int b, scale = clip * amax / (2^(b-1) - 1) per row or per group
  e  = unrotate(xr - q)                          back to the input space (exact inverse of the rotation)
  rel_out = sum ||W e||^2 / sum ||W x||^2         output-error energy relative to the output (W: the bf16 weight; rows
                                                 subsampled to <= 4096, an unbiased estimate of the ratio)
  rel_in  = sum ||e||^2 / sum ||x||^2
A stacked gate_up quantizes each half's rotation separately (suh for gate_proj, suh1 for up_proj) and sums both.
One JSON line per (tensor, config). Calibrate against the E1b family KLDs (proxy sums -> KLD increments).
MAP: the hook's index map (N K mats per index; v3/e1b/mkmask.py labels it). Env: E3_CKPT (bf16 model dir).
"""
import json, math, os, sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkmask

CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16")
P = "model.language_model.layers."
CONFIGS = [(8, g, c) for g in (0, 1024, 512, 256, 128, 64, 32) for c in (1.0, 0.9)] + \
          [(4, g, c) for g in (0, 512, 256, 128, 64, 32) for c in (1.0, 0.85, 0.7)]


def hadamard(n=128):
    h = np.array([[1.0]])
    while h.shape[0] < n: h = np.block([[h, h], [h, -h]])
    return h


H = hadamard().astype(np.float32)


def rotate(x, s):   # x [R, K] f32, s [K] -> f16-rounded rotated rows, as the kernel
    y = (x * s).astype(np.float16).astype(np.float32)
    R, K = y.shape
    y = (y.reshape(R, K // 128, 128) @ H.T).reshape(R, K) / math.sqrt(128)
    return y.astype(np.float16).astype(np.float32)


def unrotate(e, s):  # exact inverse of the rotation's linear part
    R, K = e.shape
    return (e.reshape(R, K // 128, 128) @ H).reshape(R, K) / math.sqrt(128) / s


def quant(x, bits, group, clip):
    Q = float(2 ** (bits - 1) - 1)
    R, K = x.shape
    g = K if group == 0 else group
    xg = x.reshape(R, K // g, g)
    m = np.abs(xg).max(2, keepdims=True)
    sc = np.where(m > 0, clip * m / Q, 1.0)
    q = np.clip(np.rint(xg / sc), -Q, Q) * sc
    return q.reshape(R, K).astype(np.float16).astype(np.float32)


_IDX = None
def weight(names):
    global _IDX
    if _IDX is None: _IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    out = []
    for n in names:
        with safe_open(f"{CK}/{_IDX[n]}", "pt") as f:   # bf16: through torch (numpy has no bfloat16)
            w = f.get_tensor(n); N = w.shape[0]
            if N > 4096:
                rows = np.sort(np.random.default_rng(len(n)).choice(N, 4096, replace=False))
                w = w[torch.from_numpy(rows)]
        out.append(w.float().numpy())
    return out


def hf_names(fam, layer, n, k, idx_layer_shapes):
    if fam == "head": return ["lm_head.weight"]
    p = f"{P}{layer}."
    if fam == "gate_up": return [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]
    if fam == "down": return [p + "mlp.down_proj.weight"]
    if fam == "ssm_out": return [p + "linear_attn.out_proj.weight"]
    if fam == "attn_out": return [p + "self_attn.o_proj.weight"]
    cands = ([p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"] if fam == "gdn_in"
             else [p + f"self_attn.{t}_proj.weight" for t in ("q", "k", "v")])
    return [idx_layer_shapes[(c, n)] for c in cands if (c, n) in idx_layer_shapes][:1]


def shapes_for(names):
    global _IDX
    if _IDX is None: _IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    d = {}
    for n in names:
        if n in _IDX:
            with safe_open(f"{CK}/{_IDX[n]}", "pt") as f: d[(n, f.get_slice(n).get_shape()[0])] = n
    return d


def main():
    D, mp, idxs, out = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3].split(",")], sys.argv[4]
    rows = sorted(tuple(map(int, l.split())) for l in open(mp) if l.strip())
    lab = mkmask.label(rows)
    for i in idxs:
        fam, layer = lab[i]; _, n, k, mats = rows[i]
        K, bf, m2 = map(int, open(f"{D}/{i}.meta").read().split())
        raw = np.fromfile(f"{D}/{i}.x", dtype=np.uint16).reshape(-1, K)
        x = ((raw.astype(np.uint32) << 16).view(np.float32) if bf else raw.view(np.float16).astype(np.float32))
        suhs = [np.fromfile(f"{D}/{i}.suh", dtype=np.float16).astype(np.float32)]
        if m2 == 2: suhs.append(np.fromfile(f"{D}/{i}.suh1", dtype=np.float16).astype(np.float32))
        p = f"{P}{layer}."
        cands = ([p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"] if fam == "gdn_in" else
                 [p + f"self_attn.{t}_proj.weight" for t in ("q", "k", "v")] if fam == "attn_in" else [])
        names = hf_names(fam, layer, n, k, shapes_for(cands))
        Ws = weight(names)
        assert len(Ws) == len(suhs), (i, fam, names, len(suhs))
        xr = [rotate(x, s) for s in suhs]
        ref = sum(float((x @ W.T).astype(np.float64).__pow__(2).sum()) for W in Ws)
        x2 = float((x.astype(np.float64) ** 2).sum()) * len(suhs)
        for bits, g, c in CONFIGS:
            if g and K % g: continue
            eo = ei = 0.0
            for s, r, W in zip(suhs, xr, Ws):
                e = unrotate(r - quant(r, bits, g, c), s)
                eo += float(((e @ W.T).astype(np.float64) ** 2).sum()); ei += float((e.astype(np.float64) ** 2).sum())
            o = {"i": i, "fam": fam, "layer": layer, "N": n, "K": K, "rows": x.shape[0], "bits": bits, "group": g,
                 "clip": c, "rel_out": eo / ref, "rel_in": ei / x2}
            with open(out, "a") as f: f.write(json.dumps(o) + "\n")
        print(i, fam, layer, "done", flush=True)


if __name__ == "__main__":
    main()
