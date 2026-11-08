#!/usr/bin/env python3
"""btrace82.py LAYER OUT.json -- #117 1(a) (../w82.md): per-128-block trace shares of every residual reader of one
layer. w = block trace of the LDLQ metric diag(su) (H + delta I) diag(su) (settle.P2's su, adopted damping), raw = block
trace of the calib H_in. One JSON object {tensor: {...}} written whole (OUT.part, then renamed). E3_CKPT / E3_HESS as e3.py."""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, ".."), os.path.join(_h, "../..")]
import e3
torch.set_grad_enabled(False)

READERS = {"ssm_qkv": ("linear_attn.in_proj_qkv", "input", 0.005), "ssm_z": ("linear_attn.in_proj_z", "input", 0.005),
           "attn_qg": ("self_attn.q_proj", "input", 0.005), "attn_k": ("self_attn.k_proj", "input", 0.005),
           "attn_v": ("self_attn.v_proj", "input", 0.005),
           "ffn_gate": ("mlp.gate_proj", "post_attention.input", 0.025), "ffn_up": ("mlp.up_proj", "post_attention.input", 0.025)}


def hdiag(layer, hn):
    with safe_open(f"{e3.HD}/{e3.P}{layer}.{hn}.safetensors", "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").double()
    # row i of the packed upper triangle starts at i*k - i*(i-1)/2 (torch.triu_indices order)
    i = torch.arange(k, dtype=torch.int64)
    return up[i * k - i * (i - 1) // 2]


def main():
    layer, out = int(sys.argv[1]), sys.argv[2]
    idx = json.load(open(f"{e3.CK}/model.safetensors.index.json"))["weight_map"]
    res, hd = {}, {}
    for t, (name, hn, sig) in READERS.items():
        k = f"{e3.P}{layer}.{name}.weight"
        if k not in idx: continue
        if hn not in hd: hd[hn] = hdiag(layer, hn)
        H = hd[hn]
        with safe_open(f"{e3.CK}/{idx[k]}", "pt") as f: W = f.get_tensor(k).double()
        r = W.pow(2).mean(1).sqrt().clamp(min=1e-12)
        su = (W / r[:, None]).pow(2).mean(0).sqrt(); su[su < 1e-30] = 0.1
        Hd = H.clone(); Hd[Hd == 0] = 1; Hd += sig * H.mean()
        m = su.pow(2) * Hd
        wb = m.reshape(-1, 128).sum(1); rb = H.reshape(-1, 128).sum(1)
        res[t] = {"N": W.shape[0], "Kd": W.shape[1], "w": (wb / wb.sum()).tolist(), "raw": (rb / rb.sum()).tolist(),
                  "su_block_ms": su.pow(2).reshape(-1, 128).mean(1).tolist()}
        del W
    with open(out + ".part", "w") as f: json.dump({"layer": layer, "t": res}, f)
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main()
