#!/usr/bin/env python3
"""objcheck.py MODEL SNAP QDIR... -- for each quant dir (ourq --save-wq), per linear: one-sided error tr(E Hc E^T)/tr(W Hc W^T)
(Hc = the calib input Hessian ourq used) and two-sided tr(E Hi E^T Ho)/tr(W Hi W^T Ho) (Hi, Ho = Sketch B from SNAP).
A two-sided quant that does not win on the two-sided objective means the quantizer path is wrong. CPU."""
import json, os, struct, sys, torch
from safetensors import safe_open
torch.set_num_threads(12)
mp, snap, qdirs = sys.argv[1], sys.argv[2], sys.argv[3:]
idx = json.load(open(f"{mp}/model.safetensors.index.json"))["weight_map"]
def ck(k):
    with safe_open(f"{mp}/{idx[k]}", "pt") as f: return f.get_tensor(k).double()
def up(p, t="hessian_upper"):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
    with safe_open(p, "pt") as f: u = f.get_tensor(t).double()
    H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = u; return H + H.triu(1).T
L = "model.language_model.layers"
def grp(key):
    l = key.split(".")[3]; p = f"{L}.{l}"; t = key.split(f"{p}.")[1]
    return {"linear_attn.in_proj_qkv": "input", "linear_attn.in_proj_z": "input", "self_attn.q_proj": "input",
            "self_attn.k_proj": "input", "self_attn.v_proj": "input", "linear_attn.out_proj": "linear_attn.out_proj.input",
            "self_attn.o_proj": "self_attn.o_proj.input", "mlp.gate_proj": "post_attention.input",
            "mlp.up_proj": "post_attention.input", "mlp.down_proj": "mlp.down_proj.input"}[t]
keys = [f"{L}.{l}.{t}" for l in (0, 5, 11, 19, 23) for t in ("linear_attn.in_proj_qkv", "linear_attn.out_proj", "mlp.gate_proj", "mlp.down_proj")]
keys = [k for k in keys if os.path.exists(f"{snap}/hout/{k}.safetensors")]
print("key".ljust(42) + "".join(f"{os.path.basename(q)[2:]:>22}" for q in qdirs) + "   (one-sided / two-sided, rel)")
for k in keys:
    W = ck(k + ".weight"); l = k.split(".")[3]
    Hc = up(f"/data/h08/calib/{L}.{l}.{grp(k)}.safetensors")
    Hi = up(f"{snap}/hin/{k}.safetensors")
    with safe_open(f"{snap}/hout/{k}.safetensors", "pt") as f: Ho = f.get_tensor("hout").double()
    r1 = (W @ Hc * W).sum(); r2 = ((W @ Hi @ W.T) * Ho).sum()
    row = []
    for q in qdirs:
        with safe_open(f"{q}/wq/{k}.safetensors", "pt") as f: Q = f.get_tensor("weight").double()
        E = W - Q
        row.append(f"{((E @ Hc * E).sum() / r1).item():9.2e}/{(((E @ Hi @ E.T) * Ho).sum() / r2).item():9.2e}")
    print(k.split("layers.")[1].ljust(42) + "".join(f"{x:>22}" for x in row), flush=True)
