#!/usr/bin/env python3
"""hscore.py CKPT HESS TESTDIR ARM=WQDIR ... -- output-weighted proxy error of each arm's dequantized linears on an
independent output factor: E = sum_i h_i (w_i - q_i) H_in (w_i - q_i)^T / sum_i h_i w_i H_in w_i^T, h = TESTDIR's
diagonal (odd blocks), H_in = the calib input Hessian. One JSON line per (arm, tensor); GPU if available."""
import glob, json, os, sys, struct
import torch
from safetensors import safe_open
CK, HD, TD = sys.argv[1:4]; ARMS = [a.split("=", 1) for a in sys.argv[4:]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def hin_group(key):   # model.language_model.layers.L.<rest> -> the challenge input-Hessian name
    s = key.split("."); p, rest = ".".join(s[:4]), ".".join(s[4:])
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"): return f"{p}.input"
    return {"self_attn.o_proj": f"{p}.self_attn.o_proj.input", "linear_attn.out_proj": f"{p}.linear_attn.out_proj.input",
            "mlp.gate_proj": f"{p}.post_attention.input", "mlp.up_proj": f"{p}.post_attention.input",
            "mlp.down_proj": f"{p}.mlp.down_proj.input"}.get(rest)
HC = {}
def H(g):
    if g not in HC:
        HC.clear(); f = f"{HD}/{g}.safetensors"
        with safe_open(f, "pt") as h: k = int(h.metadata()["k"]); u = h.get_tensor("hessian_upper")
        M = torch.zeros(k, k, dtype=torch.float32, device=dev); iu = torch.triu_indices(k, k, device=dev)
        M[iu[0], iu[1]] = u.to(dev, torch.float32); M = M + M.triu(1).T; HC[g] = M / M.diagonal().mean()
    return HC[g]
keys = sorted({os.path.basename(f)[:-12] for _, d in ARMS for f in glob.glob(f"{d}/*.safetensors")})
for key in keys:
    g = hin_group(key)
    if g is None: continue
    with safe_open(f"{CK}/{idx[key + '.weight']}", "pt") as h: W = h.get_tensor(key + ".weight").to(dev, torch.float32)
    tf = f"{TD}/{key}.safetensors"
    if not os.path.exists(tf): continue
    with safe_open(tf, "pt") as h: hv = h.get_tensor("hout_diag").to(dev, torch.float32)
    Hm = H(g); den = float((hv * ((W @ Hm) * W).sum(1)).sum())
    for arm, d in ARMS:
        f = f"{d}/{key}.safetensors"
        if not os.path.exists(f): continue
        with safe_open(f, "pt") as h: Q = h.get_tensor("weight").to(dev, torch.float32)
        D = W - Q; num = float((hv * ((D @ Hm) * D).sum(1)).sum()); plain = float(((D @ Hm) * D).sum() / ((W @ Hm) * W).sum())
        print(json.dumps({"arm": arm, "key": key, "params": W.numel(), "E_out": num / den, "E_in": plain}), flush=True)
