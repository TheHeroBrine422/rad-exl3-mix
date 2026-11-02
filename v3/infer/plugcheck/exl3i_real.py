#!/usr/bin/env python3
"""exl3i_real.py LEVEL_DIR CODEBOOK NAME... -- end-to-end import check of a real quantizer level: each logical weight
is imported and decoded through exl3quant.so (exl3i_real), then compared with the bf16 weight under the calib input
Hessian: rel = tr(E H E^T) / tr(W H W^T) per part, beside ourq's logged proxy_err (same quantity, computed by the
quantizer on its own decode). A wrong record layout, part order or scale shows up as rel ~ 1 or more; agreement to a
few percent (the logged figure uses the damped Hessian) means the plugin decodes what the quantizer wrote.
Env: EXL3I_REAL (the driver), EXL3QUANT_SO, RXM_CKPT, HESS."""
import json, os, struct, subprocess, sys
import numpy as np, torch
from safetensors import safe_open
D, CB, names = sys.argv[1], sys.argv[2], sys.argv[3:]
CK = os.environ.get("RXM_CKPT", "/data/workspace/models/qwen38-27b-bf16")
HESS = os.environ.get("HESS", "/root/challenge/hessians/calib")
DRV, SO = os.environ["EXL3I_REAL"], os.environ["EXL3QUANT_SO"]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def ck(n):
    with safe_open(f"{CK}/{idx[n]}", "pt") as f: return f.get_tensor(n).float()
logged = {}
for line in open(f"{D}/proxy.jsonl"):
    r = json.loads(line); logged[r["key"]] = r["proxy_err"]
L = "model.language_model.layers"
def parts(n):   # logical name -> [(hf key, hessian)]
    b = int(n.split(".")[1]); p = f"{L}.{b}"; t = n.split(".")[2]
    m = {"ffn_down": [("mlp.down_proj", "mlp.down_proj.input")],
         "ffn_gate_up": [("mlp.gate_proj", "post_attention.input"), ("mlp.up_proj", "post_attention.input")],
         "attn_qg": [("self_attn.q_proj", "input")], "attn_k": [("self_attn.k_proj", "input")],
         "attn_v": [("self_attn.v_proj", "input")], "attn_output": [("self_attn.o_proj", "self_attn.o_proj.input")],
         "ssm_qkv": [("linear_attn.in_proj_qkv", "input")], "ssm_z": [("linear_attn.in_proj_z", "input")],
         "ssm_out": [("linear_attn.out_proj", "linear_attn.out_proj.input")]}[t]
    return [(f"{p}.{a}", f"{p}.{h}") for a, h in m]
def hess(name):
    pth = f"{HESS}/{name}.safetensors"
    with open(pth, "rb") as fh: k = int(json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))["__metadata__"]["k"])
    with safe_open(pth, "pt") as f: up = f.get_tensor("hessian_upper").float()
    H = torch.zeros(k, k); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up; del up
    return H + H.triu(1).T
bad = 0
for n in names:
    ps = parts(n); W = torch.cat([ck(a + ".weight") for a, _ in ps]); rows, cols = W.shape
    out = f"/tmp/exl3i_real.{os.getpid()}.f32"
    r = subprocess.run([DRV, SO, D, CB, n, str(rows), str(cols), out], capture_output=True, text=True)
    print(r.stdout.strip()); 
    if r.returncode: print("  FAIL", r.stderr.strip()); bad += 1; continue
    Q = torch.from_numpy(np.fromfile(out, np.float32).reshape(rows, cols)); os.unlink(out)
    H = hess(ps[0][1]); per = rows // len(ps)
    for j, (a, _) in enumerate(ps):
        Wp, Qp = W[j * per:(j + 1) * per], Q[j * per:(j + 1) * per]; E = Wp - Qp
        rel = float((E @ H * E).sum() / (Wp @ H * Wp).sum()); lg = logged.get(a)
        ratio = rel / lg if lg else float("nan"); ok = lg is not None and 0.8 < ratio < 1.25
        bad += not ok
        print(f"  {a:<52} rel {rel:.6f}  logged {lg if lg is None else f'{lg:.6f}'}  ratio {ratio:.3f}  {'ok' if ok else 'FAIL'}")
print("EXL3I REAL", "PASS" if bad == 0 else f"FAIL ({bad})"); sys.exit(1 if bad else 0)
