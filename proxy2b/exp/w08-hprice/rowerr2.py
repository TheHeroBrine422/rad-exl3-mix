#!/usr/bin/env python3
"""rowerr2.py CKPT HESS HOUTDIR ARM_WQ [ARM_WQ ...] -- per-tensor one-sided and diagonal-two-sided proxy errors of
0.8B quants, recomputed from their dequantized weights (ourq.py --save-wq <key>.safetensors), the bf16 checkpoint and
the raw calib input Hessians (HESS/<group>.safetensors, packed upper triangle, sum over rows).

Why not ourq's <key>.rowerr.npz: quantize_exl3 regularizes its weight argument in place (weight /= sv, Hadamard,
/= su, * g_scale), and ourq passes `w` itself, so the rowerr.npz `err` and `ref` are energies of the REGULARIZED weight
(err / ref = 1.000-1.002 on every 0.8B tensor; w08 README, finding 1). This recomputes them from scratch.

Per linear (weight W [out, in], quant Q, E = W - Q, H the calib Hessian, h = HOUTDIR's hout_diag over output rows):
  e1  = Σ_r E_r H E_rᵀ / Σ_r W_r H W_rᵀ                       (one-sided: what proxy.jsonl / KN_LEVELS price by)
  e2  = Σ_r h_r E_r H E_rᵀ / Σ_r h_r W_r H W_rᵀ               (diagonal two-sided: the objective ldlq_2hess targets)
  e2u = Σ_r h_r E_r H E_rᵀ / (mean(h) Σ_r W_r H W_rᵀ)        (absolute h-weighted error, ref unweighted)
One JSON line per (arm, key) on stdout; per-row err / ref arrays with --rows DIR (npz per arm and key).
CPU only; torch threads from OMP_NUM_THREADS (the 4070 rule: <= 4)."""
import glob, json, os, struct, sys
import numpy as np, torch
from safetensors import safe_open

a = sys.argv[1:]
ROWS = None
if "--rows" in a: i = a.index("--rows"); ROWS = a[i + 1]; del a[i:i + 2]
CK, HESS, HOUT, arms = a[0], a[1], a[2], a[3:]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_f = {}
def ck(k):
    f = idx[k]
    if f not in _f: _f[f] = safe_open(f"{CK}/{f}", "pt")
    return _f[f].get_tensor(k)

def hname(key):                     # ourq.py layer_plan: the input-Hessian group of a linear
    p, rest = key.split(".layers.")[0] + ".layers." + key.split(".layers.")[1].split(".")[0], key.split(".", 4)[4]
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        return f"{p}.input"
    return {"self_attn.o_proj": f"{p}.self_attn.o_proj.input", "linear_attn.out_proj": f"{p}.linear_attn.out_proj.input",
            "mlp.gate_proj": f"{p}.post_attention.input", "mlp.up_proj": f"{p}.post_attention.input",
            "mlp.down_proj": f"{p}.mlp.down_proj.input"}[rest]

_H = {}
def hess(name):
    if name not in _H:
        _H.clear()
        p = f"{HESS}/{name}.safetensors"
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
        with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").double()
        H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
        _H[name] = H + H.triu(1).T
    return _H[name]

def rowq(H, M):                     # per row r of M [out, in]: M_r H M_rᵀ (float64)
    return ((M @ H) * M).sum(1)

def hdiag(key):
    p = f"{HOUT}/{key}.safetensors"
    if not os.path.exists(p): return None
    with safe_open(p, "pt") as f:
        return (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).double()

for arm in arms:
    tag = os.path.basename(os.path.dirname(arm.rstrip("/")))
    for fq in sorted(glob.glob(f"{arm}/*.safetensors")):
        key = os.path.basename(fq)[:-len(".safetensors")]
        with safe_open(fq, "pt") as f: Q = f.get_tensor("weight").double()
        W = ck(key + ".weight").double(); assert W.shape == Q.shape, (key, W.shape, Q.shape)
        H = hess(hname(key))
        err, ref = rowq(H, W - Q), rowq(H, W)
        h = hdiag(key)
        r = {"arm": tag, "key": key, "e1": float(err.sum() / ref.sum())}
        if h is not None:
            r["e2"] = float((h * err).sum() / (h * ref).sum()); r["e2u"] = float((h * err).sum() / (h.mean() * ref.sum()))
        print(json.dumps(r), flush=True)
        if ROWS:
            os.makedirs(f"{ROWS}/{tag}", exist_ok=True)
            np.savez(f"{ROWS}/{tag}/{key}.npz", err=err.numpy(), ref=ref.numpy())
