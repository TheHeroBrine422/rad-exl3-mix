#!/usr/bin/env python3
"""d2score.py OUT.jsonl LAYERS HA HB HPUB ARM... -- D2: is our input-side proxy in-sample?

Per tensor of the input groups of LAYERS (q/k/v or GDN qkv/z, o_proj / out_proj, gate/up), per ARM, the proxy error
rel = tr(E^T H E) / tr(W^T H W) (E = W_q - W, W the bf16 checkpoint weight, (in, out)) under three Hessians:
HA / HB (collect.py YAQA_HSPLIT: disjoint document halves of the non-verify calib docs) and HPUB (the published calib
Hessians, which include every calib doc). ARM:
  name:ours:DIR   our quant, dequantized weights from ourq.py --save-wq (DIR/wq/<key>.safetensors "weight", (out, in))
  name:td:DIR     turboderp's EXL3 download (DIR = td-4.00bpw etc.), dequantized with exllamav3's LinearEXL3
Reading: arms quantized on HA are in-sample on HA and out-of-sample on HB; turboderp's on neither. If our advantage
over his (or the damping arm that wins) holds on HB as on HA, the proxy is not misleading us. One card."""
import json, os, struct, sys
import torch
from safetensors import safe_open

OUT, LAYERS, HA, HB, HPUB = sys.argv[1:6]; ARMS = sys.argv[6:]
CK = os.environ.get("RXM_CKPT", "/data/workspace/models/qwen38-27b-bf16")
dev = torch.device("cuda")
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]


def ck(key):
    with safe_open(f"{CK}/{idx[key]}", "pt") as f: return f.get_tensor(key)


def hess(d, name):
    p = f"{d}/{name}.safetensors"
    if not os.path.exists(p): return None
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
    with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").to(dev).float()
    H = torch.zeros(k, k, device=dev); iu = torch.triu_indices(k, k, device=dev); H[iu[0], iu[1]] = up
    return H + H.triu(1).T


def groups(l):
    p = f"model.language_model.layers.{l}"
    if l % 4 == 3:
        return [(f"{p}.input", [f"{p}.self_attn.{x}" for x in ("q_proj", "k_proj", "v_proj")]),
                (f"{p}.self_attn.o_proj.input", [f"{p}.self_attn.o_proj"]),
                (f"{p}.post_attention.input", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])]
    return [(f"{p}.input", [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"]),
            (f"{p}.linear_attn.out_proj.input", [f"{p}.linear_attn.out_proj"]),
            (f"{p}.post_attention.input", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])]


_td = {}
def td_weight(d, key):
    """turboderp's tensor `key` dequantized, (in, out) float"""
    if d not in _td:
        ix = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"] if os.path.exists(f"{d}/model.safetensors.index.json") \
            else {k: "model.safetensors" for k in safe_open(f"{d}/model.safetensors", "pt").keys()}
        _td[d] = ix
    ix = _td[d]
    def g(sub):
        k = f"{key}.{sub}"
        if k not in ix: return None
        with safe_open(f"{d}/{ix[k]}", "pt") as f: return f.get_tensor(k).to(dev)
    from exllamav3.modules.quant.exl3 import LinearEXL3
    tr, suh, svh = g("trellis"), g("suh"), g("svh")
    lin = LinearEXL3(None, suh.shape[0], svh.shape[0], suh=suh, svh=svh, trellis=tr, mcg=g("mcg"), mul1=g("mul1"), key=key)
    return lin.get_weight_tensor().float()


def ours_weight(d, key):
    with safe_open(f"{d}/wq/{key}.safetensors", "pt") as f: return f.get_tensor("weight").to(dev).float().T


out = open(OUT, "a")
for l in [int(x) for x in LAYERS.split(",")]:
    for gname, keys in groups(l):
        Hs = {n: hess(d, gname) for n, d in (("A", HA), ("B", HB), ("pub", HPUB))}
        if Hs["A"] is None or Hs["B"] is None: print(f"skip {gname}: no split H", flush=True); continue
        for key in keys:
            W = ck(key + ".weight").to(dev).float().T            # (in, out)
            ref = {n: float((W * (H @ W)).sum()) for n, H in Hs.items() if H is not None}
            for arm in ARMS:
                name, kind, d = arm.split(":", 2)
                try: Wq = ours_weight(d, key) if kind == "ours" else td_weight(d, key)
                except Exception as e: print(f"{name} {key}: {e}", flush=True); continue
                E = Wq - W
                r = {"layer": l, "key": key, "group": gname, "arm": name,
                     **{f"rel_{n}": float((E * (H @ E)).sum()) / ref[n] for n, H in Hs.items() if H is not None}}
                out.write(json.dumps(r) + "\n"); out.flush(); print(json.dumps(r), flush=True)
                del Wq, E
            del W
        del Hs; torch.cuda.empty_cache()
