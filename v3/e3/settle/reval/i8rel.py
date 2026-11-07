#!/usr/bin/env python3
"""i8rel.py QDIR CKPT HESSDIR OUT.json -- w46 item 5: rel = tr(dW H dW^T) / tr(W H W^T) per linear for every
QDIR/k<K>-<cb>/wq/<key>.safetensors (i8real.sh's --save-wq output) against the bf16 checkpoint's weight and the
undamped calib H_in (challenge layout, hessian_upper), then the i8 / mul1 ratio table. CPU only.
(ourq's own wq/<key>.rowerr.npz came out err ≈ ref on these runs, i.e. it compared against an already-consumed w, so
it is not used.)"""
import glob, json, os, struct, sys
import torch
from safetensors import safe_open

torch.set_num_threads(4)
Q, CK, HD, OUT = sys.argv[1:5]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]


def layer_of(key): return key.split(".layers.")[0] + ".layers." + key.split(".layers.")[1].split(".")[0]


def hname(key):
    """the input Hessian ourq.py's layer_plan uses for this linear"""
    if key.endswith(("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")): return f"{layer_of(key)}.input"
    if key.endswith(("gate_proj", "up_proj")): return f"{layer_of(key)}.post_attention.input"
    return f"{key}.input"                                  # o_proj, out_proj, down_proj


_H = {}
def H_of(name):
    if name not in _H:
        p = f"{HD}/{name}.safetensors"
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
        with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").double()
        H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
        _H.clear(); _H[name] = H + H.triu(1).T
    return _H[name]


R = {}
files = sorted(glob.glob(f"{Q}/k*-*/wq/*.safetensors"), key=lambda f: os.path.basename(f))
for f in files:
    arm = f.split("/")[-3]; K, cb = arm[1:].split("-"); key = os.path.basename(f)[:-len(".safetensors")]
    hn = hname(key)
    with safe_open(f"{CK}/{idx[key + '.weight']}", "pt") as g: W = g.get_tensor(key + ".weight").double()
    with safe_open(f, "pt") as g: Wq = g.get_tensor("weight").double()
    H = H_of(hn); E = W - Wq
    R[f"{key}|{K}|{cb}"] = float(((E @ H) * E).sum() / ((W @ H) * W).sum())
json.dump(R, open(OUT, "w"), indent=1)
keys = sorted({k.split("|")[0] for k in R}); Ks = sorted({float(k.split("|")[1]) for k in R})
print("tensor                         " + " ".join(f"i8/mul1 K{K:g}" for K in Ks) + "   rel mul1 " + " ".join(f"K{K:g}" for K in Ks))
lr = {K: [] for K in Ks}
for key in keys:
    r = []
    for K in Ks:
        a, b = R.get(f"{key}|{K:g}|i8"), R.get(f"{key}|{K:g}|mul1")
        r.append(a / b if a and b else None)
        if a and b: lr[K].append(a / b)
    print(f"{key.split('layers.')[1]:30s} " + " ".join(f"{x:12.4f}" if x else "           -" for x in r) + "   " +
          " ".join(f"{R.get(f'{key}|{K:g}|mul1', 0):.2e}" for K in Ks))
import numpy as np
print("median                         " + " ".join(f"{np.median(lr[K]):12.4f}" for K in Ks))
print("geo mean                       " + " ".join(f"{np.exp(np.mean(np.log(lr[K]))):12.4f}" for K in Ks))
print("min / max                      " + " ".join(f"{min(lr[K]):.3f}/{max(lr[K]):.3f}" for K in Ks))
