#!/usr/bin/env python3
"""mkperm.py build PERMS.npz NAME CKPT HIN HDIAG OUTDIR -- B6: a copy of the 0.8B whose MLP intermediate neurons are
permuted (gate / up rows and down columns by PERMS L<l>/NAME; the function is unchanged: SwiGLU is elementwise),
for ourq.py:
  OUTDIR/model/                 config + tokenizer symlinked, one shard with every tensor (MLP ones permuted)
  OUTDIR/w/challenge/hessians/calib/   symlinks, except mlp.down_proj.input (rows / columns permuted)
  OUTDIR/hout/                  symlinks, except mlp.gate_proj / up_proj (permuted vectors)
  OUTDIR/mlp.txt                the MLP keys (ourq --only: gate+up and down are whole Hessian groups)
mkperm.py unperm PERMS.npz NAME PWQ DONORWQ OUTWQ -- the permuted quant's dequantized MLP weights put back in the
natural order (q[p] = q_perm for gate / up rows, down columns), every other tensor a symlink to DONORWQ (the natural
quant of the same seed and settings: ourq's seeds depend only on a tensor's position, so those files are what a
full run would have written) -> an evalq.py --wq dir for the original model."""
import glob, json, os, struct, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

mode, pf, name = sys.argv[1:4]
PZ = np.load(pf); P = {int(k.split("/")[0][1:]): torch.from_numpy(PZ[k]) for k in PZ.files if k.endswith("/" + name)}
L = "model.language_model.layers"
def link(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.abspath(src), dst)

if mode == "build":
    ck, hin, hd, o = sys.argv[4:8]
    os.makedirs(f"{o}/model", exist_ok=True)
    for f in os.listdir(ck):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json": link(f"{ck}/{f}", f"{o}/model/{f}")
    idx = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
    T = {}
    for f in sorted(set(idx.values())):
        with safe_open(f"{ck}/{f}", "pt") as h:
            for k in h.keys(): T[k] = h.get_tensor(k)
    for l, p in P.items():
        for x in ("gate", "up"): T[f"{L}.{l}.mlp.{x}_proj.weight"] = T[f"{L}.{l}.mlp.{x}_proj.weight"][p].contiguous()
        T[f"{L}.{l}.mlp.down_proj.weight"] = T[f"{L}.{l}.mlp.down_proj.weight"][:, p].contiguous()
    sh = "model-perm.safetensors"
    save_file(T, f"{o}/model/{sh}", metadata={"format": "pt", "w04_perm": name})
    json.dump({"metadata": {}, "weight_map": {k: sh for k in T}}, open(f"{o}/model/model.safetensors.index.json", "w"))
    hc = f"{o}/w/challenge/hessians/calib"; os.makedirs(hc, exist_ok=True)
    for f in glob.glob(f"{hin}/*.safetensors"): link(f, f"{hc}/{os.path.basename(f)}")
    for l, p in P.items():
        g = f"{L}.{l}.mlp.down_proj.input.safetensors"
        with open(f"{hin}/{g}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
        k = int(meta["k"]); up = load_file(f"{hin}/{g}")["hessian_upper"]
        H = torch.zeros(k, k, dtype=up.dtype); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up; H = H + H.triu(1).T
        H = H[p][:, p]
        os.remove(f"{hc}/{g}"); save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hc}/{g}", metadata=dict(meta, w04_perm=name))
    os.makedirs(f"{o}/hout", exist_ok=True)
    for f in glob.glob(f"{hd}/*.safetensors"): link(f, f"{o}/hout/{os.path.basename(f)}")
    for l, p in P.items():
        for x in ("gate", "up"):
            f = f"{L}.{l}.mlp.{x}_proj.safetensors"; v = load_file(f"{hd}/{f}")["hout_diag"]
            os.remove(f"{o}/hout/{f}"); save_file({"hout_diag": v[p].contiguous()}, f"{o}/hout/{f}", metadata={"w04_perm": name})
    with open(f"{o}/mlp.txt", "w") as fh:
        for l in sorted(P):
            for x in ("gate", "up", "down"): fh.write(f"{L}.{l}.mlp.{x}_proj\n")
    print(f"built {o}: {len(P)} layers permuted ({name})")
elif mode == "unperm":
    pwq, dwq, owq = sys.argv[4:7]
    os.makedirs(owq, exist_ok=True); n = 0
    for f in glob.glob(f"{dwq}/*.safetensors"): link(f, f"{owq}/{os.path.basename(f)}")
    for l, p in P.items():
        for x in ("gate", "up", "down"):
            f = f"{L}.{l}.mlp.{x}_proj.safetensors"; q = load_file(f"{pwq}/{f}")["weight"]; r = torch.empty_like(q)
            if x == "down": r[:, p] = q
            else: r[p] = q
            os.remove(f"{owq}/{f}"); save_file({"weight": r}, f"{owq}/{f}"); n += 1
    print(f"unpermuted {n} MLP tensors into {owq}")
