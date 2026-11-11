#!/usr/bin/env python3
"""mkperm101.py CKPT HOUT OUT KEY... -- w101 #130-01 (../w101.md): the P16 arm's inputs. For every KEY (a linear,
without ".weight"): the bf16 weight with its output rows stored in P16 order (composable perm2.py gperm(R, 16): stored
128-row block b holds the 16-row tiles b, b + nb, b + 2 nb, ... of the natural order) and its bn H_out diagonal permuted
the same way.
  OUT/ckpt/model.safetensors   the permuted weights (an rxq --ckpt dir: no index, one file)
  OUT/hout/<key>.safetensors   "hout_diag", permuted
  OUT/perm.json                {key: perm} (stored row j = natural row perm[j])
Check: un-permuting each written tensor gives the source bit for bit. OUT/.done last."""
import json, os, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perm101 import gperm, inverse  # noqa: E402

CK, HO, OUT = sys.argv[1:4]; KEYS = sys.argv[4:]
os.makedirs(f"{OUT}/ckpt", exist_ok=True); os.makedirs(f"{OUT}/hout", exist_ok=True)
wmap = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
W, P = {}, {}
for k in KEYS:
    with safe_open(f"{CK}/{wmap[k + '.weight']}", "pt") as f: w = f.get_tensor(k + ".weight")
    with safe_open(f"{HO}/{k}.safetensors", "pt") as f: h = f.get_tensor("hout_diag")
    R = w.shape[0]; p = gperm(R, 16); ip = inverse(p)
    assert h.numel() == R and sorted(p.tolist()) == list(range(R))
    wp, hp = w[p].contiguous(), h[p].contiguous()
    assert torch.equal(wp[ip], w) and torch.equal(hp[ip], h)
    W[k + ".weight"] = wp; P[k] = p.tolist()
    save_file({"hout_diag": hp}, f"{OUT}/hout/{k}.safetensors", metadata={"w101": "P16", "src": f"{HO}/{k}.safetensors"})
    print(f"{k}: R {R}, first stored rows {p[:20].tolist()}", flush=True)
save_file(W, f"{OUT}/ckpt/model.safetensors.part", metadata={"w101": "P16"})
os.replace(f"{OUT}/ckpt/model.safetensors.part", f"{OUT}/ckpt/model.safetensors")
with safe_open(f"{OUT}/ckpt/model.safetensors", "pt") as f:     # read back
    for k in KEYS:
        with safe_open(f"{CK}/{wmap[k + '.weight']}", "pt") as g:
            assert torch.equal(f.get_tensor(k + ".weight")[inverse(torch.tensor(P[k]))], g.get_tensor(k + ".weight")), k
json.dump(P, open(f"{OUT}/perm.json", "w"))
open(f"{OUT}/.done", "w").close()
print("mkperm101: ok", flush=True)
