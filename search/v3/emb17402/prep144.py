#!/usr/bin/env python3
"""prep144.py CKPT OUT -- (w144, #174-02) inputs for quantizing the token table with ourq.py's head path:
  OUT/ckpt/  a one-tensor "checkpoint": model.safetensors.index.json + emb.safetensors holding CKPT's
             model.language_model.embed_tokens.weight under the name lm_head.weight (bf16, unchanged bytes)
  OUT/work/challenge/hessians/calib/lm_head.input.safetensors
             the identity input Hessian (k 5120, rows 1, packed upper triangle): LDLQ without feedback, i.e. plain row
             error, the measure w135 priced the trellis embedding on (embprices174.json)
ourq.py runs with RXM_CKPT=OUT/ckpt RXM_WORK=OUT/work --only-head; quant144.sh renames the output to the
model.language_model.embed_tokens.* keys the exl3i importer reads for token_embd.weight."""
import json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ck, out = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
name = "model.language_model.embed_tokens.weight"
os.makedirs(f"{out}/ckpt", exist_ok=True)
if not os.path.exists(f"{out}/ckpt/emb.safetensors"):
    with safe_open(f"{ck}/{idx[name]}", "pt") as f: w = f.get_tensor(name)
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (248320, 5120), (w.dtype, w.shape)
    save_file({"lm_head.weight": w.contiguous()}, f"{out}/ckpt/emb.safetensors.part", metadata={"source": f"{ck} {name}"})
    os.replace(f"{out}/ckpt/emb.safetensors.part", f"{out}/ckpt/emb.safetensors")
    json.dump({"metadata": {}, "weight_map": {"lm_head.weight": "emb.safetensors"}}, open(f"{out}/ckpt/model.safetensors.index.json", "w"))
hd = f"{out}/work/challenge/hessians/calib"; os.makedirs(hd, exist_ok=True)
if not os.path.exists(f"{hd}/lm_head.input.safetensors"):
    k = 5120
    iu = torch.triu_indices(k, k)
    up = (iu[0] == iu[1]).float()
    save_file({"hessian_upper": up}, f"{hd}/lm_head.input.safetensors.part",
              metadata={"k": str(k), "rows": "1", "layout": "upper triangle, row-major, packed", "linears": "lm_head",
                        "note": "identity (w144 #174-02: token table, plain row error)"})
    os.replace(f"{hd}/lm_head.input.safetensors.part", f"{hd}/lm_head.input.safetensors")
print("prep done", out)
