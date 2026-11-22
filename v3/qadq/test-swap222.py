#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test-swap222.py OUTDIR MODEL_DIR LAYERS_CKPT -- builds two synthetic checkpoints from a real one (w222 test of swap222.py): OUTDIR/ident = the checkpoint + norms.safetensors with the bf16 checkpoint's own norm
weights (HF names, exactly what write_ckpt exports at gain 1) + scales/lm_head.safetensors with r = c = 1; OUTDIR/gain = the same with every norm weight (1 + w) * 1.01 - 1 and head r = 1.02. Then swap222 of `ident` must equal
swap222 of the plain checkpoint byte for byte, `gain` must differ only in the norm / head planes (--verify)."""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
out, M, ck = sys.argv[1:4]
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
names = [k for k in idx if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "linear_attn.norm.weight")) and k.startswith("model.language_model.layers.")] + ["model.language_model.norm.weight"]
def get(k):
    with safe_open(f"{M}/{idx[k]}", "pt") as f: return f.get_tensor(k)
for tag, g, hr in (("ident", 1.0, 1.0), ("gain", 1.01, 1.02)):
    d = f"{out}/{tag}"; os.makedirs(f"{d}/scales", exist_ok=True)
    for f in os.listdir(f"{ck}/scales"): os.symlink(os.path.realpath(f"{ck}/scales/{f}"), f"{d}/scales/{f}") if not os.path.exists(f"{d}/scales/{f}") else None
    nw = {}
    for k in names:
        w = get(k).float(); gated = k.endswith("linear_attn.norm.weight")
        nw[k] = (w * g if gated else (1 + w) * g - 1).to(torch.bfloat16).contiguous()
    save_file(nw, f"{d}/norms.safetensors")
    save_file({"r": torch.full((5120,), hr), "c": torch.ones(248320)}, f"{d}/scales/lm_head.safetensors")
    open(f"{d}/.done", "w").close()
print(len(names), "norm tensors")
