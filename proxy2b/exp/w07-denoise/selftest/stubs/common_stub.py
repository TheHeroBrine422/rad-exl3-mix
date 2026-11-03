"""shared bits of the dry-run stubs: key / group names of the 0.8B, a tiny safetensors writer"""
import os, torch
from safetensors.torch import save_file
L = "model.language_model.layers"
def keys(l):
    p = f"{L}.{l}"
    if l % 4 == 3: return [f"{p}.self_attn.{s}_proj" for s in "qkvo"] + [f"{p}.mlp.{s}_proj" for s in ("gate", "up", "down")]
    return [f"{p}.linear_attn.{s}" for s in ("in_proj_qkv", "in_proj_z", "out_proj")] + [f"{p}.mlp.{s}_proj" for s in ("gate", "up", "down")]
def groups(l):
    p = f"{L}.{l}"
    return [f"{p}.input", f"{p}.self_attn.o_proj.input" if l % 4 == 3 else f"{p}.linear_attn.out_proj.input",
            f"{p}.post_attention.input", f"{p}.mlp.down_proj.input"]
def put(path, t, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True); save_file(t, path, metadata=meta)
