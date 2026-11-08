# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Qwen3.5-architecture plan (Qwen3.8-27B, the 0.8B / 2B proxies): which linears share an input Hessian, their seeds,
family names. Port of rad-exl3-mix quantize/ourq.py layer_plan() and its seed rule, unchanged.

Seeds (ourq): tensor j of Hessian group gi in layer l gets  l * 16 + gi * 4 + j + 100000 * seed_offset;  lm_head 9999.
The group's input-Hessian rotation (the sign vector su of finalize_capture_H) is drawn under the seed of the group's
FIRST tensor, which is why a per-tensor job of tensor j > 0 finalizes its Hessian under tensor 0's seed first
(quant.py): the result is the same as quantizing the group in one process."""

L = "model.language_model.layers"

# key suffix -> family (the challenge / knapsack names; gate and up are one family there, kept apart here)
FAMILY = (("self_attn.q_proj", "attn_qg"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
          ("self_attn.o_proj", "attn_output"), ("linear_attn.in_proj_qkv", "ssm_qkv"), ("linear_attn.in_proj_z", "ssm_z"),
          ("linear_attn.out_proj", "ssm_out"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
          ("mlp.down_proj", "ffn_down"), ("lm_head", "head"))
FAMILIES = tuple(f for _, f in FAMILY)
WRITERS = ("attn_output", "ssm_out", "ffn_down")          # residual writers (#87 asym target)
READERS = ("attn_qg", "attn_k", "attn_v", "ssm_qkv", "ssm_z", "ffn_gate", "ffn_up")   # residual readers (#117 block order)


def family(key):
    return next((f for s, f in FAMILY if key.endswith(s)), None)


def is_full_attention(l):
    return l % 4 == 3


def layer_plan(l):
    """[(input-Hessian name, [checkpoint keys of the linears that read it])] in ourq's order"""
    p = f"{L}.{l}"
    full = is_full_attention(l)
    return [(f"{p}.input", [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"] if full else
                           [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"]),
            (f"{p}.self_attn.o_proj.input" if full else f"{p}.linear_attn.out_proj.input",
             [f"{p}.self_attn.o_proj" if full else f"{p}.linear_attn.out_proj"]),
            (f"{p}.post_attention.input", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]),
            (f"{p}.mlp.down_proj.input", [f"{p}.mlp.down_proj"])]


HEAD_PLAN = [("lm_head.input", ["lm_head"])]


def layer_of(key):
    return int(key.split(".")[3]) if key.startswith(L + ".") else None


def locate(key):
    """(layer or None, group index, index in group, Hessian name, group keys) of a trunk linear or lm_head"""
    l = layer_of(key)
    plan = HEAD_PLAN if l is None else layer_plan(l)
    for gi, (hn, ks) in enumerate(plan):
        if key in ks: return l, gi, ks.index(key), hn, ks
    raise KeyError(f"{key}: not a quantized linear of this architecture")


def seed(key, seed_offset=0):
    """ourq's rotation seed of a tensor; also returns the seed of its group's first tensor (the su draw)"""
    l, gi, j, _, _ = locate(key)
    base = 9999 if l is None else l * 16 + gi * 4
    return base + j + 100000 * seed_offset, base + 100000 * seed_offset


def missing_group_heads(keys):
    """{key: its group's first tensor} for every key of a reference run (e.g. ourq --only) that is not first in its
    Hessian group while the group's first tensor is missing from keys: ourq then drew the group's su under that key's
    own seed, so the reference is not the ladder's (group run's) tensor and rxq cannot match it (gpucheck27 i4, 11-08)"""
    ks = set(keys)
    out = {}
    for k in sorted(ks):
        try: _, _, j, _, g = locate(k)
        except (KeyError, ValueError): continue    # not a quantized linear (other names of an index)
        if j > 0 and g[0] not in ks: out[k] = g[0]
    return out


def is_input_group(hname):
    """the layer-input group (attention q/k/v, GDN qkv/z): takes sigma_reg_in (ourq's test, verbatim)"""
    return hname.endswith(".layers." + hname.split(".layers.")[1].split(".")[0] + ".input") if ".layers." in hname else False


def trunk_keys(n_layers=64):
    return [k for l in range(n_layers) for _, ks in layer_plan(l) for k in ks]
