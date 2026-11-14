#!/usr/bin/env python3
"""abtrellis-mad.py CKPT HESS OUT [-K 4,5] [--codebook mad] -- #177-03 (w151): search/v3/ab178/abtrellis.py (w142) with the
codebook as an argument (default mad: the a/b T4 / T5 donors of #178-02 / #178-03 requantized with the ladder's codebook;
the settings file's [quant.codebook.mad] gives A / B). Otherwise unchanged. Original docstring:
abtrellis.py CKPT HESS OUT [-K 4,5,6] -- arms T4 / T5 / T6 of RULES-178: every GDN layer's in_proj_a | in_proj_b as one
fused weight padded to 128 rows (rows 0-47 a, 48-95 b, 96-127 a copy of a's first 32 rows: exllamav3's LDLQ needs
N % 128 = 0; the padding is decoded and dropped), quantized by rxq quant (config/v3.toml, codebook i8) with --save-wq.
The fused key joins the layer-input group (qwen35.layer_plan patched here: after qkv / z, so j = 2 and the group's su
draw is qkv's). No H_out exists for a/b: inputs.hout.enable = false, quant.refit = true (the plain refit).
Writes OUT/ckpt (the fused weights, an index), OUT/K<K>/ (rxq's job output) and OUT/ab-T<K>.safetensors
({blk.N.ssm_a.weight, blk.N.ssm_b.weight}: bf16 [48, 5120]). Run with env-ds8.sh, ROCR_VISIBLE_DEVICES = the card."""
import argparse, json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("hess"); p.add_argument("out"); p.add_argument("-K", default="4,5,6")
p.add_argument("--layers", default="0-63")
p.add_argument("--codebook", default="mad")
a = p.parse_args()
L = "model.language_model.layers"
lo, hi = map(int, a.layers.split("-"))
layers = [l for l in range(lo, hi + 1) if l % 4 != 3]
key = lambda l: f"{L}.{l}.linear_attn.in_proj_ab"

ck_dir = f"{a.out}/ckpt"; os.makedirs(ck_dir, exist_ok=True)
if not os.path.exists(f"{ck_dir}/model.safetensors.index.json"):
    idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
    t = {}
    for l in layers:
        ws = []
        for ab in ("a", "b"):
            n = f"{L}.{l}.linear_attn.in_proj_{ab}.weight"
            with safe_open(f"{a.ckpt}/{idx[n]}", "pt") as f: ws.append(f.get_tensor(n))
        assert ws[0].shape == ws[1].shape == (48, 5120), [w.shape for w in ws]
        t[key(l) + ".weight"] = torch.cat([ws[0], ws[1], ws[0][:32]]).contiguous()
    save_file(t, f"{ck_dir}/model.safetensors.part"); os.replace(f"{ck_dir}/model.safetensors.part", f"{ck_dir}/model.safetensors")
    json.dump({"weight_map": {k: "model.safetensors" for k in t}}, open(f"{ck_dir}/model.safetensors.index.json", "w"))

from rxq import qwen35, __main__ as rxmain
_plan = qwen35.layer_plan
def layer_plan(l):
    plan = _plan(l)
    if not qwen35.is_full_attention(l): plan[0] = (plan[0][0], plan[0][1] + [key(l)])
    return plan
qwen35.layer_plan = layer_plan

rxmain.main(["quant", "--keys", ",".join(key(l) for l in layers), "-K", a.K, "--codebook", a.codebook, "--ckpt", ck_dir,
             "--hess", a.hess, "-o", f"{a.out}/K{{K}}", "--save-wq", "--no-prof",
             "--set", "inputs.hout.enable=false", "--set", "quant.refit=true"])

if (lo, hi) == (0, 63):
    for K in a.K.split(","):
        t = {}
        for l in layers:
            with safe_open(f"{a.out}/K{K}/wq/{key(l)}.safetensors", "pt") as f: w = f.get_tensor("weight")
            assert w.shape == (128, 5120), w.shape
            t[f"blk.{l}.ssm_a.weight"] = w[:48].to(torch.bfloat16).contiguous()
            t[f"blk.{l}.ssm_b.weight"] = w[48:96].to(torch.bfloat16).contiguous()
        save_file(t, f"{a.out}/ab-T{K}.safetensors")
        print("wrote", f"{a.out}/ab-T{K}.safetensors", flush=True)
