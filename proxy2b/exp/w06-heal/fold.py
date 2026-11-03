#!/usr/bin/env python3
"""fold.py PARAMS SRC_WQ OUT CKPT [--jitter SD] -- write a healed arm in evalq.py's inputs.

OUT/wq/<key>.safetensors: each healed linear's dequantized weight with its multipliers folded in,
W' = diag(e^c) W diag(e^r) ((out, in) layout, fp32 product, stored f16 as ourq --save-wq); unhealed keys are symlinks
to SRC_WQ. If PARAMS trains norms: OUT/model = the checkpoint with those norm tensors replaced (gain' = gain * e^g;
Qwen3.5's gain is 1 + w), every other file a symlink. evalq.py OUT/model GOLDEN --wq OUT/wq then scores the arm.
--jitter SD (PARAMS '-'): random r, c ~ N(0, SD) on every linear instead -- the control for re-rounding every
weight (evalq casts the f16 weights to bf16)."""
import argparse, json, os
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

def fold_wq(lin, src, dst):
    os.makedirs(dst, exist_ok=True); n = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith(".safetensors"): continue
        key = f[:-len(".safetensors")]; p = lin.get(key); o = os.path.join(dst, f)
        if os.path.lexists(o): os.remove(o)
        if p is None or (p.get("r") is None and p.get("c") is None):
            os.symlink(os.path.realpath(os.path.join(src, f)), o); continue
        W = load_file(os.path.join(src, f))["weight"].float()
        if p.get("c") is not None: W *= p["c"].double().exp().float()[:, None]
        if p.get("r") is not None: W *= p["r"].double().exp().float()[None, :]
        assert torch.isfinite(W).all() and W.abs().max() < 65504, key
        save_file({"weight": W.half().contiguous()}, o + ".part"); os.replace(o + ".part", o); n += 1
    open(os.path.join(dst, ".done"), "w").close()
    return n

def new_gain(w, g, form):
    w32 = w.float(); m = g.double().exp().float()
    return ((1 + w32) * m - 1 if form == "1+w" else w32 * m).to(w.dtype)

def fold_model(norm, form, ckpt, dst):
    """copy of the checkpoint with the norm tensors replaced (shards holding them rewritten, the rest symlinked)"""
    os.makedirs(dst, exist_ok=True)
    wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    touched = {wm[k] for k in norm}
    for f in sorted(os.listdir(ckpt)):
        o = os.path.join(dst, f)
        if os.path.lexists(o): os.remove(o)
        if f not in touched: os.symlink(os.path.realpath(os.path.join(ckpt, f)), o); continue
        with safe_open(os.path.join(ckpt, f), "pt") as h:
            md = h.metadata(); t = {k: h.get_tensor(k) for k in h.keys()}
        for k, g in norm.items():
            if k in t: assert t[k].shape == g.shape, k; t[k] = new_gain(t[k], g, form)
        save_file(t, o + ".part", metadata=md); os.replace(o + ".part", o)
    open(os.path.join(dst, ".done"), "w").close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("ckpt")
    ap.add_argument("--jitter", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.jitter:
        g = torch.Generator().manual_seed(a.seed); lin = {}
        for f in sorted(os.listdir(a.src)):
            if not f.endswith(".safetensors"): continue
            with safe_open(os.path.join(a.src, f), "pt") as h: o_, i_ = h.get_slice("weight").get_shape()
            lin[f[:-12]] = {"r": torch.randn(i_, generator=g) * a.jitter, "c": torch.randn(o_, generator=g) * a.jitter}
        st = {"lin": lin, "norm": {}, "norm_form": None}
    else:
        st = torch.load(a.params, weights_only=False)
    n = fold_wq(st["lin"], a.src, os.path.join(a.out, "wq"))
    if st["norm"]: fold_model(st["norm"], st["norm_form"], a.ckpt, os.path.join(a.out, "model"))
    print(f"folded {n} linears, {len(st['norm'])} norms -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
