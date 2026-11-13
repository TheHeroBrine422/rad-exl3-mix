#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""headexport.py OUTQ HEADDIR [--run RUN] [--ref WQ] [--dev cuda] -- a quantized head (ourq --only-head dir, `lm_head`)
with a QAD run's healed suh / svh (RUN/head/lm_head.safetensors, f16 as stored; no --run = the head unchanged), written
as OUTQ/K<k>/: the head dir's small files, q-head.safetensors (same trellis / mul1, the new suh / svh) and
wq/lm_head.safetensors {"weight": f16 [V, HID]} = rxq.deq's W^T in the stored row order: the input of
v3/yaqa/headslab.py build (bf16 rows un-permuted into a container's output.weight). --ref WQ: compare the written wq
with another one (the unhealed export vs ourq --save-wq's own: the path's identity check)."""
import argparse, json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("QAD_PIPELINE", os.path.join(HERE, "..", "..", "pipeline")))
from rxq import deq  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("outq"); ap.add_argument("head")
    ap.add_argument("--run", default=None); ap.add_argument("--ref", default=None); ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    lv = deq.Level(a.head, a.dev); raw = lv.raw("lm_head"); K = raw["trellis"].shape[2] / 16
    od = os.path.join(a.outq, f"K{K:g}"); os.makedirs(os.path.join(od, "wq"), exist_ok=True)
    wq = os.path.join(od, "wq", "lm_head.safetensors")
    if not os.path.exists(wq):
        suh, svh = raw["suh"], raw["svh"]
        if a.run:
            with safe_open(os.path.join(a.run, "head", "lm_head.safetensors"), "pt") as f:
                suh, svh = f.get_tensor("suh"), f.get_tensor("svh")
            assert suh.dtype == svh.dtype == torch.float16 and suh.shape == raw["suh"].shape and svh.shape == raw["svh"].shape
        for f in os.listdir(a.head):
            if os.path.isfile(os.path.join(a.head, f)) and f != "q-head.safetensors": shutil.copy2(os.path.join(a.head, f), od)
        with safe_open(os.path.join(a.head, "q-head.safetensors"), "pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}; meta = f.metadata()
        t["lm_head.suh"], t["lm_head.svh"] = suh.contiguous(), svh.contiguous()
        save_file(t, os.path.join(od, "q-head.safetensors.part"), metadata=meta)
        os.replace(os.path.join(od, "q-head.safetensors.part"), os.path.join(od, "q-head.safetensors"))
        W = deq.weight(raw["trellis"].to(a.dev), suh.to(a.dev), svh.to(a.dev), K, lv.table)      # f32 [HID, V]
        save_file({"weight": W.T.half().cpu().contiguous()}, wq + ".part"); os.replace(wq + ".part", wq)
        del W
        print(f"wrote {od} ({'healed by ' + a.run if a.run else 'unchanged scales'})")
    if a.ref:
        with safe_open(wq, "pt") as f: A = f.get_tensor("weight")
        with safe_open(a.ref, "pt") as f: B = f.get_tensor("weight")
        assert A.shape == B.shape, (A.shape, B.shape)
        eq = float((A.view(torch.int16) == B.view(torch.int16)).float().mean())
        d = (A.float() - B.float()).abs(); rel = float(d.norm() / B.float().norm())
        print(json.dumps({"ref": a.ref, "bit_equal_frac": eq, "max_abs": float(d.max()), "rel_fro": rel}))


if __name__ == "__main__":
    main()
