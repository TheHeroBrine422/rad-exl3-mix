#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkmask220.py GRADNOISE.npz OUT.npz [--n 40] [--z 3] [--json SUMMARY.json] -- the signal mask of a w217 gradient-noise pass
(qad200 / qad220 --gradnoise N: keys <name>|mean and <name>|std, the per-scale mean and across-batch std (ddof 1) of N batches'
gradients). Scale kept (1) iff |mean| > Z * std / sqrt(N), else 0. OUT.npz: key <name> -> uint8 [numel] (the trainer's --mask).
The summary (per layer: scales kept, share) goes to stdout and optionally --json."""
import argparse, json, re
import numpy as np


def build(z, n, zc):
    """{name: uint8 mask} from the npz-like mapping z of <name>|mean / <name>|std"""
    out = {}
    for k in z.files if hasattr(z, "files") else z:
        if not k.endswith("|mean"): continue
        nm = k[:-5]
        mean = np.asarray(z[k], np.float64); std = np.asarray(z[nm + "|std"], np.float64)
        out[nm] = (np.abs(mean) > zc * std / np.sqrt(n)).astype(np.uint8)
    return out


def summary(mask):
    by = {}
    for nm, m in mask.items():
        r = re.search(r"\.layers\.(\d+)\.", nm); L = int(r.group(1)) if r else -1
        v = by.setdefault(L, [0, 0]); v[0] += int(m.sum()); v[1] += m.size
    return {str(L): {"kept": v[0], "scales": v[1], "share": v[0] / v[1]} for L, v in sorted(by.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gradnoise"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40, help="number of batches of the gradnoise pass")
    ap.add_argument("--z", type=float, default=3.0); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    z = np.load(a.gradnoise)
    mask = build(z, a.n, a.z)
    np.savez(a.out, **mask)
    sm = summary(mask)
    kept = sum(v["kept"] for v in sm.values()); tot = sum(v["scales"] for v in sm.values())
    print(f"{len(mask)} vectors, {kept} of {tot} scales kept ({kept / tot:.4f}) at |mean| > {a.z} std / sqrt({a.n})")
    for L, v in sm.items(): print(f"layer {L}: {v['kept']} / {v['scales']} ({v['share']:.4f})")
    if a.json: json.dump({"n": a.n, "z": a.z, "kept": kept, "scales": tot, "layers": sm}, open(a.json, "w"), indent=1)
