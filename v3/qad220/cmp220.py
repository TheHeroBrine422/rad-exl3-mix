#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmp220.py MODE ... -- w220's run comparisons (box2 / anywhere with torch + numpy):
  steps A B           per-step records of two runs' train.jsonl (A, B = run dirs): train_kl, gnorm, per-corpus KL, rows, lr -- bit for bit? (exit 0 = identical)
  state A B           the parameters of two runs' state.pt (max |diff|, count of differing tensors; exit 0 = identical)
  grad A B            per-scale mean gradients of two --gradnoise runs (npz of <name>|mean): max |diff|, max rel diff (|diff| / max|g| of the vector), median rel diff
  speed RUN [RUN ...] seconds per optimizer step of runs' train.jsonl: median over steps >= 3 (step 1-2 hold compile / warm-up), mean positions/s, peak GiB"""
import json, sys
import numpy as np


def recs(d): return [json.loads(l) for l in open(f"{d}/train.jsonl") if l.strip()]


def steps(a, b):
    ra, rb = [r for r in recs(a) if r["what"] == "main"], [r for r in recs(b) if r["what"] == "main"]
    n = min(len(ra), len(rb)); ok = n > 0
    for k in range(n):
        same = all(ra[k][f] == rb[k][f] for f in ("step", "rows", "lr", "train_kl", "by", "n", "gnorm"))
        ok &= same
        print(f"step {ra[k]['step']}: train_kl {ra[k]['train_kl']!r} / {rb[k]['train_kl']!r}  gnorm {ra[k]['gnorm']} / {rb[k]['gnorm']}  {'IDENTICAL' if same else 'DIFFERENT'}")
    print("ALL IDENTICAL" if ok else "DIFFERENT"); return 0 if ok else 1


def state(a, b):
    import torch
    za = torch.load(f"{a}/state.pt", weights_only=False)["snap"]["cur"]; zb = torch.load(f"{b}/state.pt", weights_only=False)["snap"]["cur"]
    assert za.keys() == zb.keys()
    nd = 0; mx = 0.0
    for k in za:
        d = (za[k].double() - zb[k].double()).abs().max().item() if za[k].numel() else 0.0
        nd += d > 0; mx = max(mx, d)
    print(f"{len(za)} tensors, {nd} differ, max |diff| {mx:.3e}"); return 0 if nd == 0 else 1


def grad(a, b):
    za, zb = np.load(f"{a}/gradnoise.npz"), np.load(f"{b}/gradnoise.npz")
    ka = sorted(k for k in za.files if k.endswith("|mean")); assert ka == sorted(k for k in zb.files if k.endswith("|mean")), "different trainable sets"
    mx = mr = 0.0; rel = []
    for k in ka:
        x, y = za[k].astype(np.float64), zb[k].astype(np.float64); s = np.abs(x).max() or 1.0
        d = np.abs(x - y); mx = max(mx, d.max()); mr = max(mr, d.max() / s); rel.append(d.max() / s)
    ja, jb = json.load(open(f"{a}/gradnoise.json")), json.load(open(f"{b}/gradnoise.json"))
    print(f"{len(ka)} vectors: max |diff| {mx:.3e}, max rel diff (to the vector's max |g|) {mr:.3e}, median over vectors {np.median(rel):.3e}")
    for x, y in zip(ja, jb): print(f"batch {x['step']}: KL/row {x['kl_row']!r} / {y['kl_row']!r}  gnorm {x['gnorm']:.8e} / {y['gnorm']:.8e}  rel {abs(x['gnorm'] / y['gnorm'] - 1):.2e}")
    return 0


def speed(*runs):
    for d in runs:
        r = [x for x in recs(d) if x["what"] == "main" and x["step"] >= 3]
        if not r: print(f"{d}: fewer than 3 steps"); continue
        w = [x["wall"] for x in r]
        print(f"{d}: {np.median(w):.1f} s / step (median of {len(w)}: {', '.join(f'{x:.1f}' for x in w)}), {np.mean([x['pos_s'] for x in r]):.0f} gen rows/s, peak {max(x['peak_gib'] for x in r):.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit({"steps": steps, "state": state, "grad": grad, "speed": speed}[sys.argv[1]](*sys.argv[2:]))
