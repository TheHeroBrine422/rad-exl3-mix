#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmp233.py RUN_A RUN_B -- w233 equivalence report of two qad220 runs on the same plan: per step train_kl / gnorm / rows (equal? max relative
difference), wall per step (median of steps >= 3), and the parameters in OUT/state.pt (snap cur: max |diff|, how many tensors differ)."""
import json, sys
import numpy as np
import torch

A, B = sys.argv[1:3]
def rec(d): return {r["step"]: r for r in (json.loads(l) for l in open(f"{d}/train.jsonl")) if r.get("what") == "main"}
ra, rb = rec(A), rec(B); st = sorted(set(ra) & set(rb))
kl = [(ra[s]["train_kl"], rb[s]["train_kl"]) for s in st]; gn = [(ra[s]["gnorm"], rb[s]["gnorm"]) for s in st]
rel = lambda x: max(abs(a / b - 1) if b else abs(a - b) for a, b in x)
print(f"steps compared: {len(st)} ({st[0]}..{st[-1]})")
print(f"train_kl: identical at {sum(a == b for a, b in kl)} / {len(kl)} steps, max rel diff {rel(kl):.3e}; rows equal: {all(ra[s]['rows'] == rb[s]['rows'] for s in st)}")
print(f"gnorm (logged to 6 digits): identical at {sum(a == b for a, b in gn)} / {len(gn)}, max rel diff {rel(gn):.3e}")
wa = [ra[s]["wall"] for s in st if s >= 3]; wb = [rb[s]["wall"] for s in st if s >= 3]
print(f"wall s / step (median of steps >= 3): {np.median(wa):.2f} vs {np.median(wb):.2f} -> {np.median(wa) / np.median(wb):.3f}x; total {sum(wa):.1f} vs {sum(wb):.1f} s")
za, zb = torch.load(f"{A}/state.pt", weights_only=False)["snap"]["cur"], torch.load(f"{B}/state.pt", weights_only=False)["snap"]["cur"]
assert set(za) == set(zb), (len(za), len(zb))
d = {k: float((za[k].double() - zb[k].double()).abs().max()) for k in za}
print(f"parameters: {len(za)} tensors, {sum(v == 0 for v in d.values())} bit for bit, max |diff| {max(d.values()):.3e} (exp(r) scale: {max(float((za[k].exp() - zb[k].exp()).abs().max()) for k in za):.3e})")
