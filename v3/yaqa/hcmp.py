#!/usr/bin/env python3
"""hcmp.py DIR_A DIR_B -- relative Frobenius difference of every H_out in both dirs (truncated vs full backward)."""
import os, sys, torch
from safetensors import safe_open
A, B = sys.argv[1:3]; worst = 0
for f in sorted(os.listdir(A)):
    if not f.endswith(".safetensors") or not os.path.exists(f"{B}/{f}"): continue
    with safe_open(f"{A}/{f}", "pt") as x, safe_open(f"{B}/{f}", "pt") as y:
        a, b = x.get_tensor("hout").double(), y.get_tensor("hout").double()
    d = float((a - b).norm() / b.norm()); worst = max(worst, d)
    print(f"{f[28:-12]:45s} rel diff {d:.2e}")
print(f"HCMP worst {worst:.2e} {'OK' if worst < 1e-3 else 'MISMATCH'}")
sys.exit(0 if worst < 1e-3 else 1)
