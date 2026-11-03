#!/usr/bin/env python3
"""moments.py stub: tiny H / C / N files for the requested layers and mode (N's size: quant 0.04, sib 0.01, noise
0.003 x (gamma / 0.01)^2)"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest", "stubs"))
from common_stub import groups, put
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
for o in ("--mode", "--wq", "--layers", "--pools", "--tokens", "--win", "--gamma", "--seed", "--sigma", "--sigma-tokens"): ap.add_argument(o)
a = ap.parse_args(); lo, hi = map(int, a.layers.split("-"))
if a.mode != "noise": assert os.path.isdir(a.wq), a.wq
if a.sigma and not os.path.exists(a.sigma): os.makedirs(os.path.dirname(a.sigma), exist_ok=True); torch.save({}, a.sigma)
nv = {"quant": 0.04, "sib": 0.01}.get(a.mode) or 0.003 * (float(a.gamma) / 0.01) ** 2
for l in range(lo, hi + 1):
    for i, g in enumerate(groups(l)):
        if a.mode == "sib" and i in (0, 2): continue
        put(f"{a.out}/{g}.safetensors", {"H": torch.eye(2), "C": -0.5 * nv * torch.eye(2), "N": nv * torch.eye(2)}, {"n": "1"})
print("MOMENTS DONE stub")
