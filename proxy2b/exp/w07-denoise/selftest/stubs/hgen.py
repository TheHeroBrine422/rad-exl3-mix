#!/usr/bin/env python3
"""hgen.py stub: layer-0 check mode copies the calib files (the gate passes); otherwise tiny files"""
import argparse, os, shutil, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w07-denoise", "selftest", "stubs"))
from common_stub import groups, put
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
for o in ("--weight", "--pools", "--exclude", "--chunk"): ap.add_argument(o)
ap.add_argument("--layers", default="0-23")
a = ap.parse_args(); lo, hi = map(int, a.layers.split("-")); H = os.path.join(os.path.dirname(a.meta), "h08")
for e in filter(None, (a.exclude or "").split(",")): assert os.path.exists(f"{e}/kld.json"), e
os.makedirs(a.out, exist_ok=True)
for l in range(lo, hi + 1):
    for g in groups(l):
        if a.weight == "all": shutil.copy(f"{H}/{g}.safetensors", f"{a.out}/{g}.safetensors")
        else: put(f"{a.out}/{g}.safetensors", {"hessian_upper": torch.tensor([0.2, 0.0, 0.2])}, {"k": "2", "rows": "200"})
print("HGEN DONE stub")
