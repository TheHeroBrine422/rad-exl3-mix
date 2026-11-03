#!/usr/bin/env python3
"""mkdenoise.py stub: hin (and wpre unless lam = rho) files plus stats lines, one per moment file"""
import argparse, glob, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest", "stubs"))
from common_stub import put
ap = argparse.ArgumentParser(); ap.add_argument("calib"); ap.add_argument("mom"); ap.add_argument("out")
for o in ("--lam", "--rho", "--sig", "--sig-in", "--ckpt"): ap.add_argument(o)
a = ap.parse_args(); fs = sorted(glob.glob(f"{a.mom}/*.input.safetensors")); assert fs
os.makedirs(f"{a.out}/hin", exist_ok=True); st = open(f"{a.out}/stats.jsonl", "w")
for f in fs:
    g = os.path.basename(f)[:-12]
    assert os.path.exists(f"{a.calib}/{g}.safetensors"), g
    put(f"{a.out}/hin/{g}.safetensors", {"hessian_upper": torch.ones(3)}, {"k": "2", "rows": "1000"})
    r = {"group": g, "clip": 0.0, "hsub_vs_calib": 0.05}
    if a.lam != a.rho: put(f"{a.out}/wpre/{g}.safetensors", {"Ft": torch.eye(2)}); r["wf_x"] = 0.99
    st.write(json.dumps(r) + "\n")
print("MKDENOISE DONE stub")
