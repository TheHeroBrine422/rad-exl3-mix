#!/usr/bin/env python3
"""selftest_gatecmp.py TMP -- gatecmp.py on synthetic REF / OFF / ON dirs: must PASS as built, FAIL when one trellis
word of ON differs, FAIL when ON's rowerr.npz is the old (regularized-weight) kind."""
import json, os, subprocess, sys
import numpy as np, torch
from safetensors.torch import save_file
T = sys.argv[1]; G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gatecmp.py")
def build(tweak=None, oldnpz=False):
    for d in ("REF", "OFF", "ON"):
        rng = np.random.default_rng(0)                 # OFF and ON are the same quant: same rows
        os.makedirs(f"{T}/{d}/wq", exist_ok=True)
        t = {"k.trellis": torch.arange(64, dtype=torch.int16), "k.suh": torch.ones(8, dtype=torch.float16)}
        if tweak == d: t["k.trellis"][3] += 1
        save_file(t, f"{T}/{d}/q-L00.safetensors", metadata={"ourq_stamp": d})
        err, ref = rng.random(16) * 0.01, rng.random(16) + 1
        if d == "REF" or (oldnpz and d == "ON"): err = ref * (1 + rng.random(16) * 1e-3)
        np.savez(f"{T}/{d}/wq/k.rowerr.npz", err=err, ref=ref)
        e1 = float(err.sum() / ref.sum())
        with open(f"{T}/{d}/proxy.jsonl", "w") as f:
            f.write(json.dumps({"key": "k", "proxy_err": e1 * (1 + 1e-5), "proxy_err1": e1, "proxy_err2": e1 * 0.9}) + "\n")
def run(): return subprocess.run([sys.executable, G, f"{T}/REF", f"{T}/OFF", f"{T}/ON"], capture_output=True, text=True).stdout
res = []
build(); o = run(); print(o); res.append("GATE PASS" in o)
build("ON"); o = run(); print(o); res.append("GATE FAIL" in o)
build(None, True); o = run(); print(o); res.append("GATE FAIL" in o)
print("SELFTEST gatecmp", "PASS" if all(res) else f"FAIL {res}")
