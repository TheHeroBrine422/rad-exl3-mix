#!/usr/bin/env python3
"""c9g1.py A B [C] -- #145 gate G1 (../w110.md, amendments 1-2): every writer's C9 scale file in A/scales within 1e-5
relative of B/scales' on "c" and equal on "r" (A = rxq.post c9 re-run with today's settings on the CPU, B = the
level's in-job LEVEL/c9 from the GPU: bit-identity is not reachable across the two). With C (the e3
dir): also prints how many writers' c differ between B and C and the largest relative change (e3 = cur at a level
where no row reaches e 0.3). Last line "G1 PASS {json}" / "G1 FAIL {json}"; exit 1 on FAIL."""
import json, os, sys
import torch
from safetensors.torch import load_file

a, b = sys.argv[1], sys.argv[2]; c = sys.argv[3] if len(sys.argv) > 3 else None
fa = sorted(f for f in os.listdir(f"{a}/scales") if f.endswith(".safetensors"))
fb = sorted(f for f in os.listdir(f"{b}/scales") if f.endswith(".safetensors"))
st = {"writers_a": len(fa), "writers_b": len(fb), "unequal": [], "bit_identical": 0, "max_rel": 0.0}
ok = fa == fb and len(fa) > 0
for f in fa if ok else []:
    x, y = load_file(f"{a}/scales/{f}"), load_file(f"{b}/scales/{f}")
    rel = float(((x["c"].double() - y["c"].double()).abs() / y["c"].double().abs()).max())
    st["max_rel"] = max(st["max_rel"], rel); st["bit_identical"] += int(torch.equal(x["c"], y["c"]))
    if not (x.keys() == y.keys() and torch.equal(x["r"], y["r"]) and rel <= 1e-5): st["unequal"].append(f[:-12])
ok = ok and not st["unequal"]
if c:
    fc = sorted(f for f in os.listdir(f"{c}/scales") if f.endswith(".safetensors"))
    assert fc == fb, "e3 dir has another writer set"
    nd, mx = 0, 0.0
    for f in fc:
        x, y = load_file(f"{b}/scales/{f}")["c"].double(), load_file(f"{c}/scales/{f}")["c"].double()
        if not torch.equal(x, y): nd += 1; mx = max(mx, float(((y - x).abs() / x.abs()).max()))
    st["e3_writers_differ"] = nd; st["e3_max_rel"] = mx
print(f"G1 {'PASS' if ok else 'FAIL'} {json.dumps(st)}")
sys.exit(0 if ok else 1)
