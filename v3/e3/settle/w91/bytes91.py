#!/usr/bin/env python3
"""bytes91.py CENSUS_PY MIX_A MIX_C PATCHDIR -- ../w91.md gate G1 (byte half): the dense writers in the arm and control
mixes. For every key in PATCHDIR/patch.json: each mix holds the weight as one bf16 plane equal byte for byte to its
patch shard (ck-A / ck-C), and A and C differ only inside the patched rows. Prints BYTES91 PASS / FAIL."""
import importlib.util, json, sys

import numpy as np
import torch
from safetensors.torch import load_file

cp, ma, mc, pd = sys.argv[1:5]
sp = importlib.util.spec_from_file_location("census", cp); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
P = json.load(open(f"{pd}/patch.json")); rows = set(P["rows"])
A, Cm = C.Rad(ma), C.Rad(mc)
SA = load_file(f"{pd}/ck-A/w91-patch.safetensors"); SC = load_file(f"{pd}/ck-C/w91-patch.safetensors")


def rad_name(key):
    l = int(key.split(".")[3]); f = key.split(".")[-1]
    return f"blk.{l}." + {"out_proj": "ssm_out", "o_proj": "attn_output", "down_proj": "ffn_down"}[f] + ".weight"


def plane(R, n):
    w = R.w[n]; assert len(w["planes"]) == 1, (n, len(w["planes"]))
    po, pn = w["planes"][0]; R.f.seek(po); return np.frombuffer(R.f.read(pn), np.uint16), w


ok, out = True, []
for key in P["keys"]:
    n = rad_name(key)
    pa, wa = plane(A, n); pc, wc = plane(Cm, n)
    u16 = lambda t: t.contiguous().view(torch.int16).numpy().view(np.uint16)
    ta = u16(SA[key + ".weight"]).reshape(-1); tc = u16(SC[key + ".weight"]).reshape(-1)
    shp = tuple(SA[key + ".weight"].shape)
    eqa, eqc = pa.size == ta.size and bool((pa == ta).all()), pc.size == tc.size and bool((pc == tc).all())
    d = sorted(set(np.nonzero((pa.reshape(shp) != pc.reshape(shp)).any(1))[0].tolist())) if pa.size == pc.size else ["size"]
    good = eqa and eqc and set(d) <= rows and wa["quant"] == wc["quant"]
    ok &= good
    out.append({"weight": n, "quant": wa["quant"], "opts": wa["opts"], "a_eq_patch": eqa, "c_eq_patch": eqc, "rows_differ": d})
    print(n, wa["quant"], "A==patch" if eqa else "A!=patch", "C==patch" if eqc else "C!=patch", "rows differ", d, flush=True)
json.dump(out, open(f"{pd}/bytes91.json", "w"), indent=1)
print(f"BYTES91 {'PASS' if ok else 'FAIL'} ({len(out)} weights)")
