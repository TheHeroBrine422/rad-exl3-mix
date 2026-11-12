#!/usr/bin/env python3
"""bytes113.py CENSUS_PY MIX_N MIX_T CK_T -- w113 #148-02 gate G2 (../w113.md; w111 bytes111.py for the T arm): in mix-T
every patched weight (CK_T/patch.json keys) is one bf16 plane equal byte for byte to CK_T/w113-patch.safetensors, it
differs from the same weight's plane in mix-N (most bytes), and both carry the same quant label. Prints BYTES113 PASS /
FAIL (last line)."""
import importlib.util, json, sys

import numpy as np
import torch
from safetensors import safe_open

cp, mn, mt, ckt = sys.argv[1:5]
sp = importlib.util.spec_from_file_location("census", cp); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
RAD = {"in_proj_qkv": "ssm_qkv", "in_proj_z": "ssm_z", "q_proj": "attn_qg", "v_proj": "attn_v"}
keys = json.load(open(f"{ckt}/patch.json"))["keys"]


def rad_name(key):
    return f"blk.{int(key.split('.')[3])}.{RAD[key.split('.')[-1]]}.weight"


def plane(R, n):
    w = R.w[n]; assert len(w["planes"]) == 1, (n, len(w["planes"]))
    po, pn = w["planes"][0]; R.f.seek(po); return np.frombuffer(R.f.read(pn), np.uint16), w


u16 = lambda t: t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
ok, out = True, []
with safe_open(f"{ckt}/w113-patch.safetensors", "pt") as ST:
    Rn, Rt = C.Rad(mn), C.Rad(mt)
    for key in keys:
        n = rad_name(key); at = u16(ST.get_tensor(key + ".weight"))
        pn_, wn = plane(Rn, n); pt_, wt = plane(Rt, n)
        eqt = pt_.size == at.size and bool((pt_ == at).all())
        frac = float((pn_ != pt_).mean()) if pn_.size == pt_.size else 1.0
        good = eqt and frac > 0.5 and wn["quant"] == wt["quant"]
        ok &= good
        out.append({"weight": n, "quant": wt["quant"], "t_eq_patch": eqt, "frac_differ_vs_N": frac})
        print(n, wt["quant"], "T==patch" if eqt else "T!=patch", f"N/T differ {frac:.3f}", flush=True)
json.dump(out, open(f"{ckt}/bytes113.json", "w"), indent=1)
print(f"BYTES113 {'PASS' if ok else 'FAIL'} ({len(out)} weights)")
