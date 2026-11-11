#!/usr/bin/env python3
"""bytes111.py CENSUS_PY MIX_N MIX_P CK_N CK_P -- w111 #146-02 gate G2 (../w111.md; = w101 bytes101.py for the 16 attn_v): in each spliced mix
every patched weight (CK_X/patch.json keys) is one bf16 plane equal byte for byte to CK_X/w111-patch.safetensors, and
the N and P tensors differ (the un-permuted P codes are not N's). Prints BYTES111 PASS / FAIL (last line)."""
import importlib.util, json, sys

import numpy as np
import torch
from safetensors import safe_open

cp, mn, mp, cn, ckp = sys.argv[1:6]
sp = importlib.util.spec_from_file_location("census", cp); C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
RAD = {"in_proj_qkv": "ssm_qkv", "in_proj_z": "ssm_z", "q_proj": "attn_qg", "v_proj": "attn_v"}
keys = json.load(open(f"{cn}/patch.json"))["keys"]
assert keys == json.load(open(f"{ckp}/patch.json"))["keys"] and len(keys) == 16


def rad_name(key):
    return f"blk.{int(key.split('.')[3])}.{RAD[key.split('.')[-1]]}.weight"


def plane(R, n):
    w = R.w[n]; assert len(w["planes"]) == 1, (n, len(w["planes"]))
    po, pn = w["planes"][0]; R.f.seek(po); return np.frombuffer(R.f.read(pn), np.uint16), w


u16 = lambda t: t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
ok, out = True, []
with safe_open(f"{cn}/w111-patch.safetensors", "pt") as SN, safe_open(f"{ckp}/w111-patch.safetensors", "pt") as SP:
    Rn, Rp = C.Rad(mn), C.Rad(mp)
    for key in keys:
        n = rad_name(key); tn, tp = SN.get_tensor(key + ".weight"), SP.get_tensor(key + ".weight")
        pn_, wn = plane(Rn, n); pp_, wp = plane(Rp, n); an, ap = u16(tn), u16(tp)
        eqn = pn_.size == an.size and bool((pn_ == an).all()); eqp = pp_.size == ap.size and bool((pp_ == ap).all())
        frac = float((an != ap).mean())
        good = eqn and eqp and frac > 0.5 and wn["quant"] == wp["quant"]
        ok &= good
        out.append({"weight": n, "quant": wn["quant"], "n_eq_patch": eqn, "p_eq_patch": eqp, "frac_differ": frac})
        print(n, wn["quant"], "N==patch" if eqn else "N!=patch", "P==patch" if eqp else "P!=patch", f"N/P differ {frac:.3f}", flush=True)
json.dump(out, open(f"{ckp}/bytes111.json", "w"), indent=1)
print(f"BYTES111 {'PASS' if ok else 'FAIL'} ({len(out)} weights)")
