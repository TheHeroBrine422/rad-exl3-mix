#!/usr/bin/env python3
"""qprof.py K LAYER OUT.json -- where our quantizer's time goes: one layer of ourq.py at K, with exllamav3's stage
functions wrapped in synchronized timers (the module globals quantize_exl3 calls, so the wrap catches every call).
Run in $RXM_VENV from $RXM_WORK (ourq/ourq.py on the path). Stage times are inclusive (finalize_capture_H contains
block_ldl), so shares overlap; quantize_exl3 (whole) vs wall is the time outside exllamav3 (Hessian load, I/O)."""
import json, os, shutil, sys, time, collections
import torch
import exllamav3.modules.quant.exl3_lib.quantize as Q

K, L, out = sys.argv[1], sys.argv[2], sys.argv[3]
tot = collections.Counter(); cnt = collections.Counter()
def wrap(name):
    f = getattr(Q, name, None)
    if f is None: return
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize(); tot[name] += time.perf_counter() - t; cnt[name] += 1
        return r
    setattr(Q, name, w)
for n in ["finalize_capture_H", "block_ldl", "regularize", "ldlq", "ldlq_batched", "quantize_tiles", "quantize_tiles_multigpu",
          "quantize_tiles_multigpu_sync", "g_scale_gss", "g_scale_search_batch", "refit_scales", "pack_trellis",
          "preapply_had_l", "preapply_had_r", "blockwise_preapply_had_l_", "blockwise_preapply_had_r_", "fallback_quant"]:
    wrap(n)
qx = Q.quantize_exl3
def qx_t(*a, **k):
    torch.cuda.synchronize(); t = time.perf_counter(); r = qx(*a, **k); torch.cuda.synchronize()
    tot["quantize_exl3 (whole)"] += time.perf_counter() - t; cnt["quantize_exl3 (whole)"] += 1; return r
Q.quantize_exl3 = qx_t
# ourq imports quantize_exl3 by name: patch before it runs
shutil.rmtree(f"/tmp/qprof-K{K}", ignore_errors=True)
sys.argv = ["ourq.py", "-K", K, "-o", f"/tmp/qprof-K{K}", "--layers", f"{L}-{L}"]
t0 = time.perf_counter()
src = open("ourq/ourq.py").read().replace("from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3",
                                           "quantize_exl3 = __import__('exllamav3.modules.quant.exl3_lib.quantize', fromlist=['x']).quantize_exl3")
exec(compile(src, "ourq/ourq.py", "exec"), {"__name__": "__main__"})
wall = time.perf_counter() - t0
rep = {"K": int(K), "layer": int(L), "wall_s": round(wall, 1),
       "stages": {n: {"s": round(tot[n], 2), "calls": cnt[n], "share": round(tot[n] / wall, 3)} for n in sorted(tot, key=lambda x: -tot[x])}}
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps(rep, indent=1))
