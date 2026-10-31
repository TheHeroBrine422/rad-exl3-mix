#!/usr/bin/env python3
"""damptest.py OUT.jsonl LAYERS KS SIGMAS [--device N] -- does a lighter GPTQ/LDLQ damping help our EXL3 quants?
(IQR-V3 §3.3: damping 0.05 -> 0.01 lowered their KL 3.5%.) For every linear of each layer, each K and each sigma_reg:
quantize with ourq.py's exact settings (mul1, output scales, same seed), and score the reconstruction against the
UNDAMPED calib Hessian: rel = tr(dW^T H dW) / tr(W^T H W) (exllamav3's own proxy uses the damped H, which would bias
the comparison). One JSON line per (tensor, K, sigma). e.g. damptest.py d.jsonl 2,3 3,5 0.025,0.005,0.001"""
import json, sys, time
import torch
sys.argv, rest = sys.argv[:1] + ["-K", "3", "-o", "/tmp/damptest-unused"] + sys.argv[5:], sys.argv[1:5]
import importlib.util, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ourq.py")).read()
src = src[:src.index("# ------------------------------------------------------------------ one layer's linears")] + \
      src[src.index("def layer_plan"):src.index("def quantize_group")]
g = {"__name__": "ourq_part", "__file__": os.path.join(here, "ourq.py")}
exec(compile(src, "ourq.py", "exec"), g)
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
OUT, LAYERS, KS, SIG = rest
out = open(OUT, "a"); dev = g["dev"]
for l in [int(x) for x in LAYERS.split(",")]:
    for hname, keys in g["layer_plan"](l):
        for j, key in enumerate(keys):
            w0 = g["ck"](key + ".weight").to(dev).float().T.contiguous()            # (in, out)
            for K in [int(x) for x in KS.split(",")]:
                for s in [float(x) for x in SIG.split(",")]:
                    Hd = g["hessian"](hname)                                         # fresh: quantize damps in place
                    H = Hd["H"].clone()
                    qa = {"seed": 1000 + j, "K": K, "devices": [g["a"].device], "device_ratios": None,
                          "apply_out_scales": True, "mul1": True, "sigma_reg": s}
                    t0 = time.time()
                    wq, perr, _ = quantize_exl3(w0.clone(), Hd, qa, True)
                    d = (wq.float() - w0)
                    rel = float((d * (H @ d)).sum() / (w0 * (H @ w0)).sum())
                    r = {"key": key, "layer": l, "K": K, "sigma_reg": s, "rel": rel, "exl_proxy": float(perr),
                         "s": round(time.time() - t0, 1)}
                    out.write(json.dumps(r) + "\n"); out.flush(); print(json.dumps(r), flush=True)
                    del Hd, H, wq, d; torch.cuda.empty_cache()
            del w0
