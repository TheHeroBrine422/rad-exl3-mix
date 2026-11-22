#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""precond218.py GDIR OUT [--sigma 0.025] -- #218: the Hessian-preconditioned gradient P(G) = M_d^-1 G per probe tensor (M_d = the stored
codes' metric Hb S H S Hb, S = diag(suh), H the calib Hessian / rows, + sigma mean diag(M) I), the same for every output column (the
input-side Newton step of the layer-wise quadratic model). Writes OUT/<key>.g + grad.json (g_rms of P(G)) in encode218.py's grad format.
On the card (f64 Cholesky). Why: a plain step along G lies in M's large-eigenvalue directions (G = X'^T dY), so its output-error cost
tr(D^T M D) is huge for its size; M^-1 G is the step of least M-cost for a given first-order KL change."""
import json, os, sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
sys.path.insert(0, HERE)
from rxq import deq  # noqa: E402
from encode218 import hname  # noqa: E402

LV = "/data/w218/lv/2.50/trunk"; HESS = "/data/challenge/hessians/calib"
gdir, out = sys.argv[1], sys.argv[2]
sigma = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 0.025
os.makedirs(out, exist_ok=True)
gm = json.load(open(f"{gdir}/grad.json"))
lev = deq.Level(LV, "cpu"); dev = torch.device("cuda", 0)
Hb = deq.hadamard128(dev).double()
meta = {}
for key, v in gm["tensors"].items():
    k, n = v["k"], v["n"]
    with safe_open(f"{HESS}/{hname(key)}.safetensors", "pt") as f:
        up = f.get_tensor("hessian_upper").to(dev).double(); rows = float(f.metadata()["rows"])
    H = torch.zeros(k, k, dtype=torch.float64, device=dev); iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up; del up, iu
    H = H + H.T - torch.diag(torch.diagonal(H)); H /= rows
    s = lev.raw(key)["suh"].double().to(dev)
    M = H * s[:, None] * s[None, :]; del H
    M = (Hb @ M.reshape(k // 128, 128, k)).reshape(k, k)
    M = (M.reshape(k, k // 128, 128) @ Hb).reshape(k, k)
    M.diagonal().add_(sigma * float(M.diagonal().mean()))
    L = torch.linalg.cholesky(M); del M
    G = torch.from_numpy(np.fromfile(f"{gdir}/{key}.g", np.float32).reshape(k, n)).to(dev).double()
    P = torch.cholesky_solve(G, L).float().cpu().numpy(); del L, G
    P.tofile(f"{out}/{key}.g")
    meta[key] = {**v, "g_rms": float(np.sqrt((P.astype(np.float64) ** 2).mean()))}
    torch.cuda.empty_cache()
    print(key, meta[key]["g_rms"], flush=True)
json.dump({**{x: gm[x] for x in gm if x != "tensors"}, "precond": f"M_d^-1, sigma {sigma}", "from": gdir, "tensors": meta},
          open(f"{out}/grad.json.part", "w"))
os.replace(f"{out}/grad.json.part", f"{out}/grad.json")
