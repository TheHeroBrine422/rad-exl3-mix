#!/usr/bin/env python3
"""hblend.py HGRAD_DIR HPLAIN_DIR OUT_DIR ALPHA -- gate 2e MARGINAL follow-up: blend the gradient-weighted input
Hessian (collect.py YAQA_HIN, ~200k tokens, noisy) with the published plain one (10M tokens) to cut its noise:
  H = (1 - ALPHA) * H_plain + ALPHA * H_grad * tr(H_plain) / tr(H_grad)
(trace-matched, so ALPHA is the share of the gradient weighting, not of the scale). rows = H_plain's, so ourq.py's
normalisation and damping see the plain Hessian's scale. Only the groups HGRAD_DIR has; challenge layout out."""
import json, os, struct, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
HG, HP, OUT, AL = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
def load(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").double()
    return up, int(meta["k"]), int(meta["rows"])
for fn in sorted(os.listdir(HG)):
    if not fn.endswith(".safetensors"): continue
    g, k, _ = load(f"{HG}/{fn}"); p, k2, rows = load(f"{HP}/{fn}"); assert k == k2, fn
    di = torch.arange(k, dtype=torch.long); di = di * k - di * (di - 1) // 2      # diagonal positions in the packed upper
    tg, tp = float(g[di].sum()), float(p[di].sum())
    h = (1 - AL) * p + AL * g * (tp / tg)
    save_file({"hessian_upper": h.float().contiguous()}, f"{OUT}/{fn}",
              metadata={"k": str(k), "rows": str(rows), "source": f"hblend alpha {AL}"})
    print(f"{fn}: k {k}, tr grad/plain scale {tp / tg:.3g}", flush=True)
    del g, p, h
