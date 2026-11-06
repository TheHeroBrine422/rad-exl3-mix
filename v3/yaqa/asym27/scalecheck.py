#!/usr/bin/env python3
"""scalecheck.py LEVEL KEY [KEY ...] -- why a level's decoded W (deq.py = the plugin oracle = what radiance serves) and
ourq's --save-wq file (the quantizer's W, scales in f32) differ: fit wq / W = r_i c_j (one factor per output row and
per input column: the fp16 rounding of svh / suh, which the shard stores and --save-wq does not) and report the
relative error before and after taking those factors out, and the share of subnormal fp16 scales. If the residual
falls to the floor of wq's own fp16 storage (~3e-4 rms), the gap is the stored scales' rounding, not the decode."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deq

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
lv = deq.Level(sys.argv[1], dev="cpu")
for key in sys.argv[2:]:
    W = lv.get(key).double()                                           # (out, in), served
    q = deq._wq(f"{sys.argv[1]}/wq", key).double()                      # (out, in), --save-wq
    rel0 = float((q - W).norm() / q.norm())
    m = W.abs() > W.abs().mean() * 0.25                                 # elements with a well-defined ratio
    L = torch.where(m, torch.log((q / W).abs().clamp_min(1e-30)), torch.zeros_like(W))
    a = torch.zeros(W.shape[0], dtype=torch.float64); b = torch.zeros(W.shape[1], dtype=torch.float64)
    cnt_r, cnt_c = m.sum(1).clamp_min(1), m.sum(0).clamp_min(1)
    for _ in range(20):                                                 # alternating least squares on log ratios
        a = ((L - b[None]) * m).sum(1) / cnt_r
        b = ((L - a[:, None]) * m).sum(0) / cnt_c
    Wc = W * torch.exp(a)[:, None] * torch.exp(b)[None]
    rel1 = float((q - Wc).norm() / q.norm())
    suh = lv._t(f"{key}.suh").float().abs(); svh = lv._t(f"{key}.svh").float().abs()
    sub = lambda s: float((s < 6.104e-5).float().mean())
    print(f"{key.split('layers.')[1]:32s} rel {rel0:.2e} -> {rel1:.2e} after row / column factors "
          f"(|log r| max {float(a.abs().max()):.1e}, |log c| max {float(b.abs().max()):.1e}); "
          f"subnormal fp16 scales: suh {sub(suh):.1%} (min {float(suh.min()):.1e}), svh {sub(svh):.1%} (min {float(svh.min()):.1e})",
          flush=True)
