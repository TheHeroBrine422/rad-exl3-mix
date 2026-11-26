#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sdpa231.py SECONDS [S_MIN S_MAX] -- w231: the deep pipeline's attention alone, forward + backward, at the shapes of a deep segment (1,024 queries,
24 heads, 4 KV heads, head dim 256, bf16; keys = a detached prefix of S - 1024 positions + the segment's own 1,024 (with grad), concatenated as
SegCacheBuf does; enable_gqa, lower-right causal mask: qad220 seg_attention), S drawn in [S_MIN, S_MAX] (default 120k .. 250k), with allocator churn
between iterations so the tensors land at different addresses. An out-of-bounds read at large S faults only when the page after a tensor is unmapped,
i.e. sporadically. Prints the backend's timings every 60 s; a CUDA error ends it with exit 1. Run with PYTORCH_NO_HIP_MEMORY_CACHING=1 too (every
tensor its own allocation: more unmapped neighbours)."""
import sys, time
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

secs = float(sys.argv[1]); lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (120000, 250000)
dev = torch.device("cuda:0"); BF = torch.bfloat16; L = 1024
g = torch.Generator().manual_seed(int(time.time()))
print(torch.__version__, torch.cuda.get_device_name(0), "S", lo, hi, flush=True)
t0 = time.time(); n = 0; tl = t0
while time.time() - t0 < secs:
    S = int(torch.randint(lo, hi, (1,), generator=g)); P = S - L
    junk = [torch.empty(int(torch.randint(1, 64 << 20, (1,), generator=g)), dtype=torch.uint8, device=dev) for _ in range(int(torch.randint(0, 6, (1,), generator=g)))]
    cap = P + L + int(torch.randint(0, 40000, (1,), generator=g))
    kb = torch.randn(1, 4, cap, 256, device=dev, dtype=BF); vb = torch.randn_like(kb)          # the carried KV buffer (capacity >= S)
    q = torch.randn(1, 24, L, 256, device=dev, dtype=BF, requires_grad=True)
    k = torch.randn(1, 4, L, 256, device=dev, dtype=BF, requires_grad=True); v = torch.randn_like(k, requires_grad=True)
    kk = torch.cat([kb[:, :, :P], k], -2); vv = torch.cat([vb[:, :, :P], v], -2)
    o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(L, S), scale=256 ** -0.5, enable_gqa=True)
    o.float().square().mean().backward()
    torch.cuda.synchronize(dev); n += 1
    del junk, kb, vb, q, k, v, kk, vv, o
    if time.time() - tl > 60: print(f"{time.strftime('%H:%M:%S')} {n} iterations (last S {S})", flush=True); tl = time.time()
print(f"done: {n} iterations, no error", flush=True)
