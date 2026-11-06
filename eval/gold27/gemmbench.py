#!/usr/bin/env python3
"""gemmbench.py -- torch bf16 GEMM TFLOPS on the 27B's layer shapes (M = 8000 tokens; the head at M = 256), for scaling
mkgold27.py's timings between cards (README "Disk and time")."""
import torch, time
torch.set_grad_enabled(False)
M = 8000
for N, K in ((10240, 5120), (34816, 5120), (5120, 17408), (248320, 5120)):
    if N == 248320: M = 256
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda"); w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    for _ in range(3): torch.nn.functional.linear(a, w)
    torch.cuda.synchronize(); t = time.time(); r = 10
    for _ in range(r): torch.nn.functional.linear(a, w)
    torch.cuda.synchronize(); dt = (time.time() - t) / r
    print(f"M={M} N={N} K={K}: {dt*1e3:.2f} ms, {2*M*N*K/dt/1e12:.1f} TFLOPS")
    del a, w; torch.cuda.empty_cache()
