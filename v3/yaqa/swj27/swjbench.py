#!/usr/bin/env python3
"""swjbench.py [J] [T] -- one block of swj grid sums at the 27B's shape (J = 17,408 channels, T positions, 81 grid
points): fsutil.swiglu_acc (torch) vs swjtri.swiglu_acc (Triton), time per position and the fused check's difference.
Synthetic heavy-tailed inputs (a Student-t like MLP activation); prints one line per path and BENCH ... last."""
import os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "proxy2b", "exp", "w02-fixedscale")); sys.path.insert(0, HERE)
import fsutil as U, swjtri  # noqa: E402

J = int(sys.argv[1]) if len(sys.argv) > 1 else 17408; T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
dev = torch.device("cuda"); U.cap_vram(); torch.manual_seed(0)
GRID = torch.arange(0.6, 1.4 + 0.005, 0.01, device=dev)
g = torch.distributions.StudentT(4.0).sample((T, J)).to(dev) * 0.5; u = torch.randn(T, J, device=dev)
aa = torch.nn.functional.silu(g) * u
gq = g + 0.05 * torch.randn_like(g); uq = u + 0.05 * torch.randn_like(u)
err = swjtri.check(U.swiglu_acc, aa, gq, uq, GRID)
res = {}
for name, f in (("torch", U.swiglu_acc), ("fused", swjtri.swiglu_acc)):
    s = [torch.zeros(len(GRID), J, dtype=torch.float64, device=dev) for _ in range(2)] + [torch.zeros(J, dtype=torch.float64, device=dev)]
    f(s, aa, gq, uq, GRID); torch.cuda.synchronize()
    n = 3 if name == "torch" else 10; t0 = time.time()
    for _ in range(n): f(s, aa, gq, uq, GRID)
    torch.cuda.synchronize(); res[name] = (time.time() - t0) / n / T * 1e6
    print(f"{name}: {res[name]:.2f} us per position (J {J}, {len(GRID)} grid points, T {T})", flush=True)
print(f"BENCH {torch.cuda.get_device_name(0)}: torch {res['torch']:.2f} us, fused {res['fused']:.2f} us per position and "
      f"layer and source; speedup {res['torch'] / res['fused']:.1f}x; fused max rel difference {err:.2e}")
