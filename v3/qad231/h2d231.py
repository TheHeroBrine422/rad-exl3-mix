#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""h2d231.py -- w231: is `cpu_tensor.to(card, non_blocking=True)` from PAGEABLE host memory asynchronous on this ROCm (the host buffer read after
the call returns)? The w225 pipeline's place() copies parked GDN / conv states that way and may free the host tensor before a host sync.
For each size: queue ~T s of GPU work, issue the copy, overwrite the host tensor right away, then sync: a copy that sees the overwrite (or a
launch that returned long before the queued work finished) read host memory asynchronously. Mode `free` frees the host tensor instead (glibc
munmaps it at MALLOC_MMAP_THRESHOLD_) and reports whether the card faulted. Run on ONE card (asserts gfx12 / not an iGPU)."""
import sys, time
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "write"
dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0); assert "Graphics" not in name, name
print(torch.__version__, name, "mode", mode, flush=True)
x = torch.randn(4096, 4096, device=dev)


def busy(n):
    global x
    for _ in range(n): x = x @ x; x = x / x.norm()


busy(2); torch.cuda.synchronize()
t0 = time.time(); busy(40); torch.cuda.synchronize(); per = (time.time() - t0) / 40
for mb in (0.004, 0.06, 1, 3, 16, 64, 256):
    n = max(1, int(mb * 2**20 / 4))
    h = torch.full((n,), 1.0)
    busy(int(0.5 / per) + 1)                    # ~0.5 s of queued GPU work ahead of the copy
    t0 = time.time(); g = h.to(dev, non_blocking=True); t1 = time.time()
    if mode == "write": h.fill_(7.0)
    else: del h
    torch.cuda.synchronize(); t2 = time.time()
    s = float(g.sum()) / n
    print(f"{mb:8.3f} MB: call {1e3 * (t1 - t0):7.1f} ms, sync after {1e3 * (t2 - t1):7.1f} ms, card sees mean {s:.3f} "
          f"({'ASYNC: read after return' if s > 1.5 else 'old value'})", flush=True)
print("done")
