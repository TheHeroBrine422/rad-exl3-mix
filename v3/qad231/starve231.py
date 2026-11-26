#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""starve231.py FREE_MB [FREE_MB ...] -- w231: what happens on this ROCm when PyTorch's caching allocator holds nearly all of a card and the HIP
runtime itself needs device memory (a new code object to load, scratch for a spilling kernel, inductor / Triton launches)? For each FREE_MB: fill the
card with torch tensors until hipMemGetInfo reports < FREE_MB free, then (1) compile + launch a NEW Triton kernel (module load), (2) launch a Triton
kernel with a large private array (scratch), (3) an inductor-compiled elementwise function on a new shape, (4) sdpa (gqa, long keys), (5) a matmul,
each in its own try; prints the error class / message per test and whether the context still works afterwards. One card (ROCR_VISIBLE_DEVICES)."""
import sys, time
import torch
import triton, triton.language as tl

dev = torch.device("cuda:0")
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)


def fill(free_mb):
    hold = []
    for chunk in (1024, 256, 64, 16, 4, 1):
        while torch.cuda.mem_get_info(dev)[0] / 2**20 > free_mb + chunk:
            try: hold.append(torch.empty(chunk << 20, dtype=torch.uint8, device=dev))
            except torch.OutOfMemoryError: break
    return hold


def new_kernel(tag):
    """a Triton kernel with a source unique to tag (a fresh compile + module load)"""
    src = f"""
import triton, triton.language as tl
@triton.jit
def k_{tag}(x, y, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(y + i, tl.load(x + i, mask=m) * {float(hash(tag) % 97) + 1.5}, mask=m)
"""
    import importlib.util, os, tempfile
    d = tempfile.mkdtemp(); p = os.path.join(d, f"k_{tag}.py"); open(p, "w").write(src)
    sp = importlib.util.spec_from_file_location(f"k_{tag}", p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, f"k_{tag}")


@triton.jit
def spill(x, y, n, B: tl.constexpr):
    """a large per-thread array: private (scratch) memory"""
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.zeros([B, 512], tl.float32)
    for j in range(16):
        acc += tl.load(x + (i[:, None] * 512 + tl.arange(0, 512)[None, :]) % n).to(tl.float32) * j
    tl.store(y + i, tl.sum(acc, 1), mask=i < n)


def run(name, f):
    t0 = time.time()
    try:
        f(); torch.cuda.synchronize(dev); print(f"  {name}: ok ({time.time() - t0:.1f} s)", flush=True); return True
    except Exception as e:
        print(f"  {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True); return False


cf = torch.compile(lambda a, b: torch.nn.functional.silu(a) * b + a.square(), dynamic=False)
for mb in [float(x) for x in sys.argv[1:]]:
    hold = fill(mb); free = torch.cuda.mem_get_info(dev)[0] / 2**20
    print(f"free {free:.0f} MiB (target < {mb:.0f}), torch reserved {torch.cuda.memory_reserved(dev) / 2**30:.2f} GiB", flush=True)
    x = torch.randn(1 << 20, device=dev); y = torch.empty_like(x)
    tag = f"t{int(mb)}_{int(time.time() * 1000) % 100000}"
    ok = run("new Triton kernel (module load)", lambda: new_kernel(tag)[(1 << 10,)](x, y, x.numel(), B=1024))
    ok &= run("Triton kernel with scratch", lambda: spill[(64,)](x, y, x.numel(), B=64))
    s = 3000 + int(mb)
    ok &= run("inductor-compiled function, new shape", lambda: cf(torch.randn(s, 77, device=dev), torch.randn(s, 77, device=dev)))
    q = torch.randn(1, 24, 1024, 256, device=dev, dtype=torch.bfloat16); k = torch.randn(1, 4, 8192, 256, device=dev, dtype=torch.bfloat16)
    ok &= run("sdpa gqa 1k x 8k", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, k, enable_gqa=True))
    a = torch.randn(4096, 5120, device=dev, dtype=torch.bfloat16)
    ok &= run("bf16 matmul", lambda: a @ a.T)
    del hold, x, y, q, k, a
    alive = run("context afterwards (alloc + add)", lambda: torch.ones(10, device=dev).add_(1))
    torch.cuda.empty_cache()
    if not alive: print("context broken: stop"); break
print("done")
