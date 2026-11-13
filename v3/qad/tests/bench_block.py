#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bench_block.py MODEL LEVEL [--layers 4] [--T 2048] -- where a training step's time goes, per block type: forward of
the QAD student block (QLinear) vs the same block in bf16, the GDN core (torch chunk rule) alone, and fwd + bwd with
checkpoint recompute; also checks QLinear in f32 against rxq.deq's W (decode identity, no bf16 rounding)."""
import argparse, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qmodel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("level"); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--dev", default="cuda:0")
a = ap.parse_args()
dev = torch.device(a.dev)
stu = qmodel.Student(a.model, a.level, [a.dev], layers=a.layers)

# f32 identity: y32 = had(had(x su) W_hat s) sv vs x W_deq, all f32
worst = 0.0
for k, q in stu.qlin.items():
    x = torch.randn(16, q.k, device=dev)
    W = stu.lev.get(k).T.to(dev)
    xs = (x * q.suh).view(-1, q.k // 128, 128) @ q.H
    y = (xs.view(-1, q.k) @ (q.wq.float() * q.s)).view(-1, q.n // 128, 128) @ q.H
    y = y.view(-1, q.n) * q.svh
    worst = max(worst, float((y - x @ W).norm() / (x @ W).norm()))
print(f"f32 identity over {len(stu.qlin)} tensors: max rel {worst:.2e}", flush=True)


def timeit(f, n=3):
    f(); torch.cuda.synchronize(dev)
    t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(dev)
    return (time.time() - t0) / n


x0 = torch.randn(1, a.T, 5120, device=dev, dtype=torch.bfloat16)
pe, masks, tpid = stu._ctx(dev, x0)
for i, ly in enumerate(stu.layers):
    kw = dict(position_embeddings=pe, attention_mask=masks[stu.cfg.layer_types[i]], position_ids=tpid)
    with torch.no_grad(): tf = timeit(lambda: ly(x0, **kw))
    x = x0.clone().requires_grad_(True)
    def fb():
        y = torch.utils.checkpoint.checkpoint(lambda h: ly(h, **kw), x, use_reentrant=False)
        y.float().pow(2).mean().backward()
    tfb = timeit(fb)
    print(f"layer {i} {stu.cfg.layer_types[i]}: fwd {tf * 1e3:.0f} ms, ckpt fwd+bwd {tfb * 1e3:.0f} ms", flush=True)
    if stu.cfg.layer_types[i] == "linear_attention":
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
        la = ly.linear_attn
        H, dk, dv = la.num_v_heads, la.head_k_dim, la.head_v_dim
        q = torch.randn(1, a.T, H, dk, device=dev, dtype=torch.bfloat16); kk = torch.randn_like(q); v = torch.randn(1, a.T, H, dv, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(1, a.T, H, device=dev); b = torch.rand(1, a.T, H, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): tg = timeit(lambda: q35.torch_chunk_gated_delta_rule(q, kk, v, g=g, beta=b, use_qk_l2norm_in_kernel=True))
        print(f"  GDN core (torch chunk rule) fwd {tg * 1e3:.0f} ms", flush=True)
        with torch.no_grad():
            tl = sum(timeit(lambda m=m: m(x0)) for m in (la.in_proj_qkv, la.in_proj_z, la.out_proj if False else la.in_proj_z))
        print(f"  in_proj_qkv + 2x in_proj_z QLinear fwd {tl * 1e3:.0f} ms", flush=True)
    with torch.no_grad():
        tm = timeit(lambda: ly.mlp(x0))
    print(f"  MLP (3 QLinear) fwd {tm * 1e3:.0f} ms", flush=True)
    W = torch.randn(5120, 17408, device=dev, dtype=torch.bfloat16)
    with torch.no_grad(): tg = timeit(lambda: x0[0] @ W)
    print(f"  plain bf16 GEMM {a.T}x5120x17408: {tg * 1e3:.1f} ms = {2 * a.T * 5120 * 17408 / tg / 1e12:.1f} TFLOPS", flush=True)
    del W
print(f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
