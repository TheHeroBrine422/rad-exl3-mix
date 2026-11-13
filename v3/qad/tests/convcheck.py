#!/usr/bin/env python3
"""convcheck.py: qmodel.shift_conv1d vs transformers' causal_conv1d_fn (F.conv1d) on GDN shapes (bf16, CPU or card):
max |diff| in bf16 ulps of the output and the gradients"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qmodel
import transformers.models.qwen3_5.modeling_qwen3_5 as q35
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)
for L in (7, 700, 2049, 8192):
    x = torch.randn(1, 10240, L, device=dev).to(torch.bfloat16).requires_grad_(True)
    w = (torch.randn(10240, 4, device=dev) * 0.3).to(torch.bfloat16).requires_grad_(True)
    ref = q35.causal_conv1d_fn(x, w, None, activation="silu"); g = torch.randn_like(ref)
    (ref.float() * g.float()).sum().backward(); gx, gw = x.grad.clone(), w.grad.clone(); x.grad = w.grad = None
    out = qmodel.shift_conv1d(x, w, None, activation="silu")
    (out.float() * g.float()).sum().backward()
    d = (out.float() - ref.float()).abs(); ulp = ref.float().abs().clamp_min(1e-3) * 2 ** -8
    print(f"L {L}: out max |d| {float(d.max()):.3e}, frac differing {float((d > 0).float().mean()):.4f}, max d/ulp {float((d / ulp).max()):.2f}; "
          f"grad x rel {float((x.grad.float() - gx.float()).norm() / gx.float().norm()):.2e}, grad w rel {float((w.grad.float() - gw.float()).norm() / gw.float().norm()):.2e}")
