#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test224.py (w224) -- CPU unit checks of qad220.py's --remap (the value map of the table-QAD probe): knots, identity = the plain bf16 weights
bit for bit, forward vs a float64 x @ f(m), the table's gradient vs finite differences (groups 1 and 3), the rounded (int8
table) straight-through variant, apply_remap's selection by K and its off-switch. Functions are taken from qad220.py (AST extraction).
Run: python3 test224.py   (exit 0 = all pass)"""
import ast, argparse, os, sys, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = ast.parse(open(os.path.join(HERE, "..", "qad220", "qad220.py")).read())
ns = {"np": np, "torch": torch, "BF": torch.bfloat16, "REMAP_M": 127, "os": os, "qmodel": types.SimpleNamespace(i8_to_bf16=lambda w: w.to(torch.bfloat16)), "save_atomic": lambda obj, fn: torch.save(obj, fn)}
for node in TREE.body:
    if (isinstance(node, ast.FunctionDef) and node.name in ("remap_knots", "remap_basis", "apply_remap", "write_ckpt")) or (isinstance(node, ast.ClassDef) and node.name in ("_RemapMatmul", "ValueRemap")):
        exec(compile(ast.Module([node], []), "qad220.py", "exec"), ns)
ValueRemap, remap_knots, remap_basis, apply_remap = ns["ValueRemap"], ns["remap_knots"], ns["remap_basis"], ns["apply_remap"]

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok: fails.append(name)

torch.manual_seed(0); rng = np.random.default_rng(0)
k, n = 96, 40
wq = torch.from_numpy(np.clip(np.round(rng.normal(0, 30, (k, n))), -126, 126).astype(np.int8))

# knots
hist = np.bincount(np.abs(wq.numpy().astype(np.int64)).ravel(), minlength=128)
xs = remap_knots(hist, 16)
check("knots: 0 .. max |m|, strictly increasing, <= 16", xs[0] == 0 and xs[-1] == np.abs(wq.numpy().astype(int)).max() and (np.diff(xs) > 0).all() and len(xs) <= 16, str(xs.tolist()))
check("knots: few distinct values -> few knots", len(remap_knots(np.bincount([0, 1, 1, 2], minlength=128), 64)) == 3)
A = remap_basis(xs, 16)
v = np.arange(int(xs[-1]) + 1); check("basis: identity knot values interpolate to the integers up to the last knot", np.abs(A[v][:, :len(xs)] @ xs.astype(np.float64) - v).max() < 1e-4)

M_ROWS = 6
xin = torch.from_numpy(rng.integers(-16, 17, (M_ROWS, k)).astype(np.float32) / 8).to(torch.bfloat16)
for G in (1, 3):
    m = ValueRemap(wq, G, 16)
    check(f"identity (groups {G}): delta table exactly 0", bool((m.dtab() == 0).all()) and bool((m.table() == m.iden).all()))
    y0 = m.matmul(xin, wq); yp = xin @ wq.to(torch.bfloat16)
    check(f"identity (groups {G}): y == the plain x @ m (bf16 GEMM)", y0.dtype == torch.bfloat16 and torch.equal(y0, yp), f"max |diff| {float((y0.float() - yp.float()).abs().max()):.3g}")
    with torch.no_grad(): m.d.copy_(torch.randn(G, 16) * 0.05)
    X = m.X.numpy().astype(np.float64); d = m.d.detach().numpy().astype(np.float64)
    def ref(dd):
        out = np.zeros((k, n))
        for g in range(G):
            r0, r1 = g * k // G, (g + 1) * k // G; nk = int((X[g][1:] > 0).sum()) + 1; kx = X[g][:nk]; ky = kx * (1 + dd[g][:nk])
            a_ = np.abs(wq[r0:r1].numpy().astype(np.float64)); out[r0:r1] = np.sign(wq[r0:r1].numpy()) * np.interp(a_, kx, ky)
        return out
    x64 = xin.float().numpy().astype(np.float64)
    y1 = m.matmul(xin, wq).detach().float().numpy(); yr = x64 @ ref(d)
    check(f"forward vs the float64 x @ f(m) (groups {G}): within bf16 rounding of y", np.abs(y1 - yr).max() <= 2 ** -8 * np.abs(yr).max(), f"max diff {np.abs(y1 - yr).max():.4f} of max |y| {np.abs(yr).max():.1f}")
    # a perturbation below bf16's step (the failed first probe's regime) must still move y: relative d ~ 1e-3
    with torch.no_grad(): m.d.copy_(torch.full((G, 16), 1e-3))
    ys = m.matmul(xin, wq).detach().float(); yp32 = yp.float()
    check(f"tiny map (d = 1e-3, groups {G}): y moves by ~ the exact amount (the bf16-rounded W' would not move)", abs(float((ys - yp32).abs().sum()) / float((torch.from_numpy(x64 @ (ref(np.full((G, 16), 1e-3)) - wq.numpy().astype(np.float64)))).abs().sum()) - 1) < 0.25
          and float((ys - yp32).abs().sum()) > 0)
    with torch.no_grad(): m.d.copy_(torch.from_numpy(d).float())
    # gradients of sum(Gy * y): d vs central finite differences of the float64 reference, x vs dy @ m^T
    Gy = torch.from_numpy(rng.integers(-8, 9, (M_ROWS, n)).astype(np.float32) / 8).to(torch.bfloat16)
    xg = xin.clone().requires_grad_(True); m.d.grad = None
    (m.matmul(xg, wq).float() * Gy.float()).sum().backward()
    gd = m.d.grad.detach().numpy().astype(np.float64); fd = np.zeros_like(gd); Gn = Gy.float().numpy().astype(np.float64)
    for g in range(G):
        nk = int((X[g][1:] > 0).sum()) + 1
        for j in range(nk):
            e = 1e-3; dp, dm = d.copy(), d.copy(); dp[g, j] += e; dm[g, j] -= e
            fd[g, j] = ((Gn * (x64 @ ref(dp))).sum() - (Gn * (x64 @ ref(dm))).sum()) / (2 * e)
    check(f"table gradient vs finite differences (groups {G})", np.abs(gd - fd).max() < 1e-2 * max(1.0, np.abs(fd).max()), f"max |diff| {np.abs(gd - fd).max():.2e} of max |fd| {np.abs(fd).max():.2f}; unused knots' grad {np.abs(gd[:, len(xs):]).max(initial=0.0):.1e}")
    check(f"dx = dy @ m^T (groups {G})", torch.equal(xg.grad, Gy @ wq.to(torch.bfloat16).t()))

# rounded: integer delta table, identity unchanged, gradient straight-through (nonzero)
mr = ValueRemap(wq, 1, 16, rounded=True)
check("rounded: identity y == x @ m", torch.equal(mr.matmul(xin, wq), xin @ wq.to(torch.bfloat16)))
with torch.no_grad(): mr.d.copy_(torch.randn(1, 16) * 0.05)
t = mr.dtab().detach()
check("rounded: delta entries are integers, f(v) within +-127", bool((t == t.round()).all()) and float((mr.iden + t).abs().max()) <= 127)
mr.d.grad = None; (mr.matmul(xin, wq).float()).sum().backward()
check("rounded: straight-through gradient reaches d", float(mr.d.grad.abs().sum()) > 0)

# apply_remap: selection by K, int8-only, parameters in stu.P, off switch
class FakeLin(torch.nn.Module):
    def __init__(self, K, wq):
        super().__init__(); self.K = K; self.exact = True; self.wq = wq; self.k, self.n = wq.shape
stu = types.SimpleNamespace(qlin={"model.language_model.layers.0.mlp.up_proj": FakeLin(2.0, wq), "model.language_model.layers.0.mlp.down_proj": FakeLin(3.0, wq),
                                  "model.language_model.layers.1.mlp.up_proj": FakeLin(1.0, wq)}, P={})
logs = []
args = lambda **kw: argparse.Namespace(**{**dict(remap=None, remap_lr=None, remap_round=False, a8=True, a8_int8=False, compile_scales=True, lora=0, split_layers=0, lr=3e-4), **kw})
apply_remap(args(), stu, logs.append)
check("off by default: no remap, no parameter, no log", stu.remap == {} and not stu.P and not logs and not hasattr(stu.qlin["model.language_model.layers.0.mlp.up_proj"], "remap"))
apply_remap(args(remap="2.0,16,2"), stu, logs.append)
check("--remap 2.0: K <= 2 tensors only (2.0 and 1.0), parameters named <key>:m", sorted(stu.P) == sorted(k_ + ":m" for k_ in ("model.language_model.layers.0.mlp.up_proj", "model.language_model.layers.1.mlp.up_proj"))
      and not hasattr(stu.qlin["model.language_model.layers.0.mlp.down_proj"], "remap") and stu.P["model.language_model.layers.0.mlp.up_proj:m"].shape == (2, 16), logs[-1][:160])
saved_remap, saved_P = dict(stu.remap), dict(stu.P)
try: apply_remap(args(remap="2.0", a8_int8=True), stu, logs.append); ok = False
except AssertionError: ok = True
check("--remap with --a8-int8 refused", ok)
# write_ckpt: remap.safetensors beside scales/, with the maps' tables / knots / d; no remap -> no file
import tempfile
from safetensors.torch import load_file
class FakeQ:
    def __init__(self, k_, n_): self.r = torch.nn.Parameter(torch.zeros(k_)); self.c = torch.nn.Parameter(torch.zeros(n_)); self.k, self.n = k_, n_
fq = {kk: FakeQ(wq.shape[0], wq.shape[1]) for kk in stu.qlin}
stu2 = types.SimpleNamespace(qlin=fq, remap=saved_remap, remap_args="2.0,16,2", P=saved_P, hq=None, norm=None, state=lambda: {n_: p_.detach().clone() for n_, p_ in saved_P.items()})
with tempfile.TemporaryDirectory() as td:
    ns["write_ckpt"](stu2, td + "/ck", {"budget": 1})
    z = load_file(td + "/ck/remap.safetensors"); some = next(iter(saved_remap))
    check("write_ckpt: remap.safetensors holds knots / d / table per remapped tensor", sorted(z) == sorted(f"{k_}.{s_}" for k_ in saved_remap for s_ in ("knots", "d", "table")) and tuple(z[some + ".table"].shape) == (2, 255) and os.path.exists(td + "/ck/params.pt"))
    stu2.remap = {}; ns["write_ckpt"](stu2, td + "/ck2", {"budget": 1})
    check("write_ckpt without remap: no remap.safetensors", not os.path.exists(td + "/ck2/remap.safetensors"))
print("ALL PASS" if not fails else f"FAILED: {fails}"); sys.exit(1 if fails else 0)
