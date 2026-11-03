#!/usr/bin/env python3
"""selftest.py QUANTIZE_ORIG.py QUANTIZE_PATCHED.py -- CPU self-test of quantize-hout-bd.patch (cpuq.py stand-in
quantizer; the module's own regularize / LDL / LDLQ / refit code).
  T1 no-op: original vs patched module, no H_out / dense 2-D H_out / scaled Hessian: outputs bit-identical.
  T2 block_ldl_batched == block_ldl per 128-block.
  T3 equivalence: H_out as a vector (new block-diagonal path) vs torch.diag of it (the old dense ldlq_2hess path):
     same dependencies, so the same quantized weight up to float summation order.
  T4 mechanism: two-sided (vector) vs one-sided on a diag output factor with a log-normal spread, weighted error
     tr(E^T H E diag(h)); and the predictor sum_b GM(d_b) / sum_b AM(d_b), d = h sv^2 per 128-row block."""
import sys, os, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpuq
torch.set_num_threads(4)
orig, pat = cpuq.load(sys.argv[1]), cpuq.load(sys.argv[2])
g = torch.Generator().manual_seed(0)
k, n = 256, 512
X = torch.randn(4000, k, generator=g) * torch.linspace(0.3, 3, k)            # anisotropic inputs
H = X.T @ X                                                                    # summed over 4000 rows (count 1000: scale only)
W = torch.randn(k, n, generator=g) * 0.02 * torch.linspace(0.5, 2, n)
A = torch.randn(n, n, generator=g); Ho = A @ A.T / n + torch.eye(n)
ok = True
def same(a, b): return torch.equal(a[0], b[0]) and all(torch.equal(a[1][x], b[1][x]) for x in a[1])
# ---- T1
for name, kw in (("one-sided", {}), ("dense H_out", {"H_out": Ho}), ("scaled Hessian", {"scaled_hessian": True}),
                 ("dense H_out + scaled", {"H_out": Ho, "scaled_hessian": True})):
    r = same(cpuq.quant(orig, W, H, **kw), cpuq.quant(pat, W, H, **kw)); ok &= r
    print(f"T1 no-op {name:22s}: {'identical' if r else 'DIFFERS'}")
# ---- T2
Hb = torch.stack([(lambda B: B @ B.T / 128 + 0.05 * torch.eye(128))(torch.randn(128, 128, generator=g)) for _ in range(5)])
Lb = pat.block_ldl_batched(Hb.clone(), 16, {})
Ls = torch.stack([pat.block_ldl(h.clone(), 16, {}, False)[0] for h in Hb])
for L_ in Ls:                                    # as prepare_H_out: identity blocks, then the scalar diagonal zeroed
    L_.diagonal().fill_(0.0)
err = float((Lb - Ls).abs().max()); r = err < 1e-4; ok &= r
print(f"T2 block_ldl_batched vs block_ldl: max |diff| {err:.2e} {'OK' if r else 'FAIL'}")
# ---- T3 / T4
h = torch.exp(2.0 * torch.randn(n, generator=g))
def werr(wq, hv):
    E = W - wq; return float(((H @ E) * (E * hv)).sum() / ((H @ W) * (W * hv)).sum())
for sh in (False, True):
    a = cpuq.quant(pat, W, H, H_out=h, sigma_reg_out=0.01, scaled_hessian=sh)
    b = cpuq.quant(pat, W, H, H_out=torch.diag(h), sigma_reg_out=0.01, scaled_hessian=sh)
    ia, ib = a[2], b[2]
    enc_a, enc_b = a[1]["trellis"], b[1]["trellis"]
    rel = float((a[0] - b[0]).norm() / b[0].norm())
    ea, eb = werr(a[0], h), werr(b[0], h)
    r = rel < 1e-3 and abs(ea / eb - 1) < 1e-3; ok &= r
    print(f"T3 vector vs dense diag (scaled={sh}): rel |dWq| {rel:.2e}, two-sided error {ea:.5f} vs {eb:.5f} {'OK' if r else 'FAIL'}")
for K in (2, 3, 4):
    one = cpuq.quant(pat, W, H, K=K)
    two = cpuq.quant(pat, W, H, K=K, H_out=h, sigma_reg_out=0.01)
    print(f"T4 K{K}: two-sided / one-sided weighted error {werr(two[0], h) / werr(one[0], h):.3f}; "
          f"unweighted {werr(two[0], torch.ones(n)) / werr(one[0], torch.ones(n)):.3f}")
# predictor: d = h sv^2 (sv = output scales of the regularized weight: column RMS of W, as regularize)
sv = W.pow(2).mean(0).sqrt(); sv = sv / sv.mean(); d = (h * sv ** 2).view(-1, 128)
d = d / d.mean() + 0.01
print(f"T4 predictor sum GM / sum AM over 128-blocks: {float(torch.exp(torch.log(d).mean(1)).sum() / d.mean(1).sum()):.3f}")
print("SELFTEST", "PASS" if ok else "FAIL")
