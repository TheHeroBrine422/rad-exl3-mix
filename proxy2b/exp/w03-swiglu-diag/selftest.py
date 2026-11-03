#!/usr/bin/env python3
"""selftest.py -- CPU checks of A9's algebra on a small random SwiGLU block (no model, no GPU):
 1. swiglu_diag.py's SiLU' equals autograd's;
 2. the per-token gate / up output gradients from autograd equal (W_dᵀδ) SiLU'(g) u and (W_dᵀδ) SiLU(g);
 3. with δ_t i.i.d. N(0, I) over tokens, the Monte Carlo mean of Sketch B's ‖Σ_t dy_t[r] x_t‖² matches the
    analytic a9x = ‖W_d[:, r]‖² Σ_t a_t,r² ‖x_t‖² (cosine and max relative error over r);
 4. with correlated δ (E[δδᵀ] = C ≠ I) the a9g form (W_d[:, r]ᵀ diag(C) W_d[:, r]) is closer than a9x when C is diagonal."""
import math, torch
torch.manual_seed(0)
T, n, m, hd = 64, 48, 96, 40          # tokens, block input, intermediate, block output
W_g, W_u, W_d = torch.randn(m, n) / n ** .5, torch.randn(m, n) / n ** .5, torch.randn(hd, m) / m ** .5
x = torch.randn(T, n) * (1 + torch.rand(T, 1) * 3)                 # uneven token norms
def silu_d(g): s = torch.sigmoid(g); return s * (1 + g * (1 - s))
ok = True
g0 = torch.randn(1000, dtype=torch.float64, requires_grad=True)
torch.nn.functional.silu(g0).sum().backward()
e1 = float((g0.grad - silu_d(g0.detach())).abs().max()); print(f"1. SiLU' vs autograd: max abs diff {e1:.2e}"); ok &= e1 < 1e-12

g, u = x @ W_g.T, x @ W_u.T
a_g, a_u = silu_d(g) * u, torch.nn.functional.silu(g)
def grads(delta):                       # autograd dy for gate / up outputs, given dL/dy_down = delta [T, hd]
    gg, uu = g.clone().requires_grad_(True), u.clone().requires_grad_(True)
    y = (torch.nn.functional.silu(gg) * uu) @ W_d.T
    (y * delta).sum().backward(); return gg.grad, uu.grad
d = torch.randn(T, hd); dg, du = grads(d); hb = d @ W_d
e2 = max(float((dg - hb * a_g).abs().max()), float((du - hb * a_u).abs().max()))
print(f"2. dy_gate / dy_up vs closed form: max abs diff {e2:.2e}"); ok &= e2 < 1e-4

x2 = x.pow(2).sum(1, keepdim=True); wd2 = W_d.pow(2).sum(0)
an_g, an_u = wd2 * (a_g.pow(2) * x2).sum(0), wd2 * (a_u.pow(2) * x2).sum(0)
R = 4000; mg = torch.zeros(m, dtype=torch.float64); mu = torch.zeros_like(mg)
for _ in range(R):
    hb = torch.randn(T, hd) @ W_d                                   # dL/dh with isotropic, token-independent δ
    mg += ((hb * a_g).T @ x).pow(2).sum(1).double(); mu += ((hb * a_u).T @ x).pow(2).sum(1).double()
mg /= R; mu /= R
cos = lambda p, q: float((p * q).sum() / (p.norm() * q.norm()))
rel = lambda p, q: float(((p - q).abs() / q).max())
print(f"3. Sketch B Monte Carlo ({R} draws) vs a9x: gate cos {cos(mg, an_g.double()):.5f} max rel {rel(mg, an_g.double()):.3f}; "
      f"up cos {cos(mu, an_u.double()):.5f} max rel {rel(mu, an_u.double()):.3f} (MC error ~{2 * (2 / R) ** .5:.3f})")
ok &= cos(mg, an_g.double()) > 0.999 and cos(mu, an_u.double()) > 0.999

c = torch.rand(hd) ** 4 * 20 + 0.05                                 # very uneven residual-gradient variances
mg2 = torch.zeros(m, dtype=torch.float64)
for _ in range(R):
    hb = (torch.randn(T, hd) * c.sqrt()) @ W_d
    mg2 += ((hb * a_g).T @ x).pow(2).sum(1).double()
mg2 /= R
an_gg = (W_d.pow(2) * c[:, None]).sum(0) * (a_g.pow(2) * x2).sum(0)
print(f"4. diagonal C: Sketch B MC vs a9g cos {cos(mg2, an_gg.double()):.5f}, vs a9x (isotropic) cos {cos(mg2, an_g.double()):.5f}")
ok &= cos(mg2, an_gg.double()) > 0.999 and cos(mg2, an_gg.double()) > cos(mg2, an_g.double())
print("SELFTEST", "PASS" if ok else "FAIL")
