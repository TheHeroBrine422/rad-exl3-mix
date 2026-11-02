#!/usr/bin/env python3
"""ldlq2test.py -- exllamav3's two-sided LDLQ (ldlq_2hess) against the one-sided ldlq on the same rotated weight and
input factor: with L_out = 0 (H_out = I in the rotated basis) the two must give the same quantized weight (the
objective is identical; only the tile order differs, and with no output-side feedback every tile's input is the same).
Then a nonzero L_out: two-sided proxy tr(E^T H_in E H_out) must be lower than one-sided's on that objective."""
import torch
import exllamav3.modules.quant.exl3_lib.quantize as Q
torch.manual_seed(0); dev = torch.device("cuda")
k, n = 1024, 512
W = torch.randn(k, n, device=dev) * 0.5
def spd(d, r):
    A = torch.randn(d, r, device=dev); H = A @ A.T / r; H.diagonal().add_(0.01 * H.diagonal().mean()); return H
Hin, Hout = spd(k, 256), spd(n, 128)
qa = {"K": 3, "devices": [0], "device_ratios": None, "mul1": True}
L_in, _ = Q.block_ldl(Hin.clone(), 16, qa, False); L_in.diagonal().fill_(0.0)
L_out, _ = Q.block_ldl(Hout.clone(), 16, qa, False); L_out.diagonal().fill_(0.0)
q1, _ = Q.ldlq(W.clone(), L_in.clone(), qa)
q2, _ = Q.ldlq_2hess(W.clone(), L_in.clone(), torch.zeros(n, n, device=dev), qa)
print(f"L_out = 0: one-sided vs two-sided identical: {torch.equal(q1, q2)}, rel diff {((q1 - q2).norm() / q1.norm()).item():.2e}")
q3, _ = Q.ldlq_2hess(W.clone(), L_in.clone(), L_out.clone(), qa)
obj = lambda q: (((W - q).T @ Hin @ (W - q)) * Hout).sum().item() / ((W.T @ Hin @ W) * Hout).sum().item()
print(f"two-sided objective tr(E^T Hin E Hout) / ref: one-sided {obj(q1):.4e}, two-sided {obj(q3):.4e} ({obj(q3) / obj(q1):.3f}x)")
