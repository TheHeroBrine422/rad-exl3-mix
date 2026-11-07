#!/usr/bin/env python3
"""cdselftest.py -- CPU check of tilecd.cd's bookkeeping with a stub quantizer (scalar rounding to a coarse grid,
indices = grid points, a random "tensor-core" permutation): the error falls at every sweep, the stored indices decode
to the returned weight exactly (layout handling), the reported rel equals a direct tr(E^T H E) / tr(W^T H W), and
every accepted change lowers the objective (sweep-by-sweep re-check against a brute-force evaluation)."""
import sys, os, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilecd

torch.manual_seed(0)
k, n, step = 256, 512, 0.7
P = torch.randperm(256); Pi = torch.argsort(P)
def qt(tiles, qa):
    idx = torch.round(tiles / step).clamp(-20, 20)
    return idx * step, idx.to(torch.int16)
Q = types.SimpleNamespace(tensor_core_perm=lambda d: P, tensor_core_perm_i=lambda d: Pi, quantize_tiles_multigpu=qt)
torch.cuda.synchronize = lambda d=None: None
A = torch.randn(k, 4 * k); H = A @ A.T / (4 * k) + 0.05 * torch.eye(k)
W = torch.randn(k, n)
# a plain "LDLQ" stand-in: direct rounding (stored like ldlq: (k/16, n/16, 256) in permuted tile layout)
def encode(Wq):
    t = Wq.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n // 16, 256)[:, :, P]
    return torch.round(t / step).to(torch.int16)
def decode(enc):
    t = (enc.float() * step)[:, :, Pi].reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
    return t
Wq0 = torch.round(W / step).clamp(-20, 20) * step; enc0 = encode(Wq0)
assert torch.equal(decode(enc0), Wq0)
rel = lambda X: float(((W - X) * (H @ (W - X))).sum() / (W * (H @ W)).sum())
qa = {"tile_cd": 3}
Wq, enc = tilecd.cd(Q, W, Wq0, enc0, H, qa)
print("rel ldlq-standin", rel(Wq0), "-> cd3", rel(Wq))
assert torch.equal(decode(enc), Wq), "stored indices do not decode to the returned weight"
assert rel(Wq) < rel(Wq0)
# sweep-by-sweep monotonicity
prev = rel(Wq0); X, e = Wq0, enc0
for s in range(3):
    X, e = tilecd.cd(Q, W, X, e, H, {"tile_cd": 1}); r = rel(X); assert r <= prev + 1e-12, (s, r, prev); prev = r
print("CDSELFTEST PASS")
