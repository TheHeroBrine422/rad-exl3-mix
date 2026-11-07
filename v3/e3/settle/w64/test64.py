#!/usr/bin/env python3
"""test64.py -- CPU self-checks of settle64.py on synthetic data (seconds; no 27B data):
 1. blockrev is an involution and keeps each 16-column block's own column order;
 2. ldlq_rev == an independent transcription of exllamav3's ldlq() (block LDL of H, last block first, feedback
    through L) with the same per-tile quantizer (trellis.vit swapped for rounding to a grid, so the check is exact);
 3. SETTLE_ORDER=fwd path == trellis.ldlq_trellis (identity), ldlq_scaled with S = 1 == ldlq_trellis;
 4. post_fit == a per-row least-squares solve done by brute force;
 5. codebooks: mul1x_i8c1.0 == mul1x_i8 == cbx.py's values; clipped variants are monotone and within +-127 steps."""
import os, sys
import numpy as np, torch
os.environ.setdefault("SETTLE_ORDER", "rev")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settle64 as S
import trellis
torch.set_grad_enabled(False)
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {info}")

# 1
p = S.blockrev(64)
check("blockrev involution", torch.equal(p[p], torch.arange(64)))
check("blockrev keeps inner order", p[:16].tolist() == list(range(48, 64)))

# 2 rounding quantizer in place of the trellis (tile-wise, deterministic)
GRID = 0.37
real_vit = trellis.vit
trellis.vit = lambda x, cb, K: (np.round(x / GRID) * GRID).astype(np.float32)
g = torch.Generator().manual_seed(0)
R, Kd = 32, 96
A = torch.randn(Kd, 3 * Kd, generator=g, dtype=torch.float64); H = A @ A.T / (3 * Kd)
H[5, :] *= 1; H[:, 7] *= 1
d = torch.rand(Kd, generator=g, dtype=torch.float64) * 30 + 0.1; H = H * d.sqrt()[:, None] * d.sqrt()[None, :]
W = torch.randn(R, Kd, generator=g, dtype=torch.float64)
damp = 0.01


def exl_ldlq(W, H, damp):
    """exllamav3 ldlq(): weight (k, n) = W^T; H = L L^T (lower Cholesky); block LDL: L <- L blockdiag(L_bb)^-1, zero
    diagonal blocks...; here the equivalent dense form: quantize block j (last first) with target
    w_j + sum_{i > j} L_ij^T e_i, where L is the unit block-lower factor of H (H = L D L^T)"""
    H = H.clone(); H.diagonal().add_(damp * H.diagonal().mean())
    n = H.shape[0]; m = n // 16
    Lc = torch.linalg.cholesky(H)
    # unit block-lower: L = Lc blockdiag(Lc_bb)^-1
    Lu = Lc.clone()
    for b in range(m):
        s = slice(16 * b, 16 * b + 16)
        Lu[:, s] = Lc[:, s] @ torch.linalg.inv(Lc[s, s])
    Wk = W.T.clone(); Q = torch.zeros_like(Wk); E = torch.zeros_like(Wk)
    for b in range(m - 1, -1, -1):
        s = slice(16 * b, 16 * b + 16)
        comp = Lu[16 * (b + 1):, s].T @ E[16 * (b + 1):] if b < m - 1 else 0
        t = Wk[s] + comp
        tiles = t.T.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()       # same tile as ldlq_trellis
        q = torch.from_numpy(trellis.vit(tiles, None, 2).astype(np.float64)).reshape(R, 16).T
        Q[s] = q; E[s] = Wk[s] - q                                                # exllamav3: err = weight - weight_q
    return Q.T


Qe = exl_ldlq(W, H, damp)
Qr = S.ldlq_rev(W, H, None, 2, damp)
check("ldlq_rev == exllamav3 ldlq() transcription", torch.allclose(Qe, Qr, atol=1e-9), f"max {float((Qe - Qr).abs().max()):.2e}, codes differ {int((Qe != Qr).sum())}")
Qf = S._ldlq_fwd(W, H, None, 2, damp)
err = lambda Q: float(torch.trace((W - Q) @ H @ (W - Q).T))
print(f"     rel err fwd {err(Qf):.4f} rev {err(Qr):.4f} (orders differ: {int((Qf != Qr).sum())} codes)")
check("ldlq_scaled(S=1) == ldlq_trellis (rev)", torch.equal(S.ldlq_scaled(W, H, None, 2, damp, torch.ones_like(W)), Qr))

# 4 post_fit vs brute force on one row
Kd2 = 384; R2 = 16
A = torch.randn(Kd2, 2 * Kd2, generator=g, dtype=torch.float64); M = A @ A.T / Kd2
T = torch.randn(R2, Kd2, generator=g, dtype=torch.float64); Q = T + 0.3 * torch.randn(R2, Kd2, generator=g, dtype=torch.float64)
Qp, s = S.post_fit(T, Q, M)
r = 3; nb = Kd2 // 128
X = torch.zeros(nb, Kd2, dtype=torch.float64)
for b in range(nb): X[b, 128 * b:128 * b + 128] = Q[r, 128 * b:128 * b + 128]
sb = torch.linalg.solve(X @ M @ X.T, X @ M @ T[r])
check("post_fit == brute-force LS", torch.allclose(s[r], sb, rtol=1e-8), f"{s[r].tolist()} vs {sb.tolist()}")
e0 = float((T[r] - Q[r]) @ M @ (T[r] - Q[r])); e1 = float((T[r] - Qp[r]) @ M @ (T[r] - Qp[r]))
check("post_fit lowers the row error", e1 <= e0, f"{e0:.3f} -> {e1:.3f}")

# 5 codebooks
trellis.vit = real_vit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../sub1/r94b"))
import cbx
c1 = S.codebook("mul1x_i8"); c1b = S.codebook("mul1x_i8c1.0"); cx = cbx.codebook("mul1x_i8")
check("mul1x_i8 == cbx", np.array_equal(c1, cx)); check("i8c1.0 == i8", np.array_equal(c1, c1b))
check("mul1x == cbx", np.array_equal(S.codebook("mul1x"), cbx.codebook("mul1x")))
v = S.codebook("mul1x")
for a in (1.15, 1.33, 1.5, 2.0):
    c = S.codebook(f"mul1x_i8c{a}"); st = 4 * float(np.float32(4.0) * np.float32(S._f16(0x1eee))) / 4 / a
    o = np.argsort(v, kind="stable")
    mono = bool(np.all(np.diff(c[o]) >= -1e-12)); lv = len(np.unique(c)); clip = float(np.mean(np.abs(c) >= np.abs(c).max() * (1 - 1e-6)))
    check(f"i8c{a} monotone in v", mono, f"levels {lv}, clipped share {clip:.4f}, range {c.min():+.3f}..{c.max():+.3f}, step {st:.5f}")
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
