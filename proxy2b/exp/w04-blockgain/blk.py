"""blk.py -- the output-side algebra of exllamav3's two-sided LDLQ with a DIAGONAL output factor (ourq --hout with
hout_diag vectors), on the CPU.

prepare_H_out builds Ho = P diag(h sv^2) P (P: blockwise 128-point Sylvester Hadamard, sv: the regularized output
scales = per-output-channel RMS / mean, signs drop out), divides by its mean diagonal, adds sigma_out I and takes the
block-16 LDL. P is block diagonal, so Ho is too: every 128-row block of output channels is its own problem, and the
error feedback of ldlq_2hess never crosses a block boundary. With white rounding noise of variance s2 per weight
(trellis tiles: equal for one- and two-sided), the error along the output axis is e = L^-T n (L unit lower block-16,
Ho_damped = L D L^T), so the expected cost in a metric G is s2 tr(L^-1 G L^-T); one-sided (no feedback) it is s2 tr(G).

  block_costs(d, d_true)   -> per 128-block (two-sided cost, one-sided cost) in the true metric d_true (default d),
                              feedback designed on d (d = h sv^2; the identity-H_out arm: d = sv^2, d_true = h sv^2)
  gm_am(d)                 -> per block GM/AM of d (the asymptotic bound, no 16-granularity, no damping)
  perm_*                   -> output-channel permutations for B6 (MLP intermediate neurons)
"""
import math
import numpy as np, torch

HB, TB = 128, 16


def had(n=HB, dtype=torch.float64):
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def out_scales2(w):
    """regularize()'s output scales, squared: w (out, in) -> (rms over inputs / mean rms)^2, zero channels 0.1"""
    r = w.float().pow(2).mean(1).sqrt().double()
    m = r.mean()
    s = r / m if m > 1e-30 else torch.ones_like(r)
    s[s.abs() < 1e-30] = 0.1
    return (s * s).numpy()


def ldl_unit(Hd, b=TB):
    """block-16 LDL of a batch of SPD matrices [B, n, n]: unit lower L (block identity diagonal) and D blocks [B, m, b, b]"""
    C = torch.linalg.cholesky(Hd)
    n = Hd.shape[-1]; m = n // b
    Cd = torch.stack([C[:, j * b:(j + 1) * b, j * b:(j + 1) * b] for j in range(m)], 1)          # [B, m, b, b]
    Dinv = torch.linalg.inv(Cd)
    L = C.clone()
    for j in range(m): L[:, :, j * b:(j + 1) * b] = C[:, :, j * b:(j + 1) * b] @ Dinv[:, j]
    return L, Cd @ Cd.transpose(-1, -2)


def block_costs(d, d_true=None, sigma_out=0.01, dtype=torch.float64):
    """d, d_true: [n] (n a multiple of 128). Returns (two, one) per block, both in d_true's metric divided by
    mean(d_true) (cost per unit rounding-noise variance)."""
    d = np.asarray(d, np.float64); dt = d if d_true is None else np.asarray(d_true, np.float64)
    n = d.shape[0]; assert n % HB == 0, n
    P = had(dtype=dtype)
    Db = torch.from_numpy(d / d.mean()).to(dtype).view(-1, HB)
    Tb = torch.from_numpy(dt / dt.mean()).to(dtype).view(-1, HB)
    Hd = P @ torch.diag_embed(Db) @ P                  # prepare_H_out: Ho / mean diag (the mean of d), then + sigma I
    Hd.diagonal(dim1=-2, dim2=-1).add_(sigma_out)
    L, _ = ldl_unit(Hd)
    G = P @ torch.diag_embed(Tb) @ P
    Li = torch.linalg.solve_triangular(L, torch.eye(HB, dtype=dtype).expand_as(L), upper=False, unitriangular=True)
    two = (Li @ G @ Li.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1)
    one = Tb.sum(-1)
    return two.numpy(), one.numpy()


def gm_am(d):
    x = np.asarray(d, np.float64).reshape(-1, HB)
    return np.exp(np.log(np.maximum(x, 1e-300)).mean(1)) / x.mean(1)


def bits_spread(c):
    """1/2 log2(AM/GM) of block costs c: the rate (bits / weight) free per-block K could save at equal distortion
    (reverse water-filling, D ~ 2^-2R, no block driven to zero rate)"""
    c = np.maximum(np.asarray(c, np.float64), 1e-300)
    return 0.5 * math.log2(c.mean() / math.exp(np.log(c).mean()))


# ---- B6 permutations of n output channels (MLP: gate / up rows, down columns) -----------------------------------
def perm_sorted(key):
    return np.argsort(-np.asarray(key))


def perm_random(n, seed):
    return np.random.default_rng(seed).permutation(n)


def perm_balanced(key, nb=HB):
    """deal the channels sorted by key in a snake over the n / nb blocks: every block gets one channel of every
    stratum, so each block's mean log key is ~ the global one (the optimum of sum_b GM_b by convexity of exp)"""
    o = np.argsort(np.asarray(key)); B = len(o) // nb
    blocks = [[] for _ in range(B)]
    for r in range(nb):
        s = o[r * B:(r + 1) * B]
        for j, c in enumerate(s if r % 2 == 0 else s[::-1]): blocks[j].append(c)
    return np.concatenate([np.array(b) for b in blocks])
