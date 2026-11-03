#!/usr/bin/env python3
"""selftest.py -- CPU checks of blk.py on synthetic data (output: selftest.txt)
 1. prepare_H_out's blockwise Hadamard of diag(h sv^2) (re-implemented as in quantize.py) is block diagonal and equals
    blk's per-block P D P;
 2. block_costs' closed form = a Monte-Carlo run of ldlq()'s recursion (16-row blocks from the far end, compensation
    L[later, block]^T err) with a dithered uniform quantizer (white noise), cost measured in the undamped metric;
 3. block_costs -> GM/AM as the spread grows within a block (asymptotic bound, sigma_out -> 0);
 4. perm_balanced reaches ~ the global GM/AM in every block; sorting minimises it;
 5. bits_spread of equal costs = 0, of a 2-level split = the closed form."""
import math, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blk

torch.manual_seed(0); rng = np.random.default_rng(0); ok = True
def check(name, cond, msg):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name}: {msg}")

# 1. prepare_H_out's transform
n = 512; d = np.exp(rng.normal(0, 1.5, n))
def had_r_(x):                                   # blockwise_preapply_had_r_: x[:, blk] @ had
    P = blk.had(); return torch.cat([x[:, i:i + 128] @ P for i in range(0, x.shape[1], 128)], 1)
def had_l_(x):
    P = blk.had(); return torch.cat([P @ x[i:i + 128] for i in range(0, x.shape[0], 128)], 0)
Ho = had_l_(had_r_(torch.diag(torch.from_numpy(d))))
off = Ho.clone()
for i in range(0, n, 128): off[i:i + 128, i:i + 128] = 0
P = blk.had(); ref = torch.block_diag(*[P @ torch.diag(torch.from_numpy(d[i:i + 128])) @ P for i in range(0, n, 128)])
check("blockdiag", off.abs().max() < 1e-12 and (Ho - ref).abs().max() < 1e-10,
      f"off-block max {off.abs().max():.1e}, vs per-block P D P {(Ho - ref).abs().max():.1e}")

# 2. Monte Carlo of the recursion
def mc(d, sigma=0.01, cols=20000, step=0.5):
    dn = d / d.mean(); H0 = ref_of(dn); Hd = H0 + sigma * torch.eye(len(d), dtype=torch.float64)
    L, _ = blk.ldl_unit(Hd[None]); L = L[0].clone(); L.diagonal().zero_()
    W = torch.randn(len(d), cols, dtype=torch.float64) * 3
    Q = torch.zeros_like(W); E = torch.zeros_like(W)
    for j in range(len(d), 0, -16):
        i = j - 16
        x = W[i:j] + L[j:, i:j].T @ E[j:]                  # ldlq: rows + b_L[bj:, block].T @ err
        u = torch.rand_like(x) * step
        Q[i:j] = torch.round((x + u) / step) * step - u     # dithered: noise uniform, white, independent
        E[i:j] = W[i:j] - Q[i:j]
    noise = step ** 2 / 12
    return float((E * (H0 @ E)).sum() / cols / noise)
def ref_of(dn):
    return torch.block_diag(*[P @ torch.diag(torch.from_numpy(dn[i:i + 128])) @ P for i in range(0, len(dn), 128)])
for name, dd in (("lognormal 1.5", d), ("flat", np.ones(n)), ("two-level x100", np.where(np.arange(n) % 3 == 0, 100.0, 1.0))):
    two, one = blk.block_costs(dd)
    m = mc(dd)
    check(f"monte-carlo {name}", abs(m / two.sum() - 1) < 0.02,
          f"closed form {two.sum():.3f} vs simulated {m:.3f} (one-sided {one.sum():.1f}, ratio {two.sum() / one.sum():.4f})")

# 3. asymptotics
for s in (0.5, 1.0, 2.0):
    dd = np.exp(rng.normal(0, s, 128 * 8))
    two, one = blk.block_costs(dd, sigma_out=1e-6)
    print(f"INFO lognormal sd {s}: two/one {two.sum() / one.sum():.4f}, GM/AM {np.mean(blk.gm_am(dd)):.4f} "
          f"(block-16 LDL keeps no feedback inside a 16-block: two/one >= GM/AM)")
    check(f"bound sd {s}", two.sum() / one.sum() >= np.mean(blk.gm_am(dd)) - 1e-9, "two-sided ratio above the GM/AM bound")
two, one = blk.block_costs(np.ones(1024))
check("flat no gain", abs(two.sum() / one.sum() - 1) < 1e-9, f"flat d: ratio {two.sum() / one.sum():.6f}")
# design != truth (identity H_out arm): designing on a wrong metric must not beat designing on the true one
dt = np.exp(rng.normal(0, 1, 1024)); ds = np.exp(rng.normal(0, 0.3, 1024))
a, o = blk.block_costs(ds, dt); b, _ = blk.block_costs(dt)
check("mismatch", b.sum() <= a.sum() + 1e-9, f"true-metric design {b.sum() / o.sum():.4f} <= wrong design {a.sum() / o.sum():.4f}")

# 4. permutations
key = np.exp(rng.normal(0, 1.2, 3584)); key[:128] *= 30            # a structured layout: one hot block
g = np.exp(np.log(key).mean()) / key.mean()
for nm, p in (("natural", np.arange(3584)), ("random", blk.perm_random(3584, 0)), ("balanced", blk.perm_balanced(key)),
              ("sorted", blk.perm_sorted(key))):
    assert sorted(p) == list(range(3584))
    two, one = blk.block_costs(key[p])
    print(f"INFO perm {nm:9s}: two/one {two.sum() / one.sum():.4f}  (global GM/AM {g:.4f}); across-block spread of "
          f"two-sided costs {blk.bits_spread(two):.3f} bit")
bb = blk.block_costs(key[blk.perm_balanced(key)])[0].sum(); bs = blk.block_costs(key[blk.perm_sorted(key)])[0].sum()
bn = blk.block_costs(key)[0].sum()
check("balanced best", bb <= bn and bb <= bs, f"balanced {bb:.1f} <= natural {bn:.1f}, sorted {bs:.1f}")

# 5. bits_spread
check("spread", abs(blk.bits_spread([1, 1, 1])) < 1e-12 and abs(blk.bits_spread([1, 4]) - 0.5 * math.log2(2.5 / 2)) < 1e-12,
      "equal costs 0; [1, 4] = 1/2 log2(2.5 / 2)")
print("SELFTEST", "PASS" if ok else "FAIL")
