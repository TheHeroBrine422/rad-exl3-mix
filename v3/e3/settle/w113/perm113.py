"""w113 (#148): P16-TP, the output-row tile interleave applied inside each tensor-parallel rank's rows.

Row layout (radiance qc-exl3-v3, see ../w113.md "Sharding"): a tensor's stored entry stacks parts P_0 .. P_{n-1} along
the rows (ssm_qkv: q 2048 | k 2048 | v 6144; one part otherwise); rank r of t holds rows
[O_p + r P_p / t, O_p + (r + 1) P_p / t) of every part p, concatenated in part order (core/format/share.cpp 139-153).

P16-TP(nest): inside every rank set S_r of the `nest` split (listed in increasing row order, n = N / nest rows),
stored position S_r[i] holds natural row S_r[gperm(n, 16)[i]]. nest = 4 (TP1 / TP2 / TP4) is the arm T; nest = 2 is
the report-only v_proj arm V; nest = 1 is w101's full P16 (= gperm(N, 16)).
Convention as perm101: perm[j] = the natural row stored at row j."""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w101"))
from perm101 import gperm, inverse  # noqa: E402,F401

# 27B (Qwen3.8-27B) row parts of the stored entries (config: 16 k heads x 128, 48 v heads x 128, 24 q heads x 2 x 256,
# 4 kv heads x 256)
PARTS = {"in_proj_qkv": (2048, 2048, 6144), "in_proj_z": (6144,), "q_proj": (12288,), "v_proj": (1024,),
         "k_proj": (1024,)}


def family(key):
    return key.split(".")[-1]


def rank_rows(parts, t, r):
    """the natural (= stored position) rows rank r of t holds, in its local order"""
    out, off = [], 0
    for P in parts:
        assert P % t == 0
        out.append(np.arange(off + r * P // t, off + (r + 1) * P // t)); off += P
    return np.concatenate(out)


def tpperm(parts, nest, g=16):
    N = sum(parts); p = np.empty(N, dtype=np.int64)
    for r in range(nest):
        S = rank_rows(parts, nest, r); assert len(S) % 128 == 0
        p[S] = S[gperm(len(S), g).numpy()]
    assert sorted(p.tolist()) == list(range(N))
    return torch.from_numpy(p)


def arm_perm(key, arm, R):
    """arm P = full P16, T = P16-TP (nest 4), V = nest 2 (v_proj / k_proj report arm)"""
    parts = PARTS[family(key)]; assert sum(parts) == R, (key, R)
    if arm == "P": return gperm(R, 16)
    if arm == "T": return tpperm(parts, 4)
    if arm == "V": return tpperm(parts, 2)
    raise ValueError(arm)


def local_map(parts, nest, t, k):
    """the kernel's closed form (no table): serving TP t, local stored 16-row tile k -> local natural tile.
    c_p = part p's tiles per nest rank, C_p their prefix, T = sum c_p = 8 nb, m = nest / t sub-ranks a serving rank."""
    m = nest // t; c = [P // (16 * nest) for P in parts]; C = np.concatenate([[0], np.cumsum(c)]); nb = int(C[-1]) // 8
    p = 0
    while k >= m * C[p + 1]: p += 1
    o = k - m * C[p]; s = o // c[p]; i4 = C[p] + o % c[p]
    j4 = 8 * (i4 % nb) + i4 // nb
    q = 0
    while j4 >= C[q + 1]: q += 1
    return int(m * C[q] + s * c[q] + (j4 - C[q]))


if __name__ == "__main__":      # properties (no data): nesting, block containment, closed form == table
    for fam, parts in PARTS.items():
        N = sum(parts)
        for nest in (1, 2, 4):
            p = tpperm(parts, nest).numpy()
            if nest == 1: assert np.array_equal(p, gperm(N, 16).numpy())
            # every stored 128-row block: its natural rows lie in one rank set at every t <= nest dividing nest
            widths = []
            for b in range(N // 128):
                nat = p[128 * b:128 * b + 128]; widths.append(len(set((nat // 128).tolist())))
                for t in (1, 2, 4):
                    if nest % t: continue
                    owner = lambda x: [r for r in range(t) if x in set(rank_rows(parts, t, r).tolist())][0]
                    if t > 1 and b % 7 == 0:
                        assert owner(int(b * 128)) == owner(int(nat[0])) and len({owner(int(x)) for x in nat[::16]}) == 1
            for t in (1, 2, 4):
                if nest % t: continue
                for r in range(t):
                    S = rank_rows(parts, t, r); pos = {int(x): i for i, x in enumerate(S)}
                    tab = [pos[int(p[S[16 * k]])] // 16 for k in range(len(S) // 16)]
                    cf = [local_map(parts, nest, t, k) for k in range(len(S) // 16)]
                    assert tab == cf, (fam, nest, t, r)
            print(f"{fam:12s} nest {nest}: width {min(widths)}..{max(widths)} natural blocks a stored block; "
                  f"closed form == table at TP {[t for t in (1, 2, 4) if nest % t == 0]}")
