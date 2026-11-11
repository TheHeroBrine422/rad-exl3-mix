"""w101 (#130): the output-row permutations. gperm = composable perm2.py's gperm (stride over groups of g rows):
stored row j holds natural row gperm(R, g)[j]."""
import numpy as np
import torch


def gperm(R, g):
    """stored 128-row block b holds the g-row groups b, b + nb, b + 2 nb, ... (nb = R / 128); perm2.py verbatim"""
    assert R % 128 == 0 and 128 % g == 0
    ng = R // g; per = 128 // g; nb = R // 128
    order = np.arange(ng).reshape(nb, per).T.reshape(-1)
    return torch.from_numpy((order[:, None] * g + np.arange(g)[None, :]).reshape(-1))


def inverse(p):
    p = torch.as_tensor(p); ip = torch.empty_like(p); ip[p] = torch.arange(p.numel()); return ip


def balanced(v, g=16):
    """composable A5 (reported only): g-row tiles sorted by mean v, dealt to the nb blocks in snake order"""
    v = np.asarray(v, dtype=np.float64); R = len(v); nb = R // 128; per = 128 // g
    tiles = np.argsort(-v.reshape(R // g, g).mean(1), kind="stable")
    blocks = [[] for _ in range(nb)]
    for i, t in enumerate(tiles):
        r, c = divmod(i, nb); blocks[c if r % 2 == 0 else nb - 1 - c].append(t)
    assert all(len(b) == per for b in blocks)
    order = np.array([t for b in blocks for t in b])
    return torch.from_numpy((order[:, None] * g + np.arange(g)[None, :]).reshape(-1))
