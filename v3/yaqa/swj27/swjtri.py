"""swjtri.py -- fsutil.swiglu_acc's grid sums in one Triton kernel (swjfit27.py --fused).

swiglu_acc makes ~7 full passes over [T, J] f32 per grid point (c * ĝ, silu, * û, a * p, sum, p², sum): on the 27B
(J = 17,408, 81 points) that is ~0.15 ms per token, layer and source, most of the fit. Here each program holds BC grid
points x BJ channels of the two f32 sums and streams the document's rows once per grid group: the same quantities
(p = silu(c ĝ) û, sums over the document's positions of a p and p² in f32, then added in f64 by the caller), in another
summation order and with Triton's exp, so not bit-identical with the torch path: check() bounds the difference
(swjfit27.py --fused runs it on its first call and aborts above --fused-tol)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sums(A, G, U, C, SAP, SPP, T, J, NG, BT: tl.constexpr, BJ: tl.constexpr, BC: tl.constexpr):
    pj = tl.program_id(0); pc = tl.program_id(1)
    oj = pj * BJ + tl.arange(0, BJ); oc = pc * BC + tl.arange(0, BC)
    mj = oj < J; mc = oc < NG
    c = tl.load(C + oc, mask=mc, other=0.0)
    sap = tl.zeros((BC, BJ), dtype=tl.float32); spp = tl.zeros((BC, BJ), dtype=tl.float32)
    for t0 in range(0, T, BT):
        ot = t0 + tl.arange(0, BT)
        m = (ot[:, None] < T) & mj[None, :]
        off = ot[:, None] * J + oj[None, :]
        a = tl.load(A + off, mask=m, other=0.0); g = tl.load(G + off, mask=m, other=0.0); u = tl.load(U + off, mask=m, other=0.0)
        z = c[:, None, None] * g[None, :, :]                       # [BC, BT, BJ]
        p = z / (1.0 + tl.exp(-z)) * u[None, :, :]
        sap += tl.sum(a[None, :, :] * p, axis=1); spp += tl.sum(p * p, axis=1)
    o = oc[:, None] * J + oj[None, :]
    mo = mc[:, None] & mj[None, :]
    tl.store(SAP + o, sap, mask=mo); tl.store(SPP + o, spp, mask=mo)


def grid_sums(aa, gq, uq, grid, BT=16, BJ=64, BC=8):
    """-> (S_ap, S_pp) f32 [G, J]: sums over the rows of aa * silu(c gq) uq and (silu(c gq) uq)² per grid point c"""
    aa, gq, uq = (x.contiguous() for x in (aa, gq, uq))
    T, J = aa.shape; NG = len(grid)
    sap = torch.empty(NG, J, dtype=torch.float32, device=aa.device); spp = torch.empty_like(sap)
    _grid_sums[(triton.cdiv(J, BJ), triton.cdiv(NG, BC))](aa, gq, uq, grid.float().contiguous(), sap, spp, T, J, NG,
                                                          BT=BT, BJ=BJ, BC=BC)
    return sap, spp


def swiglu_acc(s, aa, gq, uq, grid):
    """fsutil.swiglu_acc's contract: s = [S_ap (G, J), S_pp (G, J), S_aa (J)] float64, one block of positions added"""
    s[2] += aa.pow(2).sum(0).double()
    sap, spp = grid_sums(aa, gq, uq, grid)
    s[0] += sap.double(); s[1] += spp.double()


def check(ref_acc, aa, gq, uq, grid):
    """(max |fused - torch| / max |torch|) over S_ap and S_pp for one block (fresh sums, nothing accumulated)"""
    G, J = len(grid), aa.shape[1]
    mk = lambda: [torch.zeros(G, J, dtype=torch.float64, device=aa.device), torch.zeros(G, J, dtype=torch.float64, device=aa.device),
                  torch.zeros(J, dtype=torch.float64, device=aa.device)]
    r, f = mk(), mk()
    ref_acc(r, aa, gq, uq, grid); swiglu_acc(f, aa, gq, uq, grid)
    return max(float((f[i] - r[i]).abs().max() / r[i].abs().max().clamp_min(1e-300)) for i in (0, 1))
