# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""swjtri109.py -- #144: rxq/swjtri.py's grid sums with the launch grid reordered: program axis 0 = the grid group, axis
1 = the channel block, so the NG / BC programs that read the same a / g / u rows run side by side and share them through
L2 (today axis 0 = the channel block: the programs of one block are J / BJ launches apart and each re-reads the rows from
DRAM). The program body is rxq.swjtri._grid_sums' (same ops, shapes, BT / BJ / BC, num_warps): only which program gets
which (block, group) pair changes, so every sum is computed by the same code in the same order (#144 checks torch.equal).
python swjtri109.py: bench + identity vs rxq.swjtri at the 27B shape."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sums_cg(A, G, U, C, SAP, SPP, T, J, NG, BT: tl.constexpr, BJ: tl.constexpr, BC: tl.constexpr):
    pc = tl.program_id(0); pj = tl.program_id(1)                   # the only change vs rxq.swjtri._grid_sums
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


@triton.jit
def _grid_sums_fast(A, G, U, C, SAP, SPP, T, J, NG, BT: tl.constexpr, BJ: tl.constexpr, BC: tl.constexpr):
    """(b) proposal: _grid_sums_cg with a non-IEEE divide in SiLU (results move at the ulp level)"""
    pc = tl.program_id(0); pj = tl.program_id(1)
    oj = pj * BJ + tl.arange(0, BJ); oc = pc * BC + tl.arange(0, BC)
    mj = oj < J; mc = oc < NG
    c = tl.load(C + oc, mask=mc, other=0.0)
    sap = tl.zeros((BC, BJ), dtype=tl.float32); spp = tl.zeros((BC, BJ), dtype=tl.float32)
    for t0 in range(0, T, BT):
        ot = t0 + tl.arange(0, BT)
        m = (ot[:, None] < T) & mj[None, :]
        off = ot[:, None] * J + oj[None, :]
        a = tl.load(A + off, mask=m, other=0.0); g = tl.load(G + off, mask=m, other=0.0); u = tl.load(U + off, mask=m, other=0.0)
        z = c[:, None, None] * g[None, :, :]
        p = tl.fdiv(z, 1.0 + tl.exp(-z), ieee_rounding=False) * u[None, :, :]
        sap += tl.sum(a[None, :, :] * p, axis=1); spp += tl.sum(p * p, axis=1)
    o = oc[:, None] * J + oj[None, :]
    mo = mc[:, None] & mj[None, :]
    tl.store(SAP + o, sap, mask=mo); tl.store(SPP + o, spp, mask=mo)


FAST = {"on": False}


def grid_sums(aa, gq, uq, grid, BT=16, BJ=64, BC=8):
    aa, gq, uq = (x.contiguous() for x in (aa, gq, uq))
    T, J = aa.shape; NG = len(grid)
    sap = torch.empty(NG, J, dtype=torch.float32, device=aa.device); spp = torch.empty_like(sap)
    (_grid_sums_fast if FAST["on"] else _grid_sums_cg)[(triton.cdiv(NG, BC), triton.cdiv(J, BJ))](aa, gq, uq, grid.float().contiguous(), sap, spp, T, J, NG,
                                                             BT=BT, BJ=BJ, BC=BC)
    return sap, spp


def swiglu_acc(s, aa, gq, uq, grid, aa2=None):
    """rxq.swjtri.swiglu_acc; aa2 = aa.pow(2).sum(0).double() computed once for every source of the block (same value)"""
    s[2] += aa.pow(2).sum(0).double() if aa2 is None else aa2
    sap, spp = grid_sums(aa, gq, uq, grid)
    s[0] += sap.double(); s[1] += spp.double()


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
    from rxq import swjtri
    torch.manual_seed(0); J = 17408; grid = torch.arange(0.6, 1.4 + 0.005, 0.01, device="cuda")
    for T in (300, 1280, 4096):
        g = torch.randn(T, J, device="cuda"); u = torch.randn(T, J, device="cuda"); a = torch.nn.functional.silu(g * 1.03) * u * 0.98
        r = {}
        def fast(*x):
            FAST["on"] = True
            try: return grid_sums(*x)
            finally: FAST["on"] = False
        for name, f in (("swjtri", swjtri.grid_sums), ("cg", grid_sums), ("fast", fast)):
            y = f(a, g, u, grid); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(5): y = f(a, g, u, grid)
            torch.cuda.synchronize(); r[name] = ((time.perf_counter() - t) / 5, y)
        same = all(torch.equal(x, y) for x, y in zip(r["swjtri"][1], r["cg"][1]))
        print(f"T {T}: swjtri {r['swjtri'][0] * 1e3:.2f} ms ({r['swjtri'][0] / T * 1e6:.2f} us/pos), cg {r['cg'][0] * 1e3:.2f} ms "
              f"({r['cg'][0] / T * 1e6:.2f} us/pos), x{r['swjtri'][0] / r['cg'][0]:.2f}, identical {same}", flush=True)
        d = max(float(((x - y).abs().max() / y.abs().max())) for x, y in zip(r["fast"][1], r["swjtri"][1]))
        print(f"   fast (b): {r['fast'][0] * 1e3:.2f} ms x{r['swjtri'][0] / r['fast'][0]:.2f}, max rel diff {d:.2e}", flush=True)
