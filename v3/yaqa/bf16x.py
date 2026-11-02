"""bf16x.py -- near-fp32 matmuls on bf16 matrix cores, for the fp32 backward of the YAQA collectors (#33-13).

The model's weights are exactly bf16, so only the activation side needs more than bf16: split an fp32 tensor into
hi = bf16(x) and lo = bf16(x - hi) (|x - hi - lo| <= ~2^-16 |x|) and run bf16 GEMMs that accumulate and store in fp32
(torch.mm(..., out_dtype=torch.float32), aten::mm.dtype; CUDA / HIP only). Each bf16 x bf16 product is exact in fp32,
so the only losses are the dropped remainder and fp32 accumulation order:
  x @ Wᵀ  ~ hi @ Wᵀ + lo @ Wᵀ                         (2 GEMMs; W bf16, exact)
  g @ W   ~ ghi @ W + glo @ W                         (2 GEMMs; backward to the input)
  gᵀ x    ~ ghiᵀ xhi + ghiᵀ xlo + gloᵀ xhi            (3 GEMMs; both sides fp32: the weight gradient G_b)
Modes: "fp32" (W cast to fp32, fp32 GEMM: the reference), "x2" (the split above), "bf16" (one bf16 GEMM, fp32
accumulate and output: the activation rounded to bf16 -- what a bf16 backward costs in the GEMMs alone).
Off CUDA (CPU tests) the bf16 GEMMs are emulated in fp32 (bf16 operands upcast: same products, fp32 sums)."""
import torch

F32 = torch.float32


def split(x):
    hi = x.to(torch.bfloat16)
    return hi, (x - hi.to(F32)).to(torch.bfloat16)


_NATIVE = [None]       # does this torch have aten::mm.dtype on the GPU? (probed once)


def mmf(a, b):
    """bf16 a [M, K] @ bf16 b [K, N] -> fp32, fp32 accumulation"""
    if a.is_cuda and _NATIVE[0] is not False:
        try:
            r = torch.mm(a, b, out_dtype=F32); _NATIVE[0] = True; return r
        except (RuntimeError, TypeError) as e:
            _NATIVE[0] = False
            print(f"bf16x: no bf16->fp32 GEMM in this torch ({torch.__version__}: {str(e)[:80]}); emulating in fp32 "
                  "(correct, but no speed-up)", flush=True)
    return a.to(F32) @ b.to(F32)


def lin(x, w, mode, rows=32768):
    """x [..., K] fp32 @ w [N, K] (bf16) .T -> fp32 [..., N]; w processed in `rows`-row chunks (the 248k-row head)"""
    x2 = x.reshape(-1, x.shape[-1])
    if mode == "fp32":
        y = torch.cat([x2 @ w[i:i + rows].float().T for i in range(0, w.shape[0], rows)], -1)
    else:
        hi, lo = split(x2) if mode == "x2" else (x2.to(torch.bfloat16), None)
        parts = []
        for i in range(0, w.shape[0], rows):
            wt = w[i:i + rows].T
            p = mmf(hi, wt)
            if lo is not None: p += mmf(lo, wt)
            parts.append(p)
        y = torch.cat(parts, -1)
    return y.reshape(*x.shape[:-1], w.shape[0])


def lin_back(g, w, mode, rows=8192):
    """g [..., N] fp32 @ w [N, K] (bf16) -> fp32 [..., K] (gradient to the input)"""
    g2 = g.reshape(-1, g.shape[-1]); out = None
    for i in range(0, w.shape[0], rows):
        gc, wc = g2[:, i:i + rows], w[i:i + rows]
        if mode == "fp32": p = gc @ wc.float()
        else:
            hi, lo = split(gc) if mode == "x2" else (gc.to(torch.bfloat16), None)
            p = mmf(hi, wc)
            if lo is not None: p += mmf(lo, wc)
        out = p if out is None else out + p
    return out.reshape(*g.shape[:-1], w.shape[1])


def wgrad_rows(g, x, mode, rows=2048):
    """yield (i, G[i:i+rows]) of G = gᵀ x, g [T, M], x [T, K] fp32 (the per-sequence weight gradient, by row blocks)"""
    if mode == "fp32":
        for i in range(0, g.shape[1], rows): yield i, g[:, i:i + rows].T @ x
        return
    xh, xl = split(x) if mode == "x2" else (x.to(torch.bfloat16), None)
    for i in range(0, g.shape[1], rows):
        gc = g[:, i:i + rows]
        gh, gl = split(gc) if mode == "x2" else (gc.to(torch.bfloat16), None)
        ght = gh.T.contiguous()
        G = mmf(ght, xh)
        if xl is not None: G += mmf(ght, xl) + mmf(gl.T.contiguous(), xh)
        yield i, G


def diag_gg(g, x, mode, rows=2048):
    """rowsum(G ∘ G) for G = gᵀ x: the per-sequence diagonal of G Gᵀ, fp32 [M]"""
    d = torch.empty(g.shape[1], device=g.device, dtype=F32)
    for i, G in wgrad_rows(g, x, mode, rows): d[i:i + G.shape[0]] = G.pow(2).sum(1)
    return d


def selftest():
    """CPU: x2 vs fp32 on random data -- relative errors of the three products (expect ~1e-5 for x2, ~1e-3 for bf16)"""
    torch.manual_seed(0)
    x, w, g = torch.randn(256, 512), torch.randn(384, 512).to(torch.bfloat16), torch.randn(256, 384)
    rel = lambda a, b: float((a - b).norm() / b.norm())
    for m in ("x2", "bf16"):
        print(m, "lin", f"{rel(lin(x, w, m), lin(x, w, 'fp32')):.2e}", "back", f"{rel(lin_back(g, w, m), lin_back(g, w, 'fp32')):.2e}",
              "diag", f"{rel(diag_gg(g, x, m, 100), diag_gg(g, x, 'fp32', 100)):.2e}")


if __name__ == "__main__":
    selftest()
