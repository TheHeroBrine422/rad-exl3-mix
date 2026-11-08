#!/usr/bin/env python3
"""solvebench.py N [N ...] -- f64 SPD solve X = G^-1 B on the GPU, B n x n, by several routes (w78: mkdenoise's
torch.linalg.solve took > 20 min per 17408 down group on the R9700, rocBLAS "TRSM performance is degraded").
Prints seconds per route and the max rel difference to the first route. Env ROUTES=a,b (default all; solve skipped
above n 8192 unless --with-solve)."""
import sys, time, torch

dev = torch.device("cuda")
def spd(n, g):
    X = torch.randn(n, n // 4, dtype=torch.float64, device=dev, generator=g)
    G = X @ X.T / n; G.diagonal().add_(0.05 * float(G.diagonal().mean())); return G
def blocked_trsm_lower(L, B, bs=1024):
    """X = L^-1 B, L lower triangular: diagonal blocks by solve_triangular, the rest by GEMMs (in place in X)"""
    X = B.clone(); n = L.shape[0]
    for i in range(0, n, bs):
        j = min(i + bs, n)
        X[i:j] = torch.linalg.solve_triangular(L[i:j, i:j], X[i:j], upper=False)
        if j < n: X[j:].sub_(L[j:, i:j] @ X[i:j])
    return X
def blocked_trsm_upper_T(L, B, bs=1024):
    """X = L^-T B (upper = L^T), backwards"""
    X = B.clone(); n = L.shape[0]; starts = list(range(0, n, bs))
    for i in reversed(starts):
        j = min(i + bs, n)
        X[i:j] = torch.linalg.solve_triangular(L[i:j, i:j].T, X[i:j], upper=True)
        if i > 0: X[:i].sub_(L[i:j, :i].T @ X[i:j])
    return X
def chol_blocked(G, B):
    L = torch.linalg.cholesky(G); return blocked_trsm_upper_T(L, blocked_trsm_lower(L, B))
def inv_blocked(G, B):                       # L^-1 by blocked TRSM on I, then X = L^-T (L^-1 B) by plain GEMMs
    L = torch.linalg.cholesky(G); Li = blocked_trsm_lower(L, torch.eye(G.shape[0], dtype=G.dtype, device=dev))
    return Li.T @ (Li @ B)
ROUTES = {"solve": lambda G, B: torch.linalg.solve(G, B),
          "cholesky_solve": lambda G, B: torch.cholesky_solve(B, torch.linalg.cholesky(G)),
          "chol+blocked": chol_blocked, "chol+inv": inv_blocked}
import os
if os.environ.get("ROUTES"): ROUTES = {k: ROUTES[k] for k in os.environ["ROUTES"].split(",")}
for n in (int(x) for x in sys.argv[1:] if x.isdigit()):
    g = torch.Generator(device=dev).manual_seed(n)
    G = spd(n, g); B = torch.randn(n, n, dtype=torch.float64, device=dev, generator=g)
    t = time.time(); C = G @ B; torch.cuda.synchronize(); tg = time.time() - t; del C
    print(f"n {n}: one f64 GEMM {tg:.2f} s ({2 * n ** 3 / tg / 1e12:.2f} TFLOP/s)", flush=True)
    ref = None
    for name, f in ROUTES.items():
        if name == "solve" and n > 8192 and "--with-solve" not in sys.argv: continue
        torch.cuda.synchronize(); t = time.time(); X = f(G, B); torch.cuda.synchronize(); dt = time.time() - t
        res = float((G @ X - B).norm() / B.norm())
        d = "" if ref is None else f", max rel diff to {list(ROUTES)[0]} {float((X - ref).abs().max() / ref.abs().max()):.1e}"
        if ref is None: ref = X
        print(f"  {name:15s} {dt:7.2f} s, residual {res:.1e}{d}", flush=True)
        del X
    del ref, G, B; torch.cuda.empty_cache()
