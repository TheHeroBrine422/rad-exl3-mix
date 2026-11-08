#!/usr/bin/env python3
"""mkdenoise.py CALIB MOM OUT --lam L --rho R [--sig 0.025] [--sig-in 0.005] [--ckpt MODEL] -- the denoised LDLQ
metric G and target transform F of activation denoising (arXiv 2610.07522, Theorem 1) per input group, as ourq.py dirs:

  objective  E||W^(x + lam eta) - W(x + rho eta)||^2 + alpha ||W^ - W||^2  =  tr[(W^ - W F) G' (W^ - W F)^T] + const
  G  = H + lam^2 N + lam (C + C^T)              -> OUT/hin/<group>.safetensors (challenge layout: ourq.py --hin)
  G' = G + alpha I,  alpha = sig x mean diag G  (the damping ourq.py itself adds: sig-in for the layer-input groups)
  F  = (H + alpha I + lam C + rho C^T + rho lam N) G'^-1
                                                -> OUT/wpre/<group>.safetensors "Ft" = F^T = G'^-1 A^T (ourq.py --wpre,
                                                   wpre.patch: the group's weights (in, out) become Ft @ w = (W F)^T)
H, C, N are per-token means. In G, H is the challenge-layout calib input Hessian CALIB/<group>.safetensors (all of
calib, as the base quant's) and C, N moments.py's MOM/<group>.safetensors (a subset: the control-variate form G = H_calib
+ (G_sub - H_sub), negative eigenvalues clipped if the two H disagree; at lam = 0 G is calib's H exactly). F is the
Wiener filter of ONE joint distribution, so it uses the subset's own H_sub with its C, N (and alpha from G_sub): mixing
calib's H into F breaks the regression and blows it up (|W F| / |W| 1.6-26x on 1k-token moments, vs 0.98-1.01 from the
subset alone: selftest/fsub-vs-cv.txt). lam = rho: F = I, no wpre file (then only --hin is needed). G's file stores
G x rows with rows = the calib count, so ourq's damping and proxy see the usual scale.
OUT/stats.jsonl per group: trace shares of the correction, ||H_sub / n - H / rows|| / ||H / rows|| (the moment pass's
own clean covariance against calib's: a check that the hooks saw the right tensors), the share of G's spectrum clipped to make it PSD (0 unless the subset's
H and calib's disagree enough: then reported, and G' must still factor), and with --ckpt per linear ||W F||_F / ||W||_F. fp64 on the GPU if there is one.
Lean path (groups wider than MKD_LEAN_K = 8192, e.g. the 27B's 17408-wide down inputs; MKD_LEAN=1 forces it for
every group): the same arithmetic in the same order, but in place and blockwise (no k^2/2 index pairs, at most
~5 k x k fp64 matrices at once, the clip case's eigh included (C, N, H_sub are reloaded for F): ~12 GB at 17408
instead of ~30), so its files are bit-identical to the default path's
(v3/sib27/test_lean.py); only the stats' norms are summed by blocks.
v3/yaqa/asym27 copy = proxy2b/exp/w07-denoise/mkdenoise.py + mkdenoise-w78.patch (w78, 11-07): on ROCm (R9700, torch
2.15+rocm7.14) torch.linalg.solve at k 17408 f64 falls into rocBLAS's degraded TRSM ("Device memory allocation size is
too small for TRSM") and takes > 20 min a group (> 6 min even alone on the card); the lean path's F^T = G_s'^-1 A^T is
now a Cholesky factor + blocked triangular solves (1024-row diagonal blocks by solve_triangular, the rest GEMMs):
16 s at 17408, within ~1e-13 relative of solve (dbg/solvebench*.log), so wpre is no longer bit-identical to the default
path's (hin is). If G_s' does not factor, torch.linalg.solve as before. stats: "solve_res", the relative residual
|G_s' F^T - A^T| on 256 sampled columns, and "solve" (the route)."""
import argparse, glob, json, os, re, struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("calib"); ap.add_argument("mom"); ap.add_argument("out")
ap.add_argument("--lam", type=float, required=True); ap.add_argument("--rho", type=float, required=True)
ap.add_argument("--sig", type=float, default=0.025); ap.add_argument("--sig-in", type=float, default=0.005)
ap.add_argument("--ckpt", default=None)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("DEV", "cuda") == "cuda" else "cpu")

def load_up(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
    H = torch.zeros(k, k, dtype=torch.float64, device=dev); iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up.to(dev, torch.float64); H = H + H.triu(1).T
    return H, rows, meta
def save_up(p, H, rows, meta):
    k = H.shape[0]; iu = torch.triu_indices(k, k, device=H.device)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().cpu().contiguous()}, p + ".part", metadata=meta)
    os.replace(p + ".part", p)
W = None
if a.ckpt:
    idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
    def W(key):
        with safe_open(f"{a.ckpt}/{idx[key + '.weight']}", "pt", device="cpu") as f: return f.get_tensor(key + ".weight")
def linears(group):                                   # ourq.py's layer_plan, by group name
    l, rest = re.match(r"model\.language_model\.layers\.(\d+)\.(?:(.*)\.)?input$", group).groups()
    p = f"model.language_model.layers.{l}"
    if rest is None: return ([f"{p}.self_attn.{s}_proj" for s in "qkv"] if int(l) % 4 == 3 else
                           [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"])
    return {"self_attn.o_proj": [f"{p}.self_attn.o_proj"], "linear_attn.out_proj": [f"{p}.linear_attn.out_proj"],
            "post_attention": [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"], "mlp.down_proj": [f"{p}.mlp.down_proj"]}[rest]

os.makedirs(f"{a.out}/hin", exist_ok=True)
if a.lam != a.rho: os.makedirs(f"{a.out}/wpre", exist_ok=True)
st = open(f"{a.out}/stats.jsonl", "w")
files = sorted(glob.glob(f"{a.mom}/*.input.safetensors"))
assert files, f"no moment files in {a.mom}"
LEAN_K, LEAN, BL = int(os.environ.get("MKD_LEAN_K", 8192)), os.environ.get("MKD_LEAN") == "1", 2048
def _sym_(M):                                         # M = (M + M^T) / 2 in place, by blocks (both triangles from the old M)
    k = M.shape[0]
    for i in range(0, k, BL):
        for j in range(i, k, BL):
            s = (M[i:i + BL, j:j + BL] + M[j:j + BL, i:i + BL].T) / 2
            M[i:i + BL, j:j + BL] = s; M[j:j + BL, i:i + BL] = s.T
def _bnorm(f, k):                                     # ||.||_F of a matrix given by row blocks f(i0, i1)
    return sum(float(f(i, min(i + BL, k)).pow(2).sum()) for i in range(0, k, BL)) ** 0.5
def _load_up_lean(p):
    with open(p, "rb") as fh:
        n_ = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n_))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper").to(dev)
    H = torch.zeros(k, k, dtype=torch.float64, device=dev); o = 0
    for i in range(k): H[i, i:] = up[o:o + k - i]; o += k - i
    del up
    for i in range(0, k, BL):                         # lower triangle = upper (0 + x: the default path's H + H.triu(1).T)
        d = H[i:i + BL, i:i + BL]; d += d.triu(1).T.clone()
        for j in range(i + BL, k, BL): H[j:j + BL, i:i + BL] = H[i:i + BL, j:j + BL].T
    return H, rows, meta
def _save_up_lean(p, G, scale, meta):                 # packed upper of G * scale, f32
    k = G.shape[0]; out = torch.empty(k * (k + 1) // 2, dtype=torch.float32); o = 0
    for i in range(0, k, BL):
        seg = torch.cat([G[r, r:] for r in range(i, min(i + BL, k))]); m_ = seg.numel()
        out[o:o + m_] = (seg * scale).float().cpu(); o += m_; del seg
    save_file({"hessian_upper": out}, p + ".part", metadata=meta); os.replace(p + ".part", p)
def _combine(dst, base, X, Y, lam):                   # dst = base + lam^2 Y + lam (X + X^T), rows by blocks (dst may be base)
    k = X.shape[0]
    for i in range(0, k, BL):
        dst[i:i + BL] = base[i:i + BL] + a.lam ** 2 * Y[i:i + BL] + lam * (X[i:i + BL] + X[:, i:i + BL].T)
def _chol_solve_blocked(L, X, bs=1024):              # X <- L^-T L^-1 X in place (L lower), TRSM only on diagonal blocks
    k = L.shape[0]; st = list(range(0, k, bs))
    for i in st:
        j = min(i + bs, k); X[i:j] = torch.linalg.solve_triangular(L[i:j, i:j], X[i:j], upper=False)
        if j < k: X[j:].sub_(L[j:, i:j] @ X[i:j])
    for i in reversed(st):
        j = min(i + bs, k); X[i:j] = torch.linalg.solve_triangular(L[i:j, i:j].T, X[i:j], upper=True)
        if i > 0: X[:i].sub_(L[i:j, :i].T @ X[i:j])
    return X
def lean_group(mf, g):
    H, rows, meta = _load_up_lean(f"{a.calib}/{g}.safetensors")
    def mom(s):                                       # one moment / n, fp64 on dev (N symmetrised, as the default path)
        with safe_open(mf, "pt", device="cpu") as f: t = f.get_tensor(s).to(dev, torch.float64).div_(n)
        if s == "N": _sym_(t)
        return t
    with safe_open(mf, "pt", device="cpu") as f: n = int(f.metadata()["n"])
    Hm = H.div_(rows); k = Hm.shape[0]; trH = Hm.trace()
    Hs = mom("H"); hsub = _bnorm(lambda i, j: Hs[i:j] - Hm[i:j], k) / _bnorm(lambda i, j: Hm[i:j], k); del Hs
    C, N = mom("C"), mom("N"); trN, trC = float(a.lam ** 2 * N.trace() / trH), float(2 * a.lam * C.trace() / trH)
    G = Hm; _combine(G, Hm, C, N, a.lam); del Hm, H, C, N; _sym_(G)     # C, N reloaded for F: eigh below has room
    clip = 0.0
    if a.lam and int(torch.linalg.cholesky_ex(G)[1]) != 0:
        ev, V = torch.linalg.eigh(G); del G
        clip = float(-ev.clamp(max=0).sum() / ev.clamp(min=0).sum()); G = (V * ev.clamp(min=0)) @ V.T; del V; _sym_(G)
    sg = a.sig_in if re.search(r"layers\.\d+\.input$", g) else a.sig
    al = sg * float(G.diagonal().mean())
    _save_up_lean(f"{a.out}/hin/{g}.safetensors", G, rows, dict(meta, source=f"mkdenoise lam {a.lam} rho {a.rho} from {a.mom}"))
    G.diagonal().add_(al); chol = int(torch.linalg.cholesky_ex(G)[1]) == 0; del G
    r = {"group": g, "lam": a.lam, "rho": a.rho, "sig": sg, "n": n, "rows": rows,
         "tr_N": trN, "tr_C": trC, "hsub_vs_calib": hsub, "clip": clip, "chol_Gp": chol}
    if a.lam != a.rho:
        C, N, Hs = mom("C"), mom("N"), mom("H")
        Gs = torch.empty_like(Hs); _combine(Gs, Hs, C, N, a.lam); _sym_(Gs)
        als = sg * float(Gs.diagonal().mean()); Gs.diagonal().add_(als)
        for i in range(0, k, BL):                     # A = Hs + lam C + rho C^T + rho lam N, into Hs's storage
            Hs[i:i + BL] = Hs[i:i + BL] + a.lam * C[i:i + BL] + a.rho * C[:, i:i + BL].T + a.rho * a.lam * N[i:i + BL]
        A = Hs; del C, N; A.diagonal().add_(als)
        L, info = torch.linalg.cholesky_ex(Gs)
        if int(info) == 0: Ft = _chol_solve_blocked(L, A.T.contiguous()); r["solve"] = "chol-blocked"
        else: Ft = torch.linalg.solve(Gs, A.T); r["solve"] = "lu"
        del L
        cs = torch.randperm(k, generator=torch.Generator().manual_seed(0))[:256].to(dev)
        r["solve_res"] = float((Gs @ Ft[:, cs] - A.T[:, cs]).norm() / A.T[:, cs].norm()); del Gs, A
        save_file({"Ft": Ft.float().cpu().contiguous()}, f"{a.out}/wpre/{g}.safetensors.part",
                  metadata={"lam": str(a.lam), "rho": str(a.rho), "alpha": str(als), "sig": str(sg), "mom": a.mom})
        os.replace(f"{a.out}/wpre/{g}.safetensors.part", f"{a.out}/wpre/{g}.safetensors")
        def fmi(i, j):                                # rows i..j of F^T - I
            d = Ft[i:j].clone(); d[torch.arange(j - i, device=dev), torch.arange(i, j, device=dev)] -= 1; return d
        r["F_dev"] = _bnorm(fmi, k) / k ** 0.5
        if W:
            for key in linears(g):
                w = W(key).to(dev, torch.float64).T
                r[f"wf_{key.split('.', 4)[4]}"] = float((Ft @ w).norm() / w.norm())
    return r
for mf in files:
    g = os.path.basename(mf)[:-len(".safetensors")]
    with safe_open(mf, "pt", device="cpu") as f: kk = f.get_slice("H").get_shape()[0]
    if LEAN or kk > LEAN_K:
        r = lean_group(mf, g)
        assert r["chol_Gp"], f"{g}: the damped metric is not positive definite"
        st.write(json.dumps(r) + "\n"); st.flush()
        print(f"{g:60s} trN {r['tr_N']:+.3e} trC {r['tr_C']:+.3e} hsub {r['hsub_vs_calib']:.3f} clip {r['clip']:.1e}"
              + (f" Fdev {r['F_dev']:.3e}" if "F_dev" in r else "") + " (lean)", flush=True)
        if dev.type == "cuda": torch.cuda.empty_cache()
        continue
    H, rows, meta = load_up(f"{a.calib}/{g}.safetensors")
    with safe_open(mf, "pt", device="cpu") as f:
        n = int(f.metadata()["n"]); C, N, Hs = (f.get_tensor(s).to(dev, torch.float64) / n for s in "CNH")
    Hm = H / rows; N = (N + N.T) / 2
    G = Hm + a.lam ** 2 * N + a.lam * (C + C.T); G = (G + G.T) / 2
    clip = 0.0
    if a.lam and int(torch.linalg.cholesky_ex(G)[1]) != 0:   # the control-variate G can lose definiteness where H_calib and the
        ev, V = torch.linalg.eigh(G)                   # subset's H disagree (few moment tokens): clip negative eigenvalues
        clip = float(-ev.clamp(max=0).sum() / ev.clamp(min=0).sum()); G = (V * ev.clamp(min=0)) @ V.T; G = (G + G.T) / 2
    sg = a.sig_in if re.search(r"layers\.\d+\.input$", g) else a.sig
    al = sg * float(G.diagonal().mean())
    Gp = G.clone(); Gp.diagonal().add_(al)
    r = {"group": g, "lam": a.lam, "rho": a.rho, "sig": sg, "n": n, "rows": rows,
         "tr_N": float(a.lam ** 2 * N.trace() / Hm.trace()), "tr_C": float(2 * a.lam * C.trace() / Hm.trace()),
         "hsub_vs_calib": float((Hs - Hm).norm() / Hm.norm()),
         "clip": clip, "chol_Gp": int(torch.linalg.cholesky_ex(Gp)[1]) == 0}
    save_up(f"{a.out}/hin/{g}.safetensors", G * rows, rows, dict(meta, source=f"mkdenoise lam {a.lam} rho {a.rho} from {a.mom}"))
    if a.lam != a.rho:                                 # the filter from the subset's own moments (see above)
        Gs = Hs + a.lam ** 2 * N + a.lam * (C + C.T); Gs = (Gs + Gs.T) / 2; als = sg * float(Gs.diagonal().mean())
        Gs.diagonal().add_(als); A = Hs + a.lam * C + a.rho * C.T + a.rho * a.lam * N; A.diagonal().add_(als)
        Ft = torch.linalg.solve(Gs, A.T)               # F^T = G_s'^-1 A^T (G_s' symmetric)
        save_file({"Ft": Ft.float().cpu().contiguous()}, f"{a.out}/wpre/{g}.safetensors.part",
                  metadata={"lam": str(a.lam), "rho": str(a.rho), "alpha": str(als), "sig": str(sg), "mom": a.mom})
        os.replace(f"{a.out}/wpre/{g}.safetensors.part", f"{a.out}/wpre/{g}.safetensors")
        r["F_dev"] = float((Ft - torch.eye(Ft.shape[0], device=dev, dtype=Ft.dtype)).norm() / Ft.shape[0] ** 0.5)
        if W:
            for key in linears(g):
                w = W(key).to(dev, torch.float64).T          # (in, out), as ourq.py
                r[f"wf_{key.split('.', 4)[4]}"] = float((Ft @ w).norm() / w.norm())
    assert r["chol_Gp"], f"{g}: the damped metric is not positive definite"
    st.write(json.dumps(r) + "\n"); st.flush()
    print(f"{g:60s} trN {r['tr_N']:+.3e} trC {r['tr_C']:+.3e} hsub {r['hsub_vs_calib']:.3f} clip {r['clip']:.1e}"
          + (f" Fdev {r['F_dev']:.3e}" if "F_dev" in r else ""), flush=True)
print(f"MKDENOISE DONE {len(files)} groups -> {a.out}")
