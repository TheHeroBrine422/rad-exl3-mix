#!/usr/bin/env python3
"""recompute.py CALIB MOM DN GROUP LAM RHO [SIG] -- w11: independent numpy recomputation of mkdenoise.py's outputs for one
group, from the objective E||Ŵ(x + lam eta) - W(x + rho eta)||^2 + alpha||Ŵ - W||^2 written out directly:
  E[x~ x~^T] = H + lam(C + C^T) + lam^2 N (x~ = x + lam eta),  E[(x + rho eta) x~^T] = H + lam C + rho C^T + rho lam N,
  F = (A + alpha I)(G_sub + alpha I)^-1 from the subset's own moments; the metric G = H_calib/rows + (G_sub - H_sub).
Compares with DN/hin/GROUP (G x rows) and DN/wpre/GROUP (Ft), and checks the objective identity numerically: for a
random Ŵ, tr[(Ŵ - WF) G' (Ŵ - WF)^T] - tr[Ŵ G' Ŵ^T] + 2 tr[W A' Ŵ^T] is constant in Ŵ. CPU, numpy, float64."""
import sys, json
import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open
calib, mom, dn, g, lam, rho = sys.argv[1:7]; lam, rho = float(lam), float(rho); sig = float(sys.argv[7]) if len(sys.argv) > 7 else 0.025
def unpack(p):
    with safe_open(p, "np") as f: md = f.metadata(); up = f.get_tensor("hessian_upper").astype(np.float64)
    k = int(md["k"]); H = np.zeros((k, k)); iu = np.triu_indices(k); H[iu] = up; H = H + np.triu(H, 1).T
    return H, int(md["rows"])
Hc, rows = unpack(f"{calib}/{g}.safetensors")
with safe_open(f"{mom}/{g}.safetensors", "np") as f:
    n = int(f.metadata()["n"]); Hs, C, N = (f.get_tensor(s).astype(np.float64) / n for s in "HCN")
N = (N + N.T) / 2
Gs = Hs + lam * (C + C.T) + lam ** 2 * N
G = Hc / rows + (Gs - Hs); G = (G + G.T) / 2
w, V = np.linalg.eigh(G); clip = float(-w.clip(max=0).sum() / w.clip(min=0).sum())
if (w <= 0).any(): G = (V * w.clip(min=0)) @ V.T
Gh, _ = unpack(f"{dn}/hin/{g}.safetensors"); Gh /= rows
out = {"group": g, "n": n, "clip": clip, "G_rel_diff": float(np.linalg.norm(Gh - G) / np.linalg.norm(G))}
if lam != rho:
    al = sig * np.trace(Gs) / Gs.shape[0]
    Gp = Gs + al * np.eye(len(Gs)); A = Hs + lam * C + rho * C.T + rho * lam * N + al * np.eye(len(Gs))
    F = A @ np.linalg.inv(Gp)
    Ft = load_file(f"{dn}/wpre/{g}.safetensors")["Ft"].astype(np.float64)
    out["Ft_rel_diff"] = float(np.linalg.norm(Ft - F.T) / np.linalg.norm(F))
    out["F_dev"] = float(np.linalg.norm(F - np.eye(len(F))) / len(F) ** 0.5)
    out["F_diag_mean"] = float(np.diag(F).mean()); out["F_offdiag_share"] = float(np.linalg.norm(F - np.diag(np.diag(F))) / np.linalg.norm(F - np.eye(len(F))))
    rng = np.random.default_rng(0); W = rng.standard_normal((64, len(F)))
    def obj_const(Wh):
        D = Wh - W @ F
        return np.trace(D @ Gp @ D.T) - np.trace(Wh @ Gp @ Wh.T) + 2 * np.trace(W @ A @ Wh.T)
    c1, c2 = obj_const(rng.standard_normal(W.shape)), obj_const(W + 0.1 * rng.standard_normal(W.shape))
    out["objective_identity_rel"] = float(abs(c1 - c2) / abs(c1))
    # brute force: E||Ŵx~ - W(x + rho eta)||^2 + al||Ŵ - W||^2 minimised by Ŵ = W F (normal equations): gradient at W F
    grad = (W @ F) @ Gp - W @ A
    out["grad_at_WF_rel"] = float(np.linalg.norm(grad) / np.linalg.norm(W @ A))
print(json.dumps(out))
