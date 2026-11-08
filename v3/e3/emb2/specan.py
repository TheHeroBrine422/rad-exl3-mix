#!/usr/bin/env python3
"""specan.py SPEC.npz OUT.json -- analytic bounds from embspec.py's moments (CPU, minutes).

Embedding (rows normalized to unit RMS, as the coder sees them):
  - eigenvalues of C = Gn / V (uncentered) and of the centered C - μμᵀ; mean-row energy share ‖μ‖² / tr C
  - AM/GM of the eigenvalues = the high-rate transform-coding gain of a KLT + bit allocation over a white coder
  - Gaussian reverse water-filling D(R) / tr C at R = 0.25 .. 4 vs the white bound 2^-2R, and the bits a KLT saves at
    equal D (R_white(D) - R_wf(D)); per-128-block energy CV (what one scale per row cannot follow)
Cross-matrix prediction (raw rows): residual energy share of E given the head by least squares E ≈ Hd B
  (1 - tr(Gehᵀ... ) / tr Ge), and of Hd given E, also at rank r (best rank-r B).
Head under the calib lm_head.input Hessian H (damped 0.01 × mean diag): rows' covariance Ch = Gh / V;
  LDLQ high-rate model: rel ≈ 2^-2R · (tr Ch / n) · Σ d_j / tr(Ch H), d_j = LDL pivots of H in the signed
  128-Hadamard basis (what EXL3 codes in); vs the Gaussian water-filling bound on eig(H^½ Ch H^½).
Env HESS (lm_head.input.safetensors path), CK (checkpoint, for layer 0's readers -> M0 eigen-spectrum stats)."""
import json, math, os, sys
import numpy as np
from safetensors import safe_open

S = np.load(sys.argv[1]); OUT = sys.argv[2]; V = int(S["V"]); C = S["Gn"].shape[0]
R_GRID = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4]
res = {}


def wf(lam, R):
    """Gaussian reverse water-filling: D(R) / Σλ at mean rate R bits per component"""
    lam = np.sort(np.maximum(lam, 1e-300))[::-1]; n = len(lam)
    lo, hi = math.log(lam[-1]) - 60, math.log(lam[0])
    for _ in range(200):                                     # bisection on log θ
        mid = (lo + hi) / 2; th = math.exp(mid)
        r = np.maximum(0, 0.5 * np.log2(lam / th)).sum() / n
        lo, hi = (mid, hi) if r > R else (lo, mid)
    th = math.exp((lo + hi) / 2)
    return float(np.minimum(lam, th).sum() / lam.sum())


def rate_white(D): return -0.5 * math.log2(D)


def spec(lam, tag):
    lam = np.clip(np.sort(lam)[::-1], 1e-30, None); tot = lam.sum()
    am, gm = lam.mean(), math.exp(np.log(lam).mean())
    cum = np.cumsum(lam) / tot
    o = {"am_gm": am / gm, "am_gm_bits": 0.5 * math.log2(am / gm),
         "top_energy": {k: float(cum[k - 1]) for k in (1, 8, 64, 256, 1024, 2560)},
         "eff_rank": float(tot ** 2 / (lam ** 2).sum()), "lam_min_rel": float(lam[-1] / am)}
    curve = []
    for R in R_GRID:
        D = wf(lam, R)   # bits_saved = R_white(D_wf(R)) - R: the rate a white coder needs for the same D, minus R
        curve.append({"R": R, "D_wf": D, "D_white": 2 ** (-2 * R), "ratio": D / 2 ** (-2 * R), "bits_saved": rate_white(D) - R})
    o["wf"] = curve; res[tag] = o
    print(tag, json.dumps({k: v for k, v in o.items() if k != "wf"}))
    for c in curve: print(f"   R {c['R']:5.2f}  D_wf {c['D_wf']:.5f}  white {c['D_white']:.5f}  ratio {c['ratio']:.3f}  saves {c['bits_saved']:.3f} bit")


# --- embedding, normalized rows
Cn = S["Gn"] / V; mu = S["sn"] / V
res["mean_row_share"] = float(mu @ mu / np.trace(Cn)); print("mean-row energy share", res["mean_row_share"])
res["blk_cv"] = {"median": float(np.median(S["blk_cv"])), "p90": float(np.percentile(S["blk_cv"], 90)),
                 "iid_expect": math.sqrt(2 / 128)}
print("per-128-block energy CV", res["blk_cv"])
lam, U = np.linalg.eigh(Cn); spec(lam, "emb_uncentered")
lamc = np.linalg.eigvalsh(Cn - np.outer(mu, mu)); spec(lamc, "emb_centered")
diagC = np.diag(Cn); res["emb_diag_am_gm"] = float(diagC.mean() / math.exp(np.log(diagC).mean()))
print("per-column variance AM/GM (no rotation)", res["emb_diag_am_gm"])
np.save(os.path.splitext(OUT)[0] + "-embU.npy", U[:, ::-1].astype(np.float32))      # eigenvectors, descending

# --- cross prediction (raw rows)
Ge, Gh, Geh = S["Ge"], S["Gh"], S["Geh"]
def resid(Gyy, Gxx, Gyx, ranks=(16, 64, 256, 1024)):
    """Y ≈ X B by least squares; Gyx = Σ y xᵀ. Residual share of Y's energy, full and at rank r (best rank-r fit)."""
    L = np.linalg.cholesky(Gxx + 1e-9 * np.trace(Gxx) / len(Gxx) * np.eye(len(Gxx)))
    M = np.linalg.solve(L, Gyx.T)                            # explained energy = ‖M‖_F² ; singular values of M
    s = np.linalg.svd(M, compute_uv=False) ** 2; tot = np.trace(Gyy)
    return {"full": float(1 - s.sum() / tot), **{f"r{r}": float(1 - s[:r].sum() / tot) for r in ranks}}
res["E_from_head"] = resid(Ge, Gh, Geh); res["head_from_E"] = resid(Gh, Ge, Geh.T)
print("E from head residual share", res["E_from_head"]); print("head from E residual share", res["head_from_E"])

# --- head under H
HESS = os.environ.get("HESS", "/data/challenge/hessians/calib/lm_head.input.safetensors")
with safe_open(HESS, "np") as f: up = f.get_tensor("hessian_upper").astype(np.float64)
H = np.zeros((C, C)); H[np.triu_indices(C)] = up; H = H + H.T - np.diag(np.diag(H))
H += 0.01 * np.diag(H).mean() * np.eye(C)
Hd = np.array([[1.0]])
while len(Hd) < 128: Hd = np.block([[Hd, Hd], [Hd, -Hd]])
Hd /= math.sqrt(128); sg = np.random.default_rng(0).choice([-1.0, 1.0], C)
Rot = np.kron(np.eye(C // 128), Hd) * sg[:, None]           # x -> Rotᵀ x  (signs, then block Hadamard)
Hr = Rot.T @ H @ Rot
Lc = np.linalg.cholesky(Hr); d = np.diag(Lc) ** 2           # LDL pivots of the rotated H
Ch = Gh / V; Chr = Rot.T @ Ch @ Rot
evH = np.linalg.eigvalsh(H)
res["head_H"] = {"pivot_am_gm": float(d.mean() / math.exp(np.log(d).mean())),
                 "eigH_am_gm": float(evH.mean() / math.exp(np.log(evH).mean())),
                 "ldlq_white_const": float(np.trace(Chr) / C * d.sum() / np.trace(Ch @ H))}
w, Q = np.linalg.eigh(H); Hh = (Q * np.sqrt(w)) @ Q.T
lamh = np.linalg.eigvalsh(Hh @ Ch @ Hh)
print("head H stats", res["head_H"]); spec(lamh, "head_Hmetric")
for c in res["head_Hmetric"]["wf"]:
    c["ldlq_model"] = res["head_H"]["ldlq_white_const"] * 2 ** (-2 * c["R"]); c["wf_vs_ldlq"] = c["D_wf"] / c["ldlq_model"]
    print(f"   R {c['R']:5.2f}  LDLQ model {c['ldlq_model']:.5f}  wf/LDLQ {c['wf_vs_ldlq']:.3f}")

# --- layer-0 reader metric M0 (secondary), its eigen-spread in the embedding's KLT basis
CK = os.environ.get("CK", "/data/models/qwen38-27b-bf16")
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def T(n):
    with safe_open(f"{CK}/{wm[n]}", "pt") as f: return f.get_tensor(n).double().numpy()
p = "model.language_model.layers.0."
def gain(n):
    g = T(n); return (1 + g) if np.abs(g).mean() < 0.5 else g
g1, g2 = gain(p + "input_layernorm.weight"), gain(p + "post_attention_layernorm.weight")
M1 = sum(T(p + f"linear_attn.{t}.weight").T @ T(p + f"linear_attn.{t}.weight") for t in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"))
M2 = sum(T(p + f"mlp.{t}.weight").T @ T(p + f"mlp.{t}.weight") for t in ("gate_proj", "up_proj"))
M0 = g1[:, None] * M1 * g1[None, :] / np.trace(g1[:, None] * M1 * g1[None, :]) + g2[:, None] * M2 * g2[None, :] / np.trace(g2[:, None] * M2 * g2[None, :])
np.save(os.path.splitext(OUT)[0] + "-M0.npy", (M0 / np.trace(M0) * C).astype(np.float32))
Ud = U[:, ::-1]; m0k = np.einsum("ij,jk,ki->i", Ud.T, M0, Ud) / np.trace(M0) * C   # M0 weight of each KLT direction
res["M0_on_klt"] = {"top256_mean": float(m0k[:256].mean()), "mid_mean": float(m0k[256:2560].mean()),
                    "bottom2560_mean": float(m0k[2560:].mean()), "bottom1024_mean": float(m0k[-1024:].mean()),
                    "corr_log_lambda": float(np.corrcoef(np.log(np.clip(lam[::-1], 1e-30, None)), m0k)[0, 1])}
print("M0 weight on KLT directions (1 = average)", res["M0_on_klt"])
json.dump(res, open(OUT, "w"), indent=1)
