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
H and calib's disagree enough: then reported, and G' must still factor), and with --ckpt per linear ||W F||_F / ||W||_F. fp64 on the GPU if there is one."""
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
for mf in files:
    g = os.path.basename(mf)[:-len(".safetensors")]
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
