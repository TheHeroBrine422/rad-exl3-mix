# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Writers' asymmetric target (#87, rho 0 at 27B: #105): the denoised LDLQ metric G and the target transform F per
writer input group, from sibling moments (H = sum x x^T, C = sum x eta^T, N = sum eta eta^T through the level's own
quantized siblings; rad-exl3-mix v3/yaqa/asym27/sibmom27.py collects them: a level pass, not ported here).

Port of rad-exl3-mix proxy2b/exp/w07-denoise/mkdenoise.py (default and lean paths, the same arithmetic in the same
order; its outputs are byte-identical: tests/check_cpu.py). The writer requant is then the quant stage with
hin = OUT/hin and wpre = OUT/wpre (rxq quant --hin --wpre; tests: tools/run4070.sh C).

  G  = H + lam^2 N + lam (C + C^T)  (control-variate form: H_calib + (G_sub - H_sub), negative eigenvalues clipped)
  F  = (H_s + alpha I + lam C + rho C^T + rho lam N) G_s'^-1,  stored as Ft = F^T

python -m rxq.asym denoise CALIB MOM OUT --lam L --rho R [--sig 0.025] [--sig-in 0.005] [--ckpt MODEL]"""
import argparse, glob, json, os, re, struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BL = 2048


def _sym_(M):
    k = M.shape[0]
    for i in range(0, k, BL):
        for j in range(i, k, BL):
            s = (M[i:i + BL, j:j + BL] + M[j:j + BL, i:i + BL].T) / 2
            M[i:i + BL, j:j + BL] = s; M[j:j + BL, i:i + BL] = s.T


def _bnorm(f, k):
    return sum(float(f(i, min(i + BL, k)).pow(2).sum()) for i in range(0, k, BL)) ** 0.5


def _meta(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))["__metadata__"]


def linears(group):
    l, rest = re.match(r"model\.language_model\.layers\.(\d+)\.(?:(.*)\.)?input$", group).groups()
    p = f"model.language_model.layers.{l}"
    if rest is None: return ([f"{p}.self_attn.{s}_proj" for s in "qkv"] if int(l) % 4 == 3 else
                           [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"])
    return {"self_attn.o_proj": [f"{p}.self_attn.o_proj"], "linear_attn.out_proj": [f"{p}.linear_attn.out_proj"],
            "post_attention": [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"], "mlp.down_proj": [f"{p}.mlp.down_proj"]}[rest]


class Denoiser:
    def __init__(self, calib, mom, out, lam, rho, sig=0.025, sig_in=0.005, ckpt=None, dev=None, lean_k=8192, lean=False):
        self.calib, self.mom, self.out, self.lam, self.rho, self.sig, self.sig_in = calib, mom, out, lam, rho, sig, sig_in
        self.dev = dev or torch.device("cuda" if torch.cuda.is_available() and os.environ.get("DEV", "cuda") == "cuda" else "cpu")
        self.lean_k, self.lean = lean_k, lean
        self.W = None
        if ckpt:
            idx = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]

            def W(key):
                with safe_open(f"{ckpt}/{idx[key + '.weight']}", "pt", device="cpu") as f: return f.get_tensor(key + ".weight")
            self.W = W

    # ---- the challenge layout
    def load_up(self, p):
        dev = self.dev; meta = _meta(p); k, rows = int(meta["k"]), int(meta["rows"])
        with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
        H = torch.zeros(k, k, dtype=torch.float64, device=dev); iu = torch.triu_indices(k, k, device=dev)
        H[iu[0], iu[1]] = up.to(dev, torch.float64); H = H + H.triu(1).T
        return H, rows, meta

    def save_up(self, p, H, rows, meta):
        k = H.shape[0]; iu = torch.triu_indices(k, k, device=H.device)
        save_file({"hessian_upper": H[iu[0], iu[1]].float().cpu().contiguous()}, p + ".part", metadata=meta)
        os.replace(p + ".part", p)

    def _load_up_lean(self, p):
        dev = self.dev; meta = _meta(p); k, rows = int(meta["k"]), int(meta["rows"])
        with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper").to(dev)
        H = torch.zeros(k, k, dtype=torch.float64, device=dev); o = 0
        for i in range(k): H[i, i:] = up[o:o + k - i]; o += k - i
        del up
        for i in range(0, k, BL):
            d = H[i:i + BL, i:i + BL]; d += d.triu(1).T.clone()
            for j in range(i + BL, k, BL): H[j:j + BL, i:i + BL] = H[i:i + BL, j:j + BL].T
        return H, rows, meta

    def _save_up_lean(self, p, G, scale, meta):
        k = G.shape[0]; out = torch.empty(k * (k + 1) // 2, dtype=torch.float32); o = 0
        for i in range(0, k, BL):
            seg = torch.cat([G[r, r:] for r in range(i, min(i + BL, k))]); m_ = seg.numel()
            out[o:o + m_] = (seg * scale).float().cpu(); o += m_; del seg
        save_file({"hessian_upper": out}, p + ".part", metadata=meta); os.replace(p + ".part", p)

    def _combine(self, dst, base, X, Y, lam):
        k = X.shape[0]
        for i in range(0, k, BL):
            dst[i:i + BL] = base[i:i + BL] + self.lam ** 2 * Y[i:i + BL] + lam * (X[i:i + BL] + X[:, i:i + BL].T)

    def _sig(self, g):
        return self.sig_in if re.search(r"layers\.\d+\.input$", g) else self.sig

    def _wpre(self, Ft, als, sg):
        return ({"Ft": Ft.float().cpu().contiguous()},
                {"lam": str(self.lam), "rho": str(self.rho), "alpha": str(als), "sig": str(sg), "mom": self.mom})

    def _save_wpre(self, g, Ft, als, sg):
        t, m = self._wpre(Ft, als, sg)
        p = f"{self.out}/wpre/{g}.safetensors"
        save_file(t, p + ".part", metadata=m); os.replace(p + ".part", p)

    def lean_group(self, mf, g):
        dev, lam, rho = self.dev, self.lam, self.rho
        H, rows, meta = self._load_up_lean(f"{self.calib}/{g}.safetensors")
        with safe_open(mf, "pt", device="cpu") as f: n = int(f.metadata()["n"])

        def mom(s):
            with safe_open(mf, "pt", device="cpu") as f: t = f.get_tensor(s).to(dev, torch.float64).div_(n)
            if s == "N": _sym_(t)
            return t
        Hm = H.div_(rows); k = Hm.shape[0]; trH = Hm.trace()
        Hs = mom("H"); hsub = _bnorm(lambda i, j: Hs[i:j] - Hm[i:j], k) / _bnorm(lambda i, j: Hm[i:j], k); del Hs
        C, N = mom("C"), mom("N"); trN, trC = float(lam ** 2 * N.trace() / trH), float(2 * lam * C.trace() / trH)
        G = Hm; self._combine(G, Hm, C, N, lam); del Hm, H, C, N; _sym_(G)
        clip = 0.0
        if lam and int(torch.linalg.cholesky_ex(G)[1]) != 0:
            ev, V = torch.linalg.eigh(G); del G
            clip = float(-ev.clamp(max=0).sum() / ev.clamp(min=0).sum()); G = (V * ev.clamp(min=0)) @ V.T; del V; _sym_(G)
        sg = self._sig(g)
        al = sg * float(G.diagonal().mean())
        self._save_up_lean(f"{self.out}/hin/{g}.safetensors", G, rows, dict(meta, source=f"mkdenoise lam {lam} rho {rho} from {self.mom}"))
        G.diagonal().add_(al); chol = int(torch.linalg.cholesky_ex(G)[1]) == 0; del G
        r = {"group": g, "lam": lam, "rho": rho, "sig": sg, "n": n, "rows": rows,
             "tr_N": trN, "tr_C": trC, "hsub_vs_calib": hsub, "clip": clip, "chol_Gp": chol}
        if lam != rho:
            C, N, Hs = mom("C"), mom("N"), mom("H")
            Gs = torch.empty_like(Hs); self._combine(Gs, Hs, C, N, lam); _sym_(Gs)
            als = sg * float(Gs.diagonal().mean()); Gs.diagonal().add_(als)
            for i in range(0, k, BL):
                Hs[i:i + BL] = Hs[i:i + BL] + lam * C[i:i + BL] + rho * C[:, i:i + BL].T + rho * lam * N[i:i + BL]
            A = Hs; del C, N; A.diagonal().add_(als)
            Ft = torch.linalg.solve(Gs, A.T); del Gs, A
            self._save_wpre(g, Ft, als, sg)

            def fmi(i, j):
                d = Ft[i:j].clone(); d[torch.arange(j - i, device=dev), torch.arange(i, j, device=dev)] -= 1; return d
            r["F_dev"] = _bnorm(fmi, k) / k ** 0.5
            if self.W:
                for key in linears(g):
                    w = self.W(key).to(dev, torch.float64).T
                    r[f"wf_{key.split('.', 4)[4]}"] = float((Ft @ w).norm() / w.norm())
        return r

    def group(self, mf, g):
        dev, lam, rho = self.dev, self.lam, self.rho
        H, rows, meta = self.load_up(f"{self.calib}/{g}.safetensors")
        with safe_open(mf, "pt", device="cpu") as f:
            n = int(f.metadata()["n"]); C, N, Hs = (f.get_tensor(s).to(dev, torch.float64) / n for s in "CNH")
        Hm = H / rows; N = (N + N.T) / 2
        G = Hm + lam ** 2 * N + lam * (C + C.T); G = (G + G.T) / 2
        clip = 0.0
        if lam and int(torch.linalg.cholesky_ex(G)[1]) != 0:
            ev, V = torch.linalg.eigh(G)
            clip = float(-ev.clamp(max=0).sum() / ev.clamp(min=0).sum()); G = (V * ev.clamp(min=0)) @ V.T; G = (G + G.T) / 2
        sg = self._sig(g)
        al = sg * float(G.diagonal().mean())
        Gp = G.clone(); Gp.diagonal().add_(al)
        r = {"group": g, "lam": lam, "rho": rho, "sig": sg, "n": n, "rows": rows,
             "tr_N": float(lam ** 2 * N.trace() / Hm.trace()), "tr_C": float(2 * lam * C.trace() / Hm.trace()),
             "hsub_vs_calib": float((Hs - Hm).norm() / Hm.norm()),
             "clip": clip, "chol_Gp": int(torch.linalg.cholesky_ex(Gp)[1]) == 0}
        self.save_up(f"{self.out}/hin/{g}.safetensors", G * rows, rows, dict(meta, source=f"mkdenoise lam {lam} rho {rho} from {self.mom}"))
        if lam != rho:
            Gs = Hs + lam ** 2 * N + lam * (C + C.T); Gs = (Gs + Gs.T) / 2; als = sg * float(Gs.diagonal().mean())
            Gs.diagonal().add_(als); A = Hs + lam * C + rho * C.T + rho * lam * N; A.diagonal().add_(als)
            Ft = torch.linalg.solve(Gs, A.T)
            self._save_wpre(g, Ft, als, sg)
            r["F_dev"] = float((Ft - torch.eye(Ft.shape[0], device=dev, dtype=Ft.dtype)).norm() / Ft.shape[0] ** 0.5)
            if self.W:
                for key in linears(g):
                    w = self.W(key).to(dev, torch.float64).T
                    r[f"wf_{key.split('.', 4)[4]}"] = float((Ft @ w).norm() / w.norm())
        return r

    def run(self):
        os.makedirs(f"{self.out}/hin", exist_ok=True)
        if self.lam != self.rho: os.makedirs(f"{self.out}/wpre", exist_ok=True)
        files = sorted(glob.glob(f"{self.mom}/*.input.safetensors"))
        assert files, f"no moment files in {self.mom}"
        with open(f"{self.out}/stats.jsonl", "w") as st:
            for mf in files:
                g = os.path.basename(mf)[:-len(".safetensors")]
                with safe_open(mf, "pt", device="cpu") as f: kk = f.get_slice("H").get_shape()[0]
                lean = self.lean or kk > self.lean_k
                r = self.lean_group(mf, g) if lean else self.group(mf, g)
                assert r["chol_Gp"], f"{g}: the damped metric is not positive definite"
                st.write(json.dumps(r) + "\n"); st.flush()
                print(f"{g:60s} trN {r['tr_N']:+.3e} trC {r['tr_C']:+.3e} hsub {r['hsub_vs_calib']:.3f} clip {r['clip']:.1e}"
                      + (f" Fdev {r['F_dev']:.3e}" if "F_dev" in r else "") + (" (lean)" if lean else ""), flush=True)
                if self.dev.type == "cuda": torch.cuda.empty_cache()
        open(f"{self.out}/.done", "w").close()
        print(f"MKDENOISE DONE {len(files)} groups -> {self.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.asym")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("denoise")
    d.add_argument("calib"); d.add_argument("mom"); d.add_argument("out")
    d.add_argument("--lam", type=float, required=True); d.add_argument("--rho", type=float, required=True)
    d.add_argument("--sig", type=float, default=0.025); d.add_argument("--sig-in", type=float, default=0.005)
    d.add_argument("--ckpt", default=None)
    a = ap.parse_args(argv)
    Denoiser(a.calib, a.mom, a.out, a.lam, a.rho, a.sig, a.sig_in, a.ckpt,
             lean_k=int(os.environ.get("MKD_LEAN_K", 8192)), lean=os.environ.get("MKD_LEAN") == "1").run()


if __name__ == "__main__":
    main()
