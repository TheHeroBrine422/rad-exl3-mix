#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fd218.py OUT [--dirs lr4,lr4p] [--rel-eps 0.005] -- #218 sign / curvature check of the low-rank directions (11-22): CONTINUOUS weights,
no encode. For a direction dir (runs/r0/<dir>/<key>.g, rank 4: D_4 or M_d^-1 D_4), the probe tensors' W_hat(eta) = W_hat_rel - eta delta
D / rms(D) (encode218.py --step quant's units: eta = the step's rms as a fraction of the tensor's quantization step delta), applied in the
student's GEMM as an exact f32 low-rank add (x A) B on top of the released integer codes (eta 0 = the released model, bitwise).
Tune-side batch: the first --docs documents of qad218-tune (first --cap tokens). Per direction: the predicted slope from the stored
round-0 gradient (dKL/deta = -sum G . delta D / rms(D), G = runs/r0/enc/grad = the 300k-row mean), then KL at eta = 0, +-eps, +-2 eps
(all 26 tensors; eps = rel_eps KL / |slope|), and +-eps per block. A parabola through the five all-tensor points gives eta* = -b / 2a.
Writes OUT/fd.json; one KL per line in OUT/fd.log."""
import argparse, json, os, sys, time, types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import probe218 as P  # noqa: E402

BF = torch.bfloat16


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--dirs", default="lr4,lr4p")
    ap.add_argument("--rel-eps", type=float, default=0.005); ap.add_argument("--docs", type=int, default=34); ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--model", default="/data/workspace/models/qwen38-27b-bf16"); ap.add_argument("--lv", default="/data/w218/lv/2.50")
    ap.add_argument("--calib", default="/data/challenge/golden/calib"); ap.add_argument("--data", default="/data/w218/data218.json")
    ap.add_argument("--layers", default="8,23,40,55"); ap.add_argument("--R", default="/data/w218/runs/r0")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    log = P.log_to(os.path.join(a.out, "fd.log")); log("fd218.py " + json.dumps(vars(a)))
    stu = P.build(a, log); dev = stu.devs[0]
    z, gold = P.load_data(a); docs = z["tune"][:a.docs]
    qe = json.load(open("/data/w218/runs/qerr.json"))
    gmean = json.load(open(f"{a.R}/enc/grad/grad.json"))["tensors"]
    # the perturbation: per tensor a list of (A [k, r] f32, B [r, n] f32, coef); y_gemm += coef * (x A) B
    for k in stu.probe:
        ql = stu.qlin[k]; ql.lr = None
        def fwd(self, x):
            sh = x.shape
            x = (x.reshape(-1, self.k) * self.su()).to(BF)
            x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
            y = x @ qmodel.i8_to_bf16(self.wq)
            if self.lr is not None and self.lr[2] != 0:
                A, B, c = self.lr
                y = (y.float() + c * ((x.float() @ A) @ B)).to(BF)
            y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
            y = y * (self.sv() * self.s)
            return y.to(BF).view(*sh[:-1], self.n)
        ql.forward = types.MethodType(fwd, ql)
    def kl():
        r = P.eval_docs(stu, gold, docs, a.cap); return sum(x[0] for x in r) / sum(x[1] for x in r)
    res = {"docs": docs, "cap": a.cap}
    t0 = time.time(); k0 = kl(); res["kl0"] = k0; log(f"KL0 {k0:.6f} ({time.time() - t0:.0f} s)")
    blocks = sorted({int(k.split(".")[3]) for k in stu.probe})
    for d in a.dirs.split(","):
        meta = json.load(open(f"{a.R}/{d}/grad.json"))["tensors"]
        fac, slope, sl_blk = {}, 0.0, {b: 0.0 for b in blocks}
        for k in stu.probe:
            m = meta[k]; kk, n = m["k"], m["n"]
            Dm = torch.from_numpy(np.fromfile(f"{a.R}/{d}/{k}.g", np.float32).reshape(kk, n)).to(dev)
            unit = qe[k]["delta"] / m["g_rms"]                      # W_hat(eta) = W_hat - eta unit D
            G = torch.from_numpy(np.fromfile(f"{a.R}/enc/grad/{k}.g", np.float32).reshape(kk, n)).to(dev)
            s = -float((G.double() * Dm.double()).sum()) * unit     # dKL / deta from the stored gradient
            slope += s; sl_blk[int(k.split(".")[3])] += s
            U, S, V = torch.svd_lowrank(Dm, q=8, niter=6)
            r = int((S > S[0] * 1e-4).sum()); A = (U[:, :r] * S[:r]).contiguous(); B = V[:, :r].T.contiguous()
            err = float((A @ B - Dm).norm() / Dm.norm())
            ql = stu.qlin[k]
            fac[k] = (A, B, unit / ql.s, err)                         # GEMM units: y_gemm = x m, W_hat = s m
            del Dm, G
        log(f"{d}: predicted slope dKL/deta {slope:.4e} (per block " + ", ".join(f"{b}: {v:.3e}" for b, v in sl_blk.items())
            + f"); factor rel err max {max(v[3] for v in fac.values()):.1e}")
        eps = a.rel_eps * k0 / max(abs(slope), 1e-30)
        def setall(eta, only=None):
            for k in stu.probe:
                A, B, c, _ = fac[k]
                on = only is None or int(k.split(".")[3]) == only
                stu.qlin[k].lr = (A, B, -eta * c) if on else None
        pts = {}
        for m_ in (-2, -1, 1, 2):
            setall(m_ * eps); pts[m_] = kl(); log(f"{d} eta {m_ * eps:+.4e}: KL {pts[m_]:.6f} (d {pts[m_] - k0:+.3e})")
        xs = np.array([-2, -1, 0, 1, 2]) * eps; ys = np.array([pts[-2], pts[-1], k0, pts[1], pts[2]])
        c2, c1, c0 = np.polyfit(xs, ys, 2)
        fd = (pts[1] - pts[-1]) / (2 * eps)
        star = -c1 / (2 * c2) if c2 > 0 else None
        blk = {}
        for b in blocks:
            setall(eps, b); kp = kl(); setall(-eps, b); km = kl()
            blk[b] = {"kl_plus": kp, "kl_minus": km, "fd_slope": (kp - km) / (2 * eps), "pred_slope": sl_blk[b]}
            log(f"{d} block {b}: KL(+eps) {kp:.6f} KL(-eps) {km:.6f}  FD slope {(kp - km) / (2 * eps):.4e} vs predicted {sl_blk[b]:.4e}")
        setall(0)
        res[d] = {"eps": eps, "pred_slope": slope, "fd_slope": fd, "points": {str(k): v for k, v in pts.items()}, "parabola": [c2, c1, c0],
                  "eta_star": star, "kl_at_star_pred": float(np.polyval([c2, c1, c0], star)) if star is not None else None, "blocks": blk}
        log(f"{d}: FD slope {fd:.4e} vs predicted {slope:.4e}; parabola a {c2:.4e} b {c1:.4e} -> eta* {star} "
            f"(predicted KL there {res[d]['kl_at_star_pred']})")
        json.dump(res, open(os.path.join(a.out, "fd.json"), "w"), indent=1)
    log("FD DONE")


qmodel = P.qmodel
if __name__ == "__main__":
    main()
