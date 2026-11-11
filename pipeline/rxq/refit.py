# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The post-LDLQ scale polish (quant.refit, #138) and the per-tensor job diagnostics (#139), w98 2026-11-09.

install(Q) wraps exllamav3's refit_scales (outside engine's OOM wrapper) and ldlq / ldlq_2hess (outside tile CD and the
#117 block order). quant.quantize_tensor opens a context per tensor (begin / end); without one every wrapper is a
plain pass-through.

Refit modes (quant.refit, stamped):
- true: the fork's refit_scales as is (2 rounds of a per-output-column svh step and a per-input-row suh step, both in
  the damped H; the suh step weights every output column alike)
- false: no refit (the pre-refit scales are stored)
- "hout": the suh step weighted per output column by w_n = h_n / mean(h) + sigma_out (h = the job's H_out diagonal):
  A = (Q diag(w) Q^T) o H, b = rowsum(Q o HW . w). The svh step is separable, so a diagonal H_out leaves it unchanged.
- "undamped": both steps in H - d_off I, the undamped calib H_in (the error the refit reports already uses it)
- "hout+undamped": both
refit_port() is the fork's arithmetic op for op with these two switches (tests/check_diag.py: mode true == the fork).

Diagnostics (diag, on unless RXQ_DIAG=0; never change a stored tensor), in the refit's error metric H_e = the undamped
calib H_in (H - d_off I, d_off = the damping finalize_capture_H added), original basis, E = W - Wq (k, n), per output
channel n:
  e1 = (E^T H_e E)_nn of the stored weight, e1_pre the same before the refit, r1 = W_n H_e W_n^T,
  x1 = q_n^T H_e e_n, d1 = q_n^T H_e q_n (any later per-channel rescale c: e = e1 - 2 (c-1) x1 + (c-1)^2 d1),
  e2 / e2_pre = h_n e1_n / h_n e1_pre_n (H_out levels, h = the raw H_out diagonal),
  tile = the (k/128, n/128) cell sums of E o (H_e E), su_pre / sv_pre = the pre-refit scales (fp32, signs included).
Scalars (the log record's "diag"): f1 = sum e1 / sum r1 (= the logged proxy_err), f2 = sum h e1 / sum h r1, both before
and after the refit; overload counters from the LDLQ (regularized domain: decoded |q| max, share of |q| >= 0.9 max,
share of targets |w_r| beyond max |q|, share of 16x16 tiles with error > 3x the tensor median, tile p99 / median).
RXQ_REFIT_ARMS=none,hout,... (experiments, #138): the other modes evaluated from the same pre-refit state (same codes),
each with f16-rounded scales as stored; the stored arm is listed as "cur". Their decoded weights (f16, (out, in)) are
kept for the job to write when RXQ_REFIT_ARMS_WQ names a directory."""
import os

import torch

MODES = (True, False, "hout", "undamped", "hout+undamped")
ARM_NAMES = {"none": False, "hout": "hout", "undamped": "undamped", "hout+undamped": "hout+undamped", "cur": True}
_CTX = {}
TILE = 128


def check_mode(m):
    if m not in MODES: raise ValueError(f"quant.refit: one of {MODES}, not {m!r}")
    return m


def begin(key, mode, h=None, sigma_out=0.01, diag=None, arms=None):
    """the context of one tensor's quantize_exl3 call: mode (quant.refit), h = the raw H_out diagonal (n,) or None"""
    _CTX.clear()
    if diag is None: diag = os.environ.get("RXQ_DIAG", "1") != "0"
    if arms is None: arms = [a for a in os.environ.get("RXQ_REFIT_ARMS", "").split(",") if a]
    for a in arms:
        if a not in ARM_NAMES or a == "cur": raise ValueError(f"RXQ_REFIT_ARMS: {a} (known: none, hout, undamped, hout+undamped)")
    _CTX.update(key=key, mode=check_mode(mode), h=None if h is None else h.flatten().double().cpu(), sigma_out=float(sigma_out),
                diag=bool(diag), arms=list(arms), out={})
    return _CTX


def end():
    out = _CTX.get("out", {})
    _CTX.clear()
    return out


def _w_out(h, sigma_out, dev):
    """the suh step's output weights: h / mean(h) + sigma_out"""
    w = h.to(dev, torch.float)
    return w / w.mean().clamp_min(1e-30) + sigma_out


def refit_port(weight, weight_q, H, su, sv, rounds=2, chunk=16384, damping=0.0, undamped=False, w_out=None):
    """the fork's refit_scales (exllamav3 fork quantize.py, b0052f7c) with two switches: undamped (fit both steps in
    H - d_off I) and w_out (n,) (the suh step weighted per output column). Both off: the fork's arithmetic op for op."""
    dev = weight_q.device
    H = H.to(dev)
    if damping > 0.0:
        d_off = damping * H.diagonal().mean().item()
        H_err = H.clone()
        H_err.diagonal().sub_(d_off)
    else:
        H_err = H
    Hf = H_err if undamped else H
    def cols(x, a, b):
        return (x[:, a:b] if x.device == dev else x[:, a:b].to(dev)).float()
    k, n = weight_q.shape
    Q = weight_q
    su = su.float().to(dev).view(k, 1)
    sv = sv.float().to(dev).view(1, n)
    HW = torch.empty((k, n), dtype = torch.float, device = dev)
    den = 0.0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        wc = cols(weight, a, b)
        HW[:, a:b] = Hf @ wc
        den += (wc * HW[:, a:b]).sum().item()
    den_err = den
    if H_err is not Hf:
        den_err = 0.0
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            wc = cols(weight, a, b)
            den_err += (wc * (H_err @ wc)).sum().item()
    def err(Qx):
        e = 0.0
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            E = cols(weight, a, b) - Qx[:, a:b]
            e += (E * (H_err @ E)).sum().item()
        return e / max(den_err, 1e-30)
    e_before = err(Q)
    for _ in range(rounds):
        num = torch.zeros(n, device = dev); dn = torch.zeros(n, device = dev)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            HQ = Hf @ Q[:, a:b]
            num[a:b] = (Q[:, a:b] * HW[:, a:b]).sum(0)
            dn[a:b] = (Q[:, a:b] * HQ).sum(0)
        c = torch.where(dn > 1e-30, num / dn.clamp(min = 1e-30), torch.ones_like(num))
        Q = Q.mul_(c.view(1, n))
        sv = sv * c.view(1, n)
        if w_out is None:
            A = (Q @ Q.T) * Hf
        else:
            A = torch.zeros((k, k), dtype = torch.float, device = dev)
            for a in range(0, n, chunk):
                A.addmm_(Q[:, a:a + chunk] * w_out[a:a + chunk].view(1, -1), Q[:, a:a + chunk].T)
            A *= Hf
        b_ = torch.zeros(k, device = dev)
        for a in range(0, n, chunk):
            if w_out is None: b_ += (Q[:, a:a + chunk] * HW[:, a:a + chunk]).sum(1)
            else: b_ += (Q[:, a:a + chunk] * HW[:, a:a + chunk] * w_out[a:a + chunk].view(1, -1)).sum(1)
        A.diagonal().add_(1e-6 * A.diagonal().mean())
        r = torch.linalg.solve(A, b_)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        Q = Q.mul_(r.view(k, 1))
        su = su * r.view(k, 1)
    e_after = err(Q)
    return Q, su, sv, e_before, e_after


def _stats(W, Q, H, d_off, h, chunk=16384, full=True):
    """per output channel (fp64 on the CPU): e (E^T H_e E)_nn and, with full, r / x / d and the tile map; H_e = H - d_off I"""
    dev = Q.device
    k, n = Q.shape
    e = torch.empty(n, dtype=torch.float64); r = x = d = tile = None
    if full:
        r, x, d = torch.empty_like(e), torch.empty_like(e), torch.empty_like(e)
        tile = torch.zeros(((k + TILE - 1) // TILE, (n + TILE - 1) // TILE), dtype=torch.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Wc = (W[:, a:b] if W.device == dev else W[:, a:b].to(dev)).float()
        Qc = Q[:, a:b]
        E = Wc - Qc
        HE = H @ E
        if d_off: HE.add_(E, alpha=-d_off)                             # H_e E = H E - d_off E
        M = E * HE
        e[a:b] = M.sum(0).double().cpu()
        if full:
            HW = H @ Wc
            if d_off: HW.add_(Wc, alpha=-d_off)
            r[a:b] = (Wc * HW).sum(0).double().cpu()
            x[a:b] = (Qc * HE).sum(0).double().cpu()
            d[a:b] = (Qc * (HW - HE)).sum(0).double().cpu()             # H_e Q = H_e W - H_e E
            kp, np_ = tile.shape[0] * TILE - k, (b - a + TILE - 1) // TILE * TILE - (b - a)
            Mp = torch.nn.functional.pad(M, (0, np_, 0, kp)) if kp or np_ else M
            tile[:, a // TILE:a // TILE + Mp.shape[1] // TILE] = Mp.view(Mp.shape[0] // TILE, TILE, Mp.shape[1] // TILE, TILE).sum((1, 3)).double().cpu()
            del HW, Mp
        del Wc, E, HE, M
    return {"e": e, "r": r, "x": x, "d": d, "tile": tile}


def _f(e, r, h):
    f1 = float(e.sum() / r.sum().clamp_min(1e-300))
    f2 = None if h is None else float((h * e).sum() / (h * r).sum().clamp_min(1e-300))
    return f1, f2


def _try_dev(fn, *ts):
    """fn(*ts) on the tensors' device; on a CUDA OOM again on the CPU"""
    try:
        return fn(*ts)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fn(*[t.cpu() if torch.is_tensor(t) else t for t in ts])


def install(Q):
    """wrap Q.refit_scales (the engine's OOM wrapper inside) and Q.ldlq / Q.ldlq_2hess (outermost)"""
    if getattr(Q, "_rxq_refit", False): return
    inner, ldlq, ldlq2 = Q.refit_scales, Q.ldlq, Q.ldlq_2hess

    def refit_scales(weight, weight_q, H, su, sv, rounds=2, chunk=16384, damping=0.0):
        ctx = _CTX
        if not ctx: return inner(weight, weight_q, H, su, sv, rounds=rounds, chunk=chunk, damping=damping)
        mode, out, hraw = ctx["mode"], ctx["out"], ctx["h"]
        dev = weight_q.device
        try: H = H.to(dev)
        except torch.OutOfMemoryError: torch.cuda.empty_cache()
        d_off = damping * H.diagonal().mean().item() if damping > 0.0 else 0.0
        k, n = weight_q.shape
        su_pre = su.float().flatten().clone(); sv_pre = sv.float().flatten().clone()
        need_pre = ctx["diag"] or mode is False or ctx["arms"]
        pre = _try_dev(lambda W_, Q_, H_: _stats(W_, Q_, H_, d_off, None, chunk, full=True), weight, weight_q, H) if need_pre else None
        Q_pre = weight_q.clone() if ctx["arms"] else None
        if mode is True:
            Qf, su_f, sv_f, e_b, e_a = inner(weight, weight_q, H, su, sv, rounds=rounds, chunk=chunk, damping=damping)
        elif mode is False:
            f1 = float(pre["e"].sum() / pre["r"].sum().clamp_min(1e-300))
            Qf, su_f, sv_f, e_b, e_a = weight_q, su.float().view(k, 1), sv.float().view(1, n), f1, f1
        else:
            Qf, su_f, sv_f, e_b, e_a = _arm(weight, weight_q, H, su, sv, rounds, chunk, damping, mode, ctx)
        if ctx["diag"]:
            post = _try_dev(lambda W_, Q_, H_: _stats(W_, Q_, H_, d_off, None, chunk, full=True), weight, Qf, H)
            h = hraw
            f1p, f2p = _f(pre["e"], pre["r"], h); f1, f2 = _f(post["e"], post["r"], h)
            out.update(e1=post["e"], e1_pre=pre["e"], r1=post["r"], x1=post["x"], d1=post["d"], tile=post["tile"],
                       su_pre=su_pre.cpu(), sv_pre=sv_pre.cpu(), d_off=d_off, damping=damping)
            if h is not None: out.update(e2=h * post["e"], e2_pre=h * pre["e"], h=h)
            out["traj"] = {"pre_refit": {"f1": f1p, "f2": f2p}, "post_refit": {"f1": f1, "f2": f2}}
            out["check"] = {"proxy_err": float(e_a), "sum_e_over_r": f1, "rel": abs(f1 - float(e_a)) / max(abs(float(e_a)), 1e-300),
                            "e_before": float(e_b), "pre_rel": abs(f1p - float(e_b)) / max(abs(float(e_b)), 1e-300)}
        if ctx["arms"]:
            out["arms"], out["arm_wq"] = {}, {}
            for name in ["cur"] + ctx["arms"]:
                m = ARM_NAMES[name]
                if m == mode: s_u, s_v = su_f, sv_f
                elif m is False: s_u, s_v = su.float().view(k, 1), sv.float().view(1, n)
                else: _, s_u, s_v, _, _ = _arm(weight, Q_pre.clone(), H, su, sv, rounds, chunk, damping, m, ctx)
                s_u16 = s_u.flatten().half().float().to(dev); s_v16 = s_v.flatten().half().float().to(dev)
                Qa = Q_pre * (s_u16 / su_pre.to(dev)).view(k, 1) * (s_v16 / sv_pre.to(dev)).view(1, n)
                st = _try_dev(lambda W_, Q_, H_: _stats(W_, Q_, H_, d_off, None, chunk, full=False), weight, Qa, H)
                rr = pre["r"]
                a1, a2 = _f(st["e"], rr, hraw)
                out["arms"][name] = {"mode": m, "f1": a1, "f2": a2, "su_zero": int((s_u16 == 0).sum()), "sv_zero": int((s_v16 == 0).sum())}
                if os.environ.get("RXQ_REFIT_ARMS_WQ"): out["arm_wq"][name] = Qa.T.half().contiguous().cpu()
                del Qa
            del Q_pre
        return Qf, su_f, sv_f, e_b, e_a

    def _ldlq_ov(fn):
        def w(weight, *a, **kw):
            wq, enc = fn(weight, *a, **kw)
            if _CTX and _CTX["diag"]:
                try: _CTX["out"]["overload"] = _try_dev(_overload, weight, wq)
                except Exception as ex: _CTX["out"]["overload"] = {"error": repr(ex)[:200]}
            return wq, enc
        return w

    Q.refit_scales = refit_scales
    Q.ldlq, Q.ldlq_2hess = _ldlq_ov(ldlq), _ldlq_ov(ldlq2)
    Q._rxq_refit = True


def _arm(weight, weight_q, H, su, sv, rounds, chunk, damping, mode, ctx):
    hw = "hout" in str(mode)
    if hw and ctx["h"] is None: raise SystemExit(f"quant.refit = {mode!r}: {ctx['key']} has no H_out diagonal")
    w_out = _w_out(ctx["h"], ctx["sigma_out"], weight_q.device) if hw else None
    return refit_port(weight, weight_q, H, su, sv, rounds=rounds, chunk=chunk, damping=damping,
                      undamped="undamped" in str(mode), w_out=w_out)


def _overload(w_r, q):
    """LDLQ overload counters in the regularized domain (q = the decoded codes incl. tile CD, w_r = the regularized
    weight LDLQ started from, before error feedback)"""
    dev = q.device
    w_r = w_r.to(dev) if w_r.device != dev else w_r
    qa = q.abs(); qmax = float(qa.max())
    k, n = q.shape
    t = (w_r - q).square().view(k // 16, 16, n // 16, 16).sum((1, 3)).flatten()
    t = t.double().cpu(); med = float(t.median())
    return {"q_absmax": qmax, "q_outer": float((qa >= 0.9 * qmax).float().mean()), "w_over": float((w_r.abs() > qmax).float().mean()),
            "tile_3x": float((t > 3 * med).double().mean()) if med > 0 else None,
            "tile_p99_med": float(t.sort().values[int(0.99 * (t.numel() - 1))]) / med if med > 0 else None}


def tensors(key, out):
    """the diag file's tensors (fp32) from end()'s dict"""
    names = ("e1", "e2", "e1_pre", "e2_pre", "x1", "d1", "r1", "tile", "su_pre", "sv_pre")
    return {f"{key}.{n}": out[n].float().contiguous() for n in names if out.get(n) is not None}


def rescaled(out, c):
    """per-channel error after a per-output-channel rescale c (C9 / swj): e1 - 2 (c-1) x1 + (c-1)^2 d1"""
    c1 = c.double().cpu() - 1.0
    return out["e1"] - 2 * c1 * out["x1"] + c1 * c1 * out["d1"]
