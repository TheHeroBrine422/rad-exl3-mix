#!/usr/bin/env python3
"""settle.py MODE LAYER TENSOR K OUT.jsonl [ROWS] -- w43 CPU proxies for open V3 quantizer settings (README.md here).

Proxy (SETTLE_PROXY, default 2; README "Tool"): block LDLQ + bitshift trellis as trellis2.py (fractional K = shift
pattern, free start state), codebook mul1_i8 (SETTLE_CB), proxy error rel = tr(dW H dW^T) / tr(W H W^T) under the
UNDAMPED calib H_in. v2 (class P2) applies exllamav3's regularize transforms to ROWS (default 128 = one aligned
output-Hadamard block; E3_ROWSEED picks it, E3_SEED the signs) and quantizes under its scaled Hessian; v1 (class P1)
is trellis2.py's random rows with the input Hadamard only.

The global scale is the one exllamav3 picks (g_scale_search_batch on its tile sample, x LDLQ_DRIFT; quantize.py of
the fork eb5a2139; search_scale): the trellis quantizes the transformed rows x s_exl x m, i.e. m is exactly ourq.py's
--gain multiplier on g_scale. After LDLQ exllamav3 refits the stored scales (refit_scales: per output channel in
closed form under the damped H, then per input channel, 2 rounds). The per-output-channel step is emulated exactly
(one closed-form scale per original row): "rel" is the error after it (primary), "rel_norefit" before. The
input-channel step is not (5k+ free scales fitted on 128 rows would be in-sample).

Damping (sigma_reg x mean diag H, exllamav3's convention) defaults to the adopted trunk setting: 0.005 for the
layer-input group (ssm_qkv, ssm_z, attn_qg, attn_k, attn_v), 0.025 for the rest.

MODE
  gain  rel at every multiplier m of SETTLE_M (default 0.6,0.7,0.75,...,1.2,1.3), adopted damping; one LDLQ pass, m stacked
  damp  rel at each damping of SETTLE_D (default: input group 0.0025,0.005,0.01, rest 0.0125,0.025,0.05) x m in
        SETTLE_DM (default 0.9,1,1.1)
  cd    m = 1, adopted damping: LDLQ, then CD_SWEEPS (default 3) tile coordinate-descent sweeps (trellis2.ldlq_cd);
        rel and CPU seconds per stage
  selftest  the transforms invert exactly (prints the max error)
One JSON line per (mode, damping): "rel" = {m: rel} (gain / damp) or {stage: rel} (cd). OUT is written whole at the
end (OUT.part, then renamed): an existing OUT means the job is done."""
import fcntl, json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, "..")]
import e3, trellis, trellis2
torch.set_grad_enabled(False)

INPUT = ("ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v")
DRIFT = {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}       # exllamav3 LDLQ_DRIFT (fork eb5a2139)
FL = lambda s: [float(v) for v in s.split(",")]


CBS = 1.24371088                                           # exllamav3 codebook_scale: its weight units = ours x CBS


def search_scale(T, cb, K):
    """exllamav3's g_scale_search_batch on sampled tiles T [n, 256] (before the drift), in its own units and grids
    (regularize leaves column RMS = codebook_scale against a codebook of sd ~1; ours: RMS 1, sd 1): coarse 0.1 .. 1.9
    step 0.2 on every 3rd tile, fine c + 0.075 (i - 2) on all, parabola clipped to half a step. Returns the input
    scale in our units (= CBS x exllamav3's g_scale) and the search record"""
    T = (T * DRIFT.get(float(K), 1.0)).astype(np.float32)
    def mse(tiles, ss):                                    # ss in exllamav3 units
        q = trellis.vit(np.concatenate([tiles * (CBS * s) for s in ss]), cb, K).reshape(len(ss), *tiles.shape)
        return [float(((q[i] / (CBS * s) - tiles) ** 2).mean()) for i, s in enumerate(ss)]
    coarse = [0.1 + 0.2 * i for i in range(10)]
    c = coarse[int(np.argmin(mse(T[::3], coarse)))]
    fine = [c + 0.075 * (i - 2) for i in range(5)]
    y = mse(T, fine); b = int(np.argmin(y)); off = 0.0
    if 0 < b < 4:
        den = y[b - 1] - 2 * y[b] + y[b + 1]
        off = max(-0.5, min(0.5, 0.5 * (y[b - 1] - y[b + 1]) / den)) if den > 0 else 0.0
    g = max(fine[b] + off * 0.075, 0.01)
    return CBS * g, {"g_scale": g, "coarse_best": round(c, 3), "fine_mse": y, "tiles": len(T)}


def tiles_of(X):
    """X [R, Kd] -> its 16x16 tiles [R/16 * Kd/16, 256]"""
    R, Kd = X.shape
    return X.reshape(R // 16, 16, Kd // 16, 16).permute(0, 2, 1, 3).reshape(-1, 256)


_h = None
def had128():
    global _h
    if _h is None:
        h = torch.tensor([[1.0]], dtype=torch.float64)
        while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _h = h / math.sqrt(128)
    return _h

def had_cols(X):
    """X @ blockdiag(h): the 128-Hadamard along the last dim (h symmetric and orthogonal: its own inverse)"""
    h = had128().to(X.dtype); sh = X.shape
    return (X.reshape(*sh[:-1], sh[-1] // 128, 128) @ h).reshape(sh)

def had_rows(X):
    """blockdiag(h) @ X: the 128-Hadamard along the rows"""
    return had_cols(X.T).T.contiguous()


class P1:
    """the old proxy (trellis.py / trellis2.py): random rows, input Hadamard, unit-RMS rows; metric in the rotated
    basis; LDLQ damping added inside ldlq (d x mean diag of the rotated H)"""
    def __init__(s, layer, tensor, rows, seed, rseed):
        names, hn = e3.ROUTE[tensor]
        W0 = e3.load_w(layer, names); H0 = e3.load_h(layer, hn)
        s.N, s.Kd = W0.shape
        g = torch.Generator().manual_seed(7 + layer + 100003 * rseed)
        idx = torch.randperm(s.N, generator=g)[:rows].sort()[0]
        rk = e3.had_blocks(s.Kd, 1000 + layer + 7919 * seed)
        s.Wo = rk(W0[idx]); s.H = rk(rk(H0).T).T; del W0, H0
        s.rms = s.Wo.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        s.Wt = s.Wo / s.rms
        T = tiles_of(s.Wt).numpy(); rng = np.random.default_rng(17 + layer + 1000 * seed + 100 * rseed)
        o = np.argsort((T ** 2).mean(1))
        s.T = T[np.unique(np.concatenate([rng.choice(len(T), min(1024, len(T)), replace=False), o[:64], o[-64:]]))]
        s.extra = {}
    def metric(s, d): return s.H, d                        # (H for LDLQ, damping ldlq adds)
    def back(s, Q): return Q * s.rms                       # quantization domain -> scoring domain, per R rows
    def delta(s, d): return d * s.H.diagonal().mean()


class P2:
    """exllamav3's regularize on whole aligned 128-row blocks (README "proxy v2"): output-channel RMS scales with random
    signs, the 128-Hadamard over output channels, input-channel RMS scales su (over the whole tensor) with random signs,
    the input 128-Hadamard; LDLQ under P diag(su) (H + delta I) diag(su) P (--scaled-hessian, damping added to H in
    the original basis as finalize_capture_H does); the g_scale tile sample is exllamav3's (a width-3 wrapped diagonal
    + the extreme-RMS tiles) over the whole transformed tensor, capped at 2,304 tiles by a uniform subsample; scoring
    in the original basis. ffn_gate_up: gate_proj for even seeds, up_proj for odd (exllamav3 quantizes them apart)."""
    KEYS = ("Wo", "Wt", "r", "sv", "su", "T", "N", "Kd", "extra")
    def __init__(s, layer, tensor, rows, seed, rseed):
        assert rows % 128 == 0
        names, hn = e3.ROUTE[tensor]
        s.H = e3.load_h(layer, hn)
        cf = os.path.join(os.environ.get("SETTLE_CACHE", "cache"), f"{layer}-{tensor}-s{seed}-r{rseed}-{rows}.pt")
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        s._lk = open(cf + ".lock", "w"); fcntl.flock(s._lk, fcntl.LOCK_EX)   # one job computes, the others load it
        if os.path.exists(cf):                             # the transformed rows and tile sample of an earlier job
            c = torch.load(cf, weights_only=False)          # our own file (holds a numpy array)
            for k in s.KEYS: setattr(s, k, c[k])
            s._lk.close(); return
        if len(names) > 1: names = [names[rseed % len(names)]]; s.extra = {"part": names[0]}
        else: s.extra = {}
        W0 = e3.load_w(layer, names)
        s.N, s.Kd = N, Kd = W0.shape
        g = torch.Generator().manual_seed(1000 + layer + 7919 * seed)
        sv = (torch.randint(0, 2, (N,), generator=g) * 2 - 1).double()
        su_sign = (torch.randint(0, 2, (Kd,), generator=g) * 2 - 1).double()
        r = W0.pow(2).mean(1).sqrt().clamp(min=1e-12)
        X = W0 / r[:, None]
        su = X.pow(2).mean(0).sqrt(); su[su < 1e-30] = 0.1; s.su = su * su_sign
        # the whole tensor, transformed (float32: only for the scale search's tile sample)
        Xf = had_cols(had_rows((X * sv[:, None]).float()) / s.su.float())
        tn, tk = N // 16, Kd // 16
        w4 = Xf.reshape(tn, 16, tk, 16)
        dl = max(tk, tn); ii = torch.arange(dl).repeat_interleave(3); ww = torch.arange(3).repeat(dl)
        kk = ii % tk; nn = (ii + ww) % tn
        ms = w4.square().mean(dim=(1, 3)).flatten()          # (tn, tk) flattened
        nx = min(max(8, dl * 3 // 16), (ms.numel() + 1) // 2)
        ex = torch.cat((torch.topk(ms, nx).indices, torch.topk(ms, nx, largest=False).indices))
        T = w4[torch.cat((nn, ex // tk)), :, torch.cat((kk, ex % tk)), :].reshape(-1, 256).numpy()
        rng = np.random.default_rng(17 + layer + 1000 * seed + 100 * rseed)
        if len(T) > 2304: T = T[np.sort(rng.choice(len(T), 2304, replace=False))]
        s.T = T; del Xf, w4
        gb = torch.Generator().manual_seed(7 + layer + 100003 * rseed)
        blk = torch.randperm(N // 128, generator=gb)[:rows // 128].sort()[0]
        idx = (blk[:, None] * 128 + torch.arange(128)).reshape(-1)
        s.Wo = W0[idx].clone(); s.r = r[idx][:, None].clone(); s.sv = sv[idx][:, None].clone()
        s.Wt = had_cols(had_rows(X[idx] * s.sv) / s.su)
        s.extra["blocks"] = blk.tolist(); del W0, X
        torch.save({k: getattr(s, k) for k in s.KEYS}, cf + f".part{os.getpid()}"); os.replace(cf + f".part{os.getpid()}", cf)
        s._lk.close()
    def metric(s, d):
        Hd = s.H.clone(); dead = Hd.diagonal() == 0
        Hd[dead, dead] = 1; Hd.diagonal().add_(s.delta(d))
        Hd *= s.su[None, :]; Hd *= s.su[:, None]
        Hd = had_cols(Hd); Hd = had_cols(Hd.T.contiguous()).T.contiguous()
        return Hd, 0.0
    def back(s, Q): return had_rows(had_cols(Q) * s.su) * s.sv * s.r
    def delta(s, d): return d * s.H.diagonal().mean()


def rels(p, Wq, d):
    """(rel after the per-output-channel scale refit under H + delta I, rel before) for quantization-domain
    reconstructions Wq [n*R, Kd] (already divided by the input scale), stacked by R rows; scored under p.H undamped"""
    R = p.Wo.shape[0]; out = []; dl = p.delta(d)
    for i in range(Wq.shape[0] // R):
        q = p.back(Wq[i * R:(i + 1) * R]); hq = q @ p.H
        c = ((hq * p.Wo).sum(1) + dl * (q * p.Wo).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
        out.append((e3.rel(p.Wo, q * c[:, None], p.H), e3.rel(p.Wo, q, p.H)))
    return out


def main():
    mode, layer, tensor, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(Fraction(sys.argv[4])), sys.argv[5]
    rows = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    seed, rseed = int(os.environ.get("E3_SEED", 0)), int(os.environ.get("E3_ROWSEED", 0))
    proxy = int(os.environ.get("SETTLE_PROXY", 2))
    t00 = time.time()
    p = (P1 if proxy == 1 else P2)(layer, tensor, rows, seed, rseed)
    cbn = os.environ.get("SETTLE_CB", "mul1_i8")
    cb = np.ascontiguousarray(trellis2.codebook(cbn), dtype=np.float32)
    fam_in = tensor in INPUT; damp0 = 0.005 if fam_in else 0.025
    t0 = time.time()
    s_exl, sx = search_scale(p.T, cb, K)
    base = {"exp": mode, "proxy": proxy, "layer": layer, "w": tensor, "group": "input" if fam_in else "rest", "bits": K,
            "cb": cbn, "s_exl": s_exl, "drift": DRIFT.get(float(K), 1.0), "search": sx,
            "sec_search": round(time.time() - t0, 1), "seed": seed, "rowseed": rseed, "rows": rows, "N": p.N, "K": p.Kd,
            **p.extra}
    lines = []
    if mode in ("gain", "damp"):
        ms = FL(os.environ.get("SETTLE_M", "0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.3")) if mode == "gain" \
            else FL(os.environ.get("SETTLE_DM", "0.9,1.0,1.1"))
        ds = [damp0] if mode == "gain" else FL(os.environ["SETTLE_D"]) if os.environ.get("SETTLE_D") else \
            ([0.0025, 0.005, 0.01] if fam_in else [0.0125, 0.025, 0.05])
        for d in ds:
            t0 = time.time()
            Hq, dq = p.metric(d)
            Qs = trellis.ldlq_trellis(torch.cat([p.Wt * (s_exl * m) for m in ms], 0), Hq, cb, K, dq); del Hq
            for i, m in enumerate(ms): Qs[i * rows:(i + 1) * rows] /= s_exl * m
            rr = rels(p, Qs, d); del Qs
            lines.append({**base, "damp": d, "damp_adopted": d == damp0, "rel": {str(m): r[0] for m, r in zip(ms, rr)},
                          "rel_norefit": {str(m): r[1] for m, r in zip(ms, rr)}, "sec": round(time.time() - t0, 1)})
    elif mode == "cd":
        Hq, dq = p.metric(damp0); rel, rel0, acc, sec = {}, {}, {}, {}; t0 = [time.time()]
        def report(stage, Q, extra=None):
            sec[stage] = round(time.time() - t0[0], 1)          # the stage's own CPU time, scoring excluded
            rel[stage], rel0[stage] = rels(p, Q / s_exl, damp0)[0]
            if extra: acc[stage] = extra["accepted"]
            t0[0] = time.time()
        trellis2.ldlq_cd(p.Wt * s_exl, Hq, dq, cb, K, int(os.environ.get("CD_SWEEPS", 3)), report)
        lines.append({**base, "damp": damp0, "rel": rel, "rel_norefit": rel0, "accepted": acc, "sec_stage": sec})
    elif mode == "selftest":                                 # transforms invert exactly; rel(W) = 0
        q = p.back(p.Wt); print(json.dumps({"max_abs_err": float((q - p.Wo).abs().max()), "rel_exact": rels(p, p.Wt, 0.0)[0],
                                            "tiles": len(p.T), **p.extra})); return
    else:
        raise SystemExit(f"unknown mode {mode}")
    for o in lines: o["sec_job"] = round(time.time() - t00, 1)
    with open(out + ".part", "w") as fh:
        for o in lines: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)
    for o in lines: print(json.dumps({k: v for k, v in o.items() if k != "search"}), flush=True)


if __name__ == "__main__":
    main()
