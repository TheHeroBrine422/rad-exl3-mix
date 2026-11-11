#!/usr/bin/env python3
"""job102.py d3|b8 LAYER TENSOR SAMPLE OUTDIR [DONE] -- w102 (../w102.md) on w87's two-sided held-out proxy.

  d3  #136-01 (D-3): per K of W102_D3_KS (default 0.125, 0.5, 1), i8, today's sigma: the same tensor block quantized by
      two-sided LDLQ (settle87.ldlq2, bn1920 H_out) with the tile search
        F    free start (trellis.vit: the proxy's usual search; equals w95's i8 rows, gate G0)
        T1   tail-biting, exllamav3's search (NTRY 1)               (ring102.c, n = 1)
        T16  tail-biting, 16 wrap candidates (NTRY 16)
      and by one-sided block LDLQ (settle77.ldlq_cached: the 8 row tiles of a 16-column block searched together, so
      #102's chains are possible) with
        O0   free start, O1 tail-biting NTRY 1 (n = 1), O4 chained tail-biting, one ring per 4 row tiles (NTRY 1)
      Scored as settle87 (per-output-channel refit under H_A + delta I; two-sided e2 and one-sided e1 under H_B1, H_B2,
      H_A). One file per K: OUTDIR/LAYER-TENSOR-sS-K<K>.jsonl (one line per arm; skipped if present).
  b8  #136-02 (B-8): per K of W102_B8_KS (default 0.5, 2.5, 4), i8, today's sigma: settle87's two-sided quantization (free
      start, = w87 / w95's i8 rows, gate G0), then the stored scales rounded after the refit: su (input channel, all
      Kd) and sv_final = sv r c (output channel: sign x row RMS x refit) to
        fp16 (today's storage), L8 = sign + 7-bit log2 magnitude on a uniform log grid from the vector's min to max |s|
        (primary), L8su / L8sv (one side only), L9 = sign + 8-bit log2, E4M3 = fp8 e4m3 x one fp32 scale per vector
        (max |s| -> 448), L8r = su to L8 first, the output-channel refit re-solved on it, then sv_final to L8.
      sv's grid range uses every row of the tensor: [min_N r x min_block c, max_N r x max_block c].
      One file per K: OUTDIR/LAYER-TENSOR-sS-K<K>.jsonl (one line, every arm).
  DONE (optional) is touched when every K file exists. Env as settle87 (E3_CKPT, E3_HESS, SETTLE_CACHE, W77_SCALES,
  SETTLE_ORDER=rev, W77_HA / W77_HB1 / W77_HB2, W87_HOUT)."""
import ctypes, json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w87")]
import settle87  # noqa: E402  (imports settle77 / settle64 / settle / trellis / trellis2 / e3, patched as in w87)
from settle87 import settle77, settle, trellis, trellis2, e3  # noqa: E402
torch.set_grad_enabled(False)

R = ctypes.CDLL(os.path.join(_h, "libring102.so"))
R.ring_tb.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
D3_KS = os.environ.get("W102_D3_KS", "0.125,0.5,1").split(",")
B8_KS = os.environ.get("W102_B8_KS", "0.5,2.5,4").split(",")


def ring(t, cb, K, n, ntry):
    """t [ntiles, 256] -> values, chains of n consecutive tiles (one ring each), ntry wrap candidates (0 = free)"""
    nt = t.shape[0]; assert nt % n == 0, (nt, n)
    x = np.ascontiguousarray(t, dtype=np.float32).reshape(nt // n, 256 * n); out = np.empty_like(x)
    p = np.ascontiguousarray(trellis.pattern(K), dtype=np.int32)
    R.ring_tb(x.ctypes.data, out.ctypes.data, None, nt // n, 256 * n, cb.ctypes.data, p.ctypes.data, len(p), ntry)
    return out.reshape(nt, 256)


def ldlq1(W, F, qfun):
    """settle77.ldlq_cached with the tile search qfun(tiles [R/16, 256]) for all row tiles of a 16-column block"""
    perm, dead, U, Ui = F
    W = W[:, perm].contiguous(); Rr, Kd = W.shape; W[:, dead] = 0
    Q = torch.zeros_like(W)
    for a in range(0, Kd, 16):
        blk = W[:, a:a + 16]
        tiles = blk.reshape(Rr // 16, 16, 16).reshape(Rr // 16, 256).numpy()
        q = torch.from_numpy(np.asarray(qfun(tiles), dtype=np.float64)).reshape(Rr, 16)
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ Ui[a // 16]
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    out = torch.empty_like(Q); out[:, perm] = Q
    return out


class Ctx:
    def __init__(s, layer, tensor, sample):
        s.layer, s.tensor, s.sample = layer, tensor, sample
        s.p = p = settle87.prep_nohess(layer, tensor, sample)
        hn = e3.ROUTE[tensor][1]
        s.Lout, s.hw, s.hinfo = settle87.hout_metric(layer, tensor, p, settle87.SIGMA_OUT)
        s.HA = settle77.load_h(os.environ["W77_HA"], layer, hn); p.H = s.HA
        s.Hs = {"B1": settle77.load_h(os.environ["W77_HB1"], layer, hn).float(),
                "B2": settle77.load_h(os.environ["W77_HB2"], layer, hn).float(), "A": s.HA}
        p.Wo = p.Wo.double()
        s.ref2, s.ref1 = settle87.refs(p, s.Hs, s.hw)
        s._sig = None

    def metric(s, sig, one_sided):
        """(L_in for ldlq2, ldlq_prep factor for ldlq_cached) of the damped rotated H_A, one sigma cached at a time"""
        if s._sig != sig:
            s.Lin = s.F1 = None; s._sig = sig
        if s.Lin is None:
            s.Lin = settle87.in_metric(s.p, sig)
        if one_sided and s.F1 is None:
            Hq, dq = s.p.metric(sig); assert dq == 0
            s.F1 = settle77.ldlq_prep(Hq); del Hq
        return s.Lin, s.F1

    def rel(s, e):
        o = {k: e[k] / s.ref2[k] for k in s.Hs}; o["B"] = (e["B1"] + e["B2"]) / (s.ref2["B1"] + s.ref2["B2"]); return o

    def rel1(s, e):
        o = {k: e[k] / s.ref1[k] for k in s.Hs}; o["B"] = (e["B1"] + e["B2"]) / (s.ref1["B1"] + s.ref1["B2"]); return o

    def head(s, K, sig, scale):
        p = s.p
        return {"exp": "w102", "layer": s.layer, "w": s.tensor, "group": "input" if s.tensor in settle.INPUT else "rest",
                "sample": s.sample, "cb": "i8", "bits": K, "sigma": sig, "s_exl": scale["s_exl"], "drift": scale["drift"],
                "N": p.N, "Kd": p.Kd, **s.hinfo, **{k: v for k, v in p.extra.items() if k != "blocks"}}


def write(path, lines):
    with open(path + ".part", "w") as fh:
        for o in lines: fh.write(json.dumps(o) + "\n")
    os.replace(path + ".part", path)


def d3(ctx, Ks, f):
    K = float(Fraction(Ks)); sig = settle87.sigma_current(ctx.tensor, K)
    sc = settle87.get_scale(ctx.layer, ctx.tensor, ctx.sample, "i8", Ks); s = sc["s_exl"]
    cb = np.ascontiguousarray(trellis2.codebook(settle77.CBN["i8"]), dtype=np.float32)
    p = ctx.p; lines = []
    arms = [("F", 2, lambda t: trellis.vit(t, cb, K)), ("T1", 2, lambda t: ring(t, cb, K, 1, 1)),
            ("T16", 2, lambda t: ring(t, cb, K, 1, 16)), ("O0", 1, lambda t: ring(t, cb, K, 1, 0)),
            ("O1", 1, lambda t: ring(t, cb, K, 1, 1)), ("O4", 1, lambda t: ring(t, cb, K, 4, 1))]
    for arm, sides, qf in arms:
        t0 = time.time(); Lin, F1 = ctx.metric(sig, sides == 1); tm = time.time() - t0
        t0 = time.time()
        Q = (settle87.ldlq2(p.Wt * s, Lin, ctx.Lout, qf) if sides == 2 else ldlq1(p.Wt * s, F1, qf)) / s
        tq = time.time() - t0
        e = settle87.scores(p, Q, sig, p.H, ctx.Hs, ctx.hw); del Q
        lines.append({**ctx.head(K, sig, sc), "item": "d3", "arm": arm, "ldlq": "two-sided" if sides == 2 else "one-sided",
                      "rel": ctx.rel(e["e2"]), "rel1": ctx.rel1(e["e1"]), "sec": round(tq, 1), "sec_metric": round(tm, 1)})
        print(json.dumps(lines[-1]), flush=True)
    write(f, lines)


# ---- B-8 scale formats
def q_log(v, bits, lo, hi):
    """sign kept, |v| to the nearest point (in log2) of a uniform log2 grid of 2^bits points on [lo, hi]"""
    a = v.abs().clamp(min=lo, max=hi); l0, l1 = math.log2(lo), math.log2(hi); n = (1 << bits) - 1
    step = (l1 - l0) / n if l1 > l0 else 1.0
    m = torch.round((torch.log2(a) - l0) / step).clamp(0, n)
    return torch.sign(v) * torch.exp2(l0 + m * step)


def q_e4m3(v, hi):
    sc = hi / 448.0
    return (v / sc).float().to(torch.float8_e4m3fn).double() * sc


def q_f16(v):
    return v.half().double()


def b8(ctx, Ks, f):
    K = float(Fraction(Ks)); sig = settle87.sigma_current(ctx.tensor, K)
    sc = settle87.get_scale(ctx.layer, ctx.tensor, ctx.sample, "i8", Ks); s = sc["s_exl"]
    cb = np.ascontiguousarray(trellis2.codebook(settle77.CBN["i8"]), dtype=np.float32)
    p = ctx.p; Lin, _ = ctx.metric(sig, False)
    t0 = time.time(); Q = settle87.ldlq2(p.Wt * s, Lin, ctx.Lout, settle87.vit_fun(cb, K)) / s; tq = time.time() - t0
    base = settle87.scores(p, Q, sig, p.H, ctx.Hs, ctx.hw)          # = w87 / w95 (G0)
    HA = p.H; dl = sig * HA.diagonal().mean()
    Z = settle.had_cols(Q)                                          # input side before su
    su = p.su.double(); svr = (p.sv * p.r).double()[:, 0]           # [Kd], [128]

    def refit(su_):
        q = settle.had_rows(Z * su_) * svr[:, None]; hq = q @ HA
        c = ((hq * p.Wo).sum(1) + dl * (q * p.Wo).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
        return c

    c = refit(su); svf = svr * c                                    # the stored output-channel scale after the refit
    key = settle87.hout_key(ctx.layer, ctx.tensor, p); rN = settle87.row_rms(ctx.layer, key)
    lo_v, hi_v = float(rN.min() * c.abs().min()), float(rN.max() * c.abs().max())
    lo_u, hi_u = float(su.abs().min()), float(su.abs().max())

    def err(su_, sv_):
        D = p.Wo - settle.had_rows(Z * su_) * sv_[:, None]; o2, o1 = {}, {}
        for k, H in ctx.Hs.items():
            Dh = D.to(H.dtype); e = ((Dh @ H) * Dh).sum(1).double()
            o2[k] = float((ctx.hw * e).sum()); o1[k] = float(e.sum())
        return o2, o1

    arms = {"exact": (su, svf), "fp16": (q_f16(su), q_f16(svf)),
            "L8": (q_log(su, 7, lo_u, hi_u), q_log(svf, 7, lo_v, hi_v)),
            "L8su": (q_log(su, 7, lo_u, hi_u), q_f16(svf)), "L8sv": (q_f16(su), q_log(svf, 7, lo_v, hi_v)),
            "L9": (q_log(su, 8, lo_u, hi_u), q_log(svf, 8, lo_v, hi_v)),
            "E4M3": (q_e4m3(su, hi_u), q_e4m3(svf, hi_v))}
    su8 = q_log(su, 7, lo_u, hi_u); arms["L8r"] = (su8, q_log(svr * refit(su8), 7, lo_v, hi_v))
    o = {**ctx.head(K, sig, sc), "item": "b8", "rel_base": ctx.rel(base["e2"]), "sec": round(tq, 1),
         "su_range": [lo_u, hi_u], "sv_range": [lo_v, hi_v], "c_range": [float(c.min()), float(c.max())], "arms": {}}
    for a, (u, v) in arms.items():
        e2, e1 = err(u, v)
        o["arms"][a] = {"rel": ctx.rel(e2), "rel1": ctx.rel1(e1),
                        "su_maxrel": float(((u - su) / su).abs().max()), "sv_maxrel": float(((v - svf) / svf).abs().max())}
    print(json.dumps({k: v for k, v in o.items()}), flush=True)
    write(f, [o])


def main():
    item, layer, tensor, sample, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    done = sys.argv[6] if len(sys.argv) > 6 else None
    Ks = {"d3": D3_KS, "b8": B8_KS}[item]
    os.makedirs(outdir, exist_ok=True)
    path = lambda k: os.path.join(outdir, f"{layer}-{tensor}-s{sample}-K{k}.jsonl")
    todo = [k for k in Ks if not os.path.exists(path(k))]
    if todo:
        ctx = Ctx(layer, tensor, sample)
        for k in todo: (d3 if item == "d3" else b8)(ctx, k, path(k))
    if done and all(os.path.exists(path(k)) for k in Ks): open(done, "w").close()


if __name__ == "__main__":
    main()
