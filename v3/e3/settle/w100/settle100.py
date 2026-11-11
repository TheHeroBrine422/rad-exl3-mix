#!/usr/bin/env python3
"""settle100.py -- w100 (../w100.md): sub-1 hygiene (#128, item 4) and the H_out shape sweep (#129, item 8) on w87's
held-out two-sided proxy with today's stack (w93's settle93: #117 block order on the residual readers), i8.

  settle100.py sub1 LAYER TENSOR SAMPLE OUTDIR [DONE]    #128: i8 K0.125 / 0.5 / 1 at today's settings (σ_in current,
        H_out = bn1920 (W100_HF), σ_out 0.01, drift 1.08): arm "cur" (the scale exllamav3's search picks), "ext" (the
        search's coarse grid extended from 0.1..1.9 to 0.1..3.9; quantized only if it picks another scale), "drift"
        (the search with the drift this tensor's "cur" LDLQ measured; quantized only if the scale moves). Per arm:
        two-sided / one-sided rel (held-out B1 / B2, in-sample A, undamped H_pub = A + B1 + B2), C9's per-row e_n and
        c_n (post.c9_rows: M = H_pub, Q after the per-output-channel refit) for the writers, the LDLQ's measured drift.
        One file per level: OUTDIR/LAYER-TENSOR-sS-sub1-K{K}.jsonl
  settle100.py hshape LAYER TENSOR SAMPLE OUTDIR [DONE]  #129: i8 K0.5 / 2.5 / 4, the quantizer's H_out = h_E^p (even
        blocks), σ_out x p grid (W100_SO x W100_P), plus "full" (h_F = bn1920, σ_out 0.01, p 1 = production). Every Q
        scored under H_X (A, B1, B2) x h (E, O1, O2, O, F): per (X, h) the error and the reference totals.
        One file per level: OUTDIR/LAYER-TENSOR-sS-hshape-K{K}.jsonl
  settle100.py selftest OUT.json   gates G1a-d (../w100.md)

The pieces are w87's / w93's unchanged (S87.ldlq2, block_ldl, scores' refit, S93.perm / lin_of); ldlq2s below is
S87.ldlq2 plus sums of squares of the tiles it hands to the trellis (gate G1a: same Q bit for bit)."""
import json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w93")]
import settle93 as S93                                    # also puts w87, w77, settle, settle64, trellis, e3 on the path
import settle87 as S87
import settle77, settle, trellis, trellis2, e3
from safetensors import safe_open
torch.set_grad_enabled(False)

SO = [float(x) for x in os.environ.get("W100_SO", "0.003,0.01,0.03,0.1").split(",")]
PX = [float(x) for x in os.environ.get("W100_P", "0.5,0.75,1").split(",")]
SUB1_K = os.environ.get("W100_SUB1_K", "0.125,0.5,1").split(",")
HSHAPE_K = os.environ.get("W100_HSHAPE_K", "0.5,2.5,4").split(",")
ALPHA = {"ssm_out": 3.0, "attn_output": -1.0, "ffn_down": 0.0}   # v3.toml [post.c9] (gdn_out 3, attn_o -1, down 0)
E_CLIP = 0.5
HDIRS = {"F": "W100_HF", "E": "W100_HE", "O": "W100_HO", "O1": "W100_HO1", "O2": "W100_HO2"}
CB = "i8"
COARSE = [0.1 + 0.2 * i for i in range(10)]
COARSE_EXT = [0.1 + 0.2 * i for i in range(20)]           # 0.1 .. 3.9 (the reciprocal of the "0.5 floor" -> 0.25)


def ldlq2s(X, Lin, Lout, qfun):
    """S87.ldlq2 (same operations, same order) + per-input-block sums of squares of the trellis inputs t = X + F and of
    X: returns (Q, st [nblk], sx [nblk]) with blocks in processing order (last input block first)"""
    R, Kd = X.shape; F = torch.zeros_like(X); Q = torch.zeros_like(X); st, sx = [], []
    for a in range(Kd - 16, -1, -16):
        s_t = 0.0
        for c in range(R - 16, -1, -16):
            t = (X[c:c + 16, a:a + 16] + F[c:c + 16, a:a + 16]).reshape(1, 256).numpy()
            s_t += float((t.astype(np.float64) ** 2).sum())
            q = torch.from_numpy(np.asarray(qfun(t), dtype=np.float64)).reshape(16, 16)
            Q[c:c + 16, a:a + 16] = q
            if c > 0: F[:c, a:a + 16] += Lout[c:c + 16, :c].T @ (X[c:c + 16, a:a + 16] - q)
        st.append(s_t); sx.append(float((X[:, a:a + 16] ** 2).sum()))
        if a > 0:
            G = Lout.T @ (X[:, a:a + 16] - Q[:, a:a + 16])
            F[:, :a] += G @ Lin[a:a + 16, :a]
    return Q, st, sx


def drift_stats(st, sx):
    """d_all = RMS of everything the trellis saw / RMS of the pre-feedback rows (what the scale search's x drift
    emulates); d_fork = the fork comment's definition: mean over the recursion of the block RMS / the first block's"""
    rms = [math.sqrt(a / b) * math.sqrt(b / sx[0]) for a, b in zip(st, sx)]   # block RMS(t) relative to RMS(X first)
    return {"d_all": math.sqrt(sum(st) / sum(sx)), "d_fork": sum(rms) / len(rms) / rms[0], "d_last_q": sum(rms[-len(rms) // 4:]) / (len(rms) // 4) / rms[0]}


def search(T, cb, K, drift, coarse):
    """settle.search_scale with an explicit drift and coarse grid (identical arithmetic: gate G1b)"""
    CBS = settle.CBS
    T = (T * drift).astype(np.float32)
    def mse(tiles, ss):
        q = trellis.vit(np.concatenate([tiles * (CBS * s) for s in ss]), cb, K).reshape(len(ss), *tiles.shape)
        return [float(((q[i] / (CBS * s) - tiles) ** 2).mean()) for i, s in enumerate(ss)]
    ci = int(np.argmin(mse(T[::3], coarse))); c = coarse[ci]
    fine = [c + 0.075 * (i - 2) for i in range(5)]
    y = mse(T, fine); b = int(np.argmin(y)); off = 0.0
    if 0 < b < 4:
        den = y[b - 1] - 2 * y[b] + y[b + 1]
        off = max(-0.5, min(0.5, 0.5 * (y[b - 1] - y[b + 1]) / den)) if den > 0 else 0.0
    g = max(fine[b] + off * 0.075, 0.01)
    edge = "top" if (ci == len(coarse) - 1 and b == 4) else ("bottom" if (ci == 0 and b == 0) else "")
    return CBS * g, {"g_scale": g, "coarse_best": round(c, 3), "coarse_idx": ci, "fine_idx": b, "edge": edge, "fine_mse": y}


def hvec(tensor, layer, p, s):
    key = S87.hout_key(layer, tensor, p)
    with safe_open(os.path.join(os.environ[HDIRS[s]], key + ".safetensors"), "pt") as f:
        h = f.get_tensor("hout_diag").double()
    assert h.numel() == p.N and bool(torch.isfinite(h).all()) and float(h.min()) > 0, (key, s)
    return h


def lout_of(p, h, r, pexp, sigma_out):
    """S87.hout_metric's L_out with h -> h^pexp (the whole-tensor mean of h^p r^2 as prepare_H_out)"""
    idx = (torch.tensor(p.extra["blocks"])[:, None] * 128 + torch.arange(128)).reshape(-1)
    hp = h ** pexp
    d = hp[idx] * r[idx] ** 2; m = float((hp * r ** 2).mean())
    hd = settle.had128()
    Ho = (hd * d[None, :]) @ hd / m
    Ho.diagonal().add_(sigma_out)
    return S87.block_ldl(Ho)


def rowerr(p, Q, sig, Hs):
    """per output row, after the per-output-channel refit under H_A + δI (S87.scores' arithmetic) and before it:
    {X: d_n H_X d_n^T}; plus {X: w_n H_X w_n^T}"""
    HA = Hs["A"]
    q = p.back(Q); dl = sig * HA.diagonal().mean(); hq = q @ HA
    c = ((hq * p.Wo).sum(1) + dl * (q * p.Wo).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
    out = {}
    for tag, D in (("e", p.Wo - q * c[:, None]), ("e0", p.Wo - q)):
        for k, H in Hs.items():
            Dh = D.to(H.dtype); out.setdefault(tag, {})[k] = ((Dh @ H) * Dh).sum(1).double()
    return out


def refrows(p, Hs):
    return {k: ((p.Wo.to(H.dtype) @ H) * p.Wo.to(H.dtype)).sum(1).double() for k, H in Hs.items()}


def c9stats(en, al):
    """post.c9_rows' statistics from e_n (f64): clip [0, 0.5], c = (1 - e)^(-α/2)"""
    ncl = int((en > E_CLIP).sum()); e = en.clamp(min=0.0, max=E_CLIP); c = (1 - e).pow(-al / 2)
    q = torch.quantile(c.float(), torch.tensor([0.01, 0.5, 0.99])).tolist()
    return {"alpha": al, "e_mean": float(e.mean()), "e_raw_mean": float(en.mean()), "e_max": float(en.max()), "clipped": ncl,
            "c_mean": float(c.mean()), "c_min": float(c.min()), "c_max": float(c.max()), "c_q01_q50_q99": q,
            "n_gt13": int((c > 1.3).sum()), "n_lt077": int((c < 1 / 1.3).sum()), "rows": int(c.numel())}


def setup(layer, tensor, sample):
    p = S87.prep_nohess(layer, tensor, sample)
    hn = e3.ROUTE[tensor][1]
    HA = settle77.load_h(os.environ["W77_HA"], layer, hn); p.H = HA
    Hs = {"A": HA, "B1": settle77.load_h(os.environ["W77_HB1"], layer, hn).float(),
          "B2": settle77.load_h(os.environ["W77_HB2"], layer, hn).float()}
    p.Wo = p.Wo.double()
    key = S87.hout_key(layer, tensor, p); r = S87.row_rms(layer, key)
    idx = (torch.tensor(p.extra["blocks"])[:, None] * 128 + torch.arange(128)).reshape(-1)
    assert float((r[idx] / p.r[:, 0].double() - 1).abs().max()) < 1e-5
    return p, Hs, r, idx


def scale_of(layer, tensor, sample, Ks):
    return S87.get_scale(layer, tensor, sample, CB, Ks)


def lin_for(p, tensor, sig, cache):
    """L_in for σ (block order on readers), one held at a time (memory)"""
    if cache.get("sig") != sig:
        cache.clear()
        rd = S93.is_reader(tensor)
        idx, inv = S93.perm(S93.ORDER) if rd else (None, None)
        cache.update(sig=sig, L=S93.lin_of(p, sig, idx), idx=idx, inv=inv)
    return cache


def quant(p, lc, Lout, s, K):
    c = np.ascontiguousarray(trellis2.codebook(settle77.CBN[CB]), dtype=np.float32)
    X = p.Wt * s
    if lc["idx"] is None:
        Q, st, sx = ldlq2s(X, lc["L"], Lout, S87.vit_fun(c, K))
    else:
        Q, st, sx = ldlq2s(X[:, lc["idx"]].contiguous(), lc["L"], Lout, S87.vit_fun(c, K)); Q = Q[:, lc["inv"]]
    return Q / s, drift_stats(st, sx)


def write(fn, lines):
    with open(fn + ".part", "w") as fh:
        for o in lines: fh.write(json.dumps(o) + "\n")
    os.replace(fn + ".part", fn)


def base(layer, tensor, sample, K, p, exp):
    return {"exp": exp, "layer": layer, "w": tensor, "group": "input" if tensor in S87.INPUT else "rest", "sample": sample,
            "cb": CB, "bits": K, "reader": S93.is_reader(tensor), "N": p.N, "Kd": p.Kd, **{k: v for k, v in p.extra.items()}}


def sub1(layer, tensor, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    fn = lambda Ks: os.path.join(outdir, f"{layer}-{tensor}-s{sample}-sub1-K{Ks}.jsonl")
    todo = [Ks for Ks in SUB1_K if not os.path.exists(fn(Ks))]
    if todo:
        t00 = time.time()
        p, Hs, r, idx = setup(layer, tensor, sample)
        hF = hvec(tensor, layer, p, "F"); hw = hF[idx]
        Lout = lout_of(p, hF, r, 1.0, S87.SIGMA_OUT)
        ref = refrows(p, Hs); lc = {}
        cbv = np.ascontiguousarray(trellis2.codebook(settle77.CBN[CB]), dtype=np.float32)
        for Ks in todo:
            K = float(Fraction(Ks)); sig = S87.sigma_current(tensor, K); lin_for(p, tensor, sig, lc)
            sc = scale_of(layer, tensor, sample, Ks); s0 = sc["s_exl"]
            d0 = settle77.drift(K)
            s_chk, rec0 = search(p.T, cbv, K, d0, COARSE)
            chk = s_chk / s0 - 1                           # the stored scale (another box's search: BLAS / FMA) is used
            arms = [("cur", s0, rec0, d0)]
            s_ext, rec_ext = search(p.T, cbv, K, d0, COARSE_EXT)
            lines = []
            def one(arm, s, rec, dr):
                t0 = time.time()
                Q, ds = quant(p, lc, Lout, s, K)
                e = rowerr(p, Q, sig, Hs); del Q
                o = base(layer, tensor, sample, K, p, "w100-sub1")
                o.update(arm=arm, search_check=chk, sigma=sig, sigma_out=S87.SIGMA_OUT, s=s, search=rec, drift_used=dr, measured=ds)
                for tag in ("e", "e0"):
                    rel2 = {k: float((hw * e[tag][k]).sum() / (hw * ref[k]).sum()) for k in Hs}
                    rel1 = {k: float(e[tag][k].sum() / ref[k].sum()) for k in Hs}
                    for rr, w in ((rel2, hw), (rel1, None)):
                        num = sum(float(((w if w is not None else 1) * e[tag][k]).sum()) for k in ("B1", "B2"))
                        den = sum(float(((w if w is not None else 1) * ref[k]).sum()) for k in ("B1", "B2"))
                        rr["B"] = num / den
                        num = sum(float(((w if w is not None else 1) * e[tag][k]).sum()) for k in Hs)
                        den = sum(float(((w if w is not None else 1) * ref[k]).sum()) for k in Hs)
                        rr["pub"] = num / den
                    o["rel2" if tag == "e" else "rel2_norefit"] = rel2; o["rel1" if tag == "e" else "rel1_norefit"] = rel1
                en = sum(e["e"][k] for k in Hs) / sum(ref[k] for k in Hs).clamp(min=1e-300)   # C9's e_n, M = H_pub
                o["c9_e"] = {"e_mean": float(en.mean()), "e_q50": float(en.median()), "e_max": float(en.max()), "n_gt05": int((en > E_CLIP).sum()), "n_gt1": int((en > 1).sum())}
                if tensor in ALPHA:
                    o["c9"] = c9stats(en, ALPHA[tensor])
                    o["c9_alt"] = {str(a): c9stats(en, a) for a in (3.0, -1.0) if a != ALPHA[tensor]}
                o["sec"] = round(time.time() - t0, 1)
                print(json.dumps({k: v for k, v in o.items() if k not in ("blocks", "search")}), flush=True)
                lines.append(o); return ds
            ds = one("cur", s0, rec0, d0)
            if abs(s_ext / s0 - 1) > 1e-12: one("ext", s_ext, rec_ext, d0)
            else: lines[-1]["ext_same"] = True
            dm = ds["d_all"]
            s_dr, rec_dr = search(p.T, cbv, K, dm, COARSE)
            if abs(s_dr / s0 - 1) > 1e-12: one("drift", s_dr, rec_dr, dm)
            else: lines[0]["drift_same"] = True
            for o in lines: o["sec_job"] = round(time.time() - t00, 1); o["ext_search"] = rec_ext; o["drift_search"] = rec_dr
            write(fn(Ks), lines)
    if all(os.path.exists(fn(Ks)) for Ks in SUB1_K):
        open(os.path.join(outdir, f"{layer}-{tensor}-s{sample}-sub1.DONE"), "w").close()


G125 = [float(x) for x in os.environ.get("W100_G125", "2.05,2.6,3.3").split(",")]


def g125(layer, tensor, sample, outdir):
    """amendment 1 (report only): K0.125 at today's settings (as sub1's cur arm) quantized at fixed g of G125 (exllamav3
    units, s = CBS g); OUTDIR/LAYER-TENSOR-sS-g125.jsonl, one line per g (two-sided / one-sided rel, as sub1)"""
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{layer}-{tensor}-s{sample}-g125.jsonl")
    if not os.path.exists(fn):
        t00 = time.time(); K = 0.125
        p, Hs, r, idx = setup(layer, tensor, sample)
        hF = hvec(tensor, layer, p, "F"); hw = hF[idx]
        Lout = lout_of(p, hF, r, 1.0, S87.SIGMA_OUT)
        ref = refrows(p, Hs); sig = S87.sigma_current(tensor, K); lc = lin_for(p, tensor, sig, {})
        lines = []
        for g in G125:
            t0 = time.time(); s = settle.CBS * g
            Q, ds = quant(p, lc, Lout, s, K)
            e = rowerr(p, Q, sig, Hs); del Q
            o = base(layer, tensor, sample, K, p, "w100-g125")
            o.update(arm=f"g{g:g}", g=g, s=s, sigma=sig, sigma_out=S87.SIGMA_OUT, measured=ds)
            w2 = lambda k, v: float((hw * v).sum())
            o["rel2"] = {k: w2(k, e["e"][k]) / w2(k, ref[k]) for k in Hs}
            o["rel1"] = {k: float(e["e"][k].sum() / ref[k].sum()) for k in Hs}
            o["rel2"]["B"] = sum(w2(k, e["e"][k]) for k in ("B1", "B2")) / sum(w2(k, ref[k]) for k in ("B1", "B2"))
            o["rel1"]["B"] = sum(float(e["e"][k].sum()) for k in ("B1", "B2")) / sum(float(ref[k].sum()) for k in ("B1", "B2"))
            o["rel2"]["pub"] = sum(w2(k, e["e"][k]) for k in Hs) / sum(w2(k, ref[k]) for k in Hs)
            o["rel1"]["pub"] = sum(float(e["e"][k].sum()) for k in Hs) / sum(float(ref[k].sum()) for k in Hs)
            o["sec"] = round(time.time() - t0, 1)
            print(json.dumps({k: v for k, v in o.items() if k != "blocks"}), flush=True)
            lines.append(o)
        for o in lines: o["sec_job"] = round(time.time() - t00, 1)
        write(fn, lines)
    open(os.path.join(outdir, f"{layer}-{tensor}-s{sample}-g125.DONE"), "w").close()


def hshape(layer, tensor, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    fn = lambda Ks: os.path.join(outdir, f"{layer}-{tensor}-s{sample}-hshape-K{Ks}.jsonl")
    todo = [Ks for Ks in HSHAPE_K if not os.path.exists(fn(Ks))]
    if todo:
        t00 = time.time()
        p, Hs, r, idx = setup(layer, tensor, sample)
        hv = {s: hvec(tensor, layer, p, s) for s in HDIRS}
        hb = {s: v[idx] for s, v in hv.items()}
        ref = refrows(p, Hs)
        refs = {f"{k}|{s}": float((hb[s] * ref[k]).sum()) for k in Hs for s in hb}
        refs.update({f"{k}|1": float(ref[k].sum()) for k in Hs})
        lc = {}
        arms = [("full", 0.01, 1.0)] + [("E", so, px) for so in SO for px in PX]
        for Ks in todo:
            K = float(Fraction(Ks)); sig = S87.sigma_current(tensor, K); lin_for(p, tensor, sig, lc)
            s0 = scale_of(layer, tensor, sample, Ks)["s_exl"]
            lines = []
            for hs, so, px in arms:
                t0 = time.time()
                Lout = lout_of(p, hv["F" if hs == "full" else "E"], r, px, so)
                Q, ds = quant(p, lc, Lout, s0, K); del Lout
                e = rowerr(p, Q, sig, Hs); del Q
                err = {f"{k}|{s}": float((hb[s] * e["e"][k]).sum()) for k in Hs for s in hb}
                err.update({f"{k}|1": float(e["e"][k].sum()) for k in Hs})
                o = base(layer, tensor, sample, K, p, "w100-hshape")
                o.update(arm=f"{hs}-so{so:g}-p{px:g}", hq=hs, sigma_out=so, p=px, sigma=sig, s=s0, err=err, ref=refs,
                         rel={kk: err[kk] / refs[kk] for kk in err}, measured=ds, sec=round(time.time() - t0, 1))
                print(json.dumps({k: v for k, v in o.items() if k not in ("blocks", "err", "ref")}), flush=True)
                lines.append(o)
            for o in lines: o["sec_job"] = round(time.time() - t00, 1)
            write(fn(Ks), lines)
    if all(os.path.exists(fn(Ks)) for Ks in HSHAPE_K):
        open(os.path.join(outdir, f"{layer}-{tensor}-s{sample}-hshape.DONE"), "w").close()


def selftest(out):
    """G1a ldlq2s == S87.ldlq2 (Q bit for bit), 11 attn_k s0 i8 K0.5, production metrics; G1b search() with the default
    grid and drift == the stored scale of that job; G1c lout_of(h_F, p 1, σ_out 0.01) == S87.hout_metric's L_out;
    G1d the hshape "full" arm at K2.5 of 11 attn_k s0 == w93's order / m 1 row (rel B1 / B2 / A, two-sided, bn1920)
    to 1e-6 relative (W100_W93ROW: that row's JSON file)"""
    o = {}
    layer, tensor, sample = 11, "attn_k", 0
    p, Hs, r, idx = setup(layer, tensor, sample)
    hF = hvec(tensor, layer, p, "F")
    Lout = lout_of(p, hF, r, 1.0, 0.01)
    Lref, hw, _ = S87.hout_metric(layer, tensor, p, 0.01)
    o["G1c_max_abs"] = float((Lout - Lref).abs().max()); o["G1c"] = o["G1c_max_abs"] == 0.0
    cbv = np.ascontiguousarray(trellis2.codebook(settle77.CBN[CB]), dtype=np.float32)
    sc = scale_of(layer, tensor, sample, "0.5")
    s_chk, _ = search(p.T, cbv, 0.5, settle77.drift(0.5), COARSE)
    o["G1b_ratio"] = s_chk / sc["s_exl"]; o["G1b"] = abs(s_chk / sc["s_exl"] - 1) < 1e-12
    lc = lin_for(p, tensor, S87.sigma_current(tensor, 0.5), {})
    X = (p.Wt * sc["s_exl"])[:, lc["idx"]].contiguous()
    Q1 = S87.ldlq2(X, lc["L"], Lout, S87.vit_fun(cbv, 0.5)); Q2, st, sx = ldlq2s(X, lc["L"], Lout, S87.vit_fun(cbv, 0.5))
    o["G1a_max_abs"] = float((Q1 - Q2).abs().max()); o["G1a"] = o["G1a_max_abs"] == 0.0; o["G1a_drift"] = drift_stats(st, sx)
    del Q1, Q2, X
    w93 = json.load(open(os.environ["W100_W93ROW"]))
    lc = lin_for(p, tensor, S87.sigma_current(tensor, 2.5), {})
    s25 = scale_of(layer, tensor, sample, "2.5")["s_exl"]
    assert abs(s25 / w93["s_exl"] - 1) < 1e-12
    Q, _ = quant(p, lc, Lout, s25, 2.5)
    e = rowerr(p, Q, S87.sigma_current(tensor, 2.5), Hs); ref = refrows(p, Hs)
    mine = {k: float((hw * e["e"][k]).sum() / (hw * ref[k]).sum()) for k in Hs}
    o["G1d_mine"] = mine; o["G1d_w93"] = {k: w93["rel"][k] for k in Hs}
    o["G1d_max_rel"] = max(abs(mine[k] / w93["rel"][k] - 1) for k in Hs); o["G1d"] = o["G1d_max_rel"] <= 1e-6
    o["PASS"] = all(o[g] for g in ("G1a", "G1b", "G1c", "G1d"))
    print(json.dumps(o), flush=True)
    with open(out + ".part", "w") as f: json.dump(o, f)
    os.replace(out + ".part", out)


def main():
    m = sys.argv[1]
    if m == "sub1": return sub1(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if m == "hshape": return hshape(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if m == "g125": return g125(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if m == "selftest": return selftest(sys.argv[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
