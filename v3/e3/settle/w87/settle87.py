#!/usr/bin/env python3
"""settle87.py -- w87 (#114-03, ../w87.md): w77's held-out damping sweep with H_out on (two-sided LDLQ).

  settle87.py job LAYER TENSOR SAMPLE OUTDIR [DONE]   every sigma of W87_SIG x every level of W87_LEVELS; one file per
                                                  sigma, OUTDIR/LAYER-TENSOR-s{SAMPLE}-d{SIGMA}.jsonl (skipped if
                                                  present: restartable), then OUTDIR/LAYER-TENSOR-s{SAMPLE}.DONE
                                                  (DONE: that path again, ignored; lets q87.py adopt a running job)
  settle87.py selftest LAYER TENSOR SAMPLE CB K SIGMA OUT.json    gate G2: H_out = I, sigma_out 0 against w77's
                                                  one-sided ldlq_cached (same factorised metric, same trellis)

w77's settle77.py is imported unchanged (P2 transforms of one aligned 128-row output block, mul1x_i4 / mul1x_i8, ourq's
drift in the scale search, H_A for the quantizer, H_B1 / H_B2 held out). Added here, as the fork's (eb5a2139)
prepare_H_out + ldlq_2hess:
  - H_out = diag(h), h = bn1920's hout_diag of the tensor (W87_HOUT/<key>.safetensors). In the proxy's rotated basis
    (rows = P S_v W, S_v = output signs x output-channel RMS) it is P diag(h r^2) P on the block (P = 128-Hadamard,
    block-diagonal, so the 128-row block is closed), divided by the mean of h r^2 over ALL N rows of the tensor
    (= prepare_H_out's mean diagonal), + sigma_out I (W87_SIGMA_OUT, default 0.01), block-LDL(16).
  - two-sided LDLQ (ldlq2): input blocks last to first, output tiles last to first inside a block; the input of tile
    (c, a) is X + F with F += L_out'[c, :]^T dE L_in'[a, :] over the quantized tiles (L' unit block-lower,
    H = L' D L'^T, exllamav3's block_ldl). Same result as ldlq_2hess's anti-diagonals (gate87.py G1).
  - error: per-output-channel refit under H_A + delta I (as w77; its optimum does not depend on a diagonal H_out),
    then e_X = sum_i h_i d_i H_X d_i^T, rel_X = e_X / sum_i h_i w_i H_X w_i^T (X = B1, B2, A); "rel1" = the same
    without h (one-sided, reported only).
Memory: H_A fp64, H_B1 / H_B2 fp32, one factor at a time (ffn_down ~10 GB peak; the rest <= 1.5 GB)."""
import json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w77"), os.path.join(_h, ".."), os.path.join(_h, "..", "..")]
import settle77                                            # patches codebook (mul1x_i4), drift, settle64's rev order
import settle, settle64, trellis, trellis2, e3
from safetensors import safe_open
torch.set_grad_enabled(False)

SIG = [float(x) for x in os.environ.get("W87_SIG", "0.001,0.005,0.025,0.05,0.1,0.2").split(",")]
LEVELS = [tuple(x.split(":")) for x in os.environ.get("W87_LEVELS", "i4:0.125,i4:0.5,i4:1,i8:2,i8:2.5,i8:4").split(",")]
SIGMA_OUT = float(os.environ.get("W87_SIGMA_OUT", "0.01"))
INPUT = settle.INPUT


def sigma_current(tensor, K):
    """v3.toml today: K0.125 0.2 both groups (#114-02); below K1 inputs 0.05; from K1 inputs 0.005; rest 0.025"""
    if K <= 0.125: return 0.2
    return (0.05 if K < 1 else 0.005) if tensor in INPUT else 0.025


def block_ldl(H, b=16):
    """exllamav3's block_ldl (fork eb5a2139), fp64: H = L L^T (Cholesky), L <- L blockdiag(L_ii)^-1, identity diagonal
    blocks: H = L D L^T with L unit block-lower. Consumes H (returns L in a new tensor; caller drops H)"""
    n = H.shape[0]; m = n // b
    L = torch.linalg.cholesky(H)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    Lv = L.view(n, m, b)
    for i in range(m): Lv[:, i, :] = Lv[:, i, :] @ DL[i]
    Lb = L.view(m, b, m, b).permute(0, 2, 1, 3); dr = torch.arange(m)
    Lb[dr, dr] = torch.eye(b, dtype=L.dtype).expand(m, b, b)
    return L


def ldlq2(X, Lin, Lout, qfun):
    """two-sided block LDLQ in the proxy's layout: X [R, Kd] (rows = output channels, rotated), Lin [Kd, Kd],
    Lout [R, R] unit block-lower (identity diagonal blocks). qfun(tiles [n, 256] float64) -> [n, 256]. Objective
    tr(Ho E Hi E^T) with Hi = Lin D Lin^T, Ho = Lout D' Lout^T. Returns Q (float64)."""
    R, Kd = X.shape; F = torch.zeros_like(X); Q = torch.zeros_like(X)
    for a in range(Kd - 16, -1, -16):
        for c in range(R - 16, -1, -16):
            t = (X[c:c + 16, a:a + 16] + F[c:c + 16, a:a + 16]).reshape(1, 256).numpy()
            q = torch.from_numpy(np.asarray(qfun(t), dtype=np.float64)).reshape(16, 16)
            Q[c:c + 16, a:a + 16] = q
            if c > 0: F[:c, a:a + 16] += Lout[c:c + 16, :c].T @ (X[c:c + 16, a:a + 16] - q)
        if a > 0:
            G = Lout.T @ (X[:, a:a + 16] - Q[:, a:a + 16])
            F[:, :a] += G @ Lin[a:a + 16, :a]
    return Q


def vit_fun(cb, K):
    return lambda t: trellis.vit(t, cb, K)


def hout_key(layer, tensor, p):
    names = e3.ROUTE[tensor][0]
    name = p.extra.get("part", names[0])
    return f"{e3.P}{layer}.{name}"


def row_rms(layer, key):
    """RMS of every output row of the bf16 weight (exllamav3's out_channel_scales up to a constant)"""
    idx = json.load(open(f"{e3.CK}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{e3.CK}/{idx[key + '.weight']}", "pt") as f:
        W = f.get_tensor(key + ".weight").float()
    return W.pow(2).mean(1).sqrt().double()


def hout_metric(layer, tensor, p, sigma_out, ident=False):
    """(L_out [128, 128], h [128] the block's diagonal weights, info). ident: H_out = I, sigma_out as given (G2)"""
    if ident:
        Ho = torch.eye(128, dtype=torch.float64) * (1.0 + sigma_out)
        return block_ldl(Ho), torch.ones(128, dtype=torch.float64), {"hout": "identity"}
    key = hout_key(layer, tensor, p)
    with safe_open(os.path.join(os.environ["W87_HOUT"], key + ".safetensors"), "pt") as f:
        h = f.get_tensor("hout_diag").double()
    r = row_rms(layer, key)
    assert h.numel() == r.numel() == p.N, (key, h.numel(), r.numel(), p.N)
    assert bool(torch.isfinite(h).all()) and float(h.min()) > 0, key
    idx = (torch.tensor(p.extra["blocks"])[:, None] * 128 + torch.arange(128)).reshape(-1)
    assert idx.numel() == 128
    rr = float((r[idx] / p.r[:, 0].double() - 1).abs().max())
    assert rr < 1e-5, f"{key}: P2's row scales differ from the checkpoint's ({rr})"
    hb = h[idx]; d = hb * r[idx] ** 2; m = float((h * r ** 2).mean())
    hd = settle.had128()
    Ho = (hd * d[None, :]) @ hd / m                  # P diag(h r^2) P, P = P^T = P^-1
    Ho.diagonal().add_(sigma_out)
    info = {"hout_key": key, "hout_mean_all": m, "hout_block_mean": float(d.mean()) / m, "hout_cv_block": float(hb.std() / hb.mean()),
            "sigma_out": sigma_out, "row_rms_check": rr}
    return block_ldl(Ho), hb, info


def in_metric(p, sig):
    """L_in of the rotated damped H_A (P2.metric: P diag(su) (H_A + delta I) diag(su) P), block-LDL, fp64"""
    Hq, dq = p.metric(sig); assert dq == 0
    dead = int((p.H.diagonal() == 0).sum()); assert dead == 0, "dead input columns"
    return block_ldl(Hq)


def scores(p, Q, sig, HA, Hs, hw):
    """refit (per output channel under H_A + delta I, as settle77.rels) then per-row errors under each H in Hs:
    returns ({X: two-sided err}, {X: one-sided err}, same before the refit)"""
    q = p.back(Q); dl = sig * HA.diagonal().mean(); hq = q @ HA
    c = ((hq * p.Wo).sum(1) + dl * (q * p.Wo).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
    out = {}
    for tag, D in (("", p.Wo - q * c[:, None]), ("_norefit", p.Wo - q)):
        for k, H in Hs.items():
            Dh = D.to(H.dtype); e = ((Dh @ H) * Dh).sum(1).double()
            out.setdefault("e2" + tag, {})[k] = float((hw * e).sum()); out.setdefault("e1" + tag, {})[k] = float(e.sum())
    return out


def refs(p, Hs, hw):
    o2, o1 = {}, {}
    for k, H in Hs.items():
        W = p.Wo.to(H.dtype); e = ((W @ H) * W).sum(1).double()
        o2[k] = float((hw * e).sum()); o1[k] = float(e.sum())
    return o2, o1


def get_scale(layer, tensor, sample, cb, Ks):
    path = settle77.scale_path(layer, tensor, sample, cb, Ks)
    if not os.path.exists(path): settle77.do_scale(layer, tensor, sample, cb, Ks)
    return json.load(open(path))


def prep_nohess(layer, tensor, sample):
    """settle77.prep without P2 loading the published H (not used here: memory)"""
    lh = e3.load_h; e3.load_h = lambda *a: None
    try: return settle77.prep(layer, tensor, sample)
    finally: e3.load_h = lh


def job(layer, tensor, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{layer}-{tensor}-s{sample}")
    todo = [s for s in SIG if not os.path.exists(f"{stem}-d{s:g}.jsonl")]
    t00 = time.time()
    p = prep_nohess(layer, tensor, sample)
    hn = e3.ROUTE[tensor][1]
    Lout, hw, hinfo = hout_metric(layer, tensor, p, SIGMA_OUT)
    sc = {(cb, Ks): get_scale(layer, tensor, sample, cb, Ks) for cb, Ks in LEVELS}
    if todo:
        HA = settle77.load_h(os.environ["W77_HA"], layer, hn); p.H = HA
        Hs = {"B1": settle77.load_h(os.environ["W77_HB1"], layer, hn).float(),
              "B2": settle77.load_h(os.environ["W77_HB2"], layer, hn).float(), "A": HA}
        p.Wo = p.Wo.double()
        ref2, ref1 = refs(p, Hs, hw)
    for sig in todo:
        t0 = time.time(); Lin = in_metric(p, sig); sec_metric = round(time.time() - t0, 1)
        lines = []
        for cb, Ks in LEVELS:
            K = float(Fraction(Ks)); s = sc[(cb, Ks)]["s_exl"]
            c = np.ascontiguousarray(trellis2.codebook(settle77.CBN[cb]), dtype=np.float32)
            t0 = time.time()
            Q = ldlq2(p.Wt * s, Lin, Lout, vit_fun(c, K)) / s
            e = scores(p, Q, sig, p.H, Hs, hw); del Q
            rel = {k: e["e2"][k] / ref2[k] for k in Hs}; rel["B"] = (e["e2"]["B1"] + e["e2"]["B2"]) / (ref2["B1"] + ref2["B2"])
            rel1 = {k: e["e1"][k] / ref1[k] for k in Hs}; rel1["B"] = (e["e1"]["B1"] + e["e1"]["B2"]) / (ref1["B1"] + ref1["B2"])
            reln = {k: e["e2_norefit"][k] / ref2[k] for k in Hs}
            lines.append({"exp": "w87", "layer": layer, "w": tensor, "group": "input" if tensor in INPUT else "rest",
                          "sample": sample, "cb": cb, "bits": K, "sigma": sig, "sigma_current": sigma_current(tensor, K),
                          "s_exl": s, "drift": sc[(cb, Ks)]["drift"], "rel": rel, "rel1": rel1, "rel_norefit": reln,
                          "err2": e["e2"], "ref2": ref2, "sec": round(time.time() - t0, 1), "sec_metric": sec_metric,
                          "order": "rev2", "N": p.N, "Kd": p.Kd, **hinfo, **{k: v for k, v in p.extra.items()}})
            print(json.dumps({k: v for k, v in lines[-1].items() if k not in ("blocks", "err2", "ref2")}), flush=True)
        del Lin
        for o in lines: o["sec_job"] = round(time.time() - t00, 1)
        f = f"{stem}-d{sig:g}.jsonl"
        with open(f + ".part", "w") as fh:
            for o in lines: fh.write(json.dumps(o) + "\n")
        os.replace(f + ".part", f)
    if all(os.path.exists(f"{stem}-d{s:g}.jsonl") for s in SIG):
        open(stem + ".DONE", "w").close()


def selftest(layer, tensor, sample, cb, Ks, sig, out):
    """G2: H_out = I, sigma_out = 0 -> one-sided LDLQ; against settle77.ldlq_cached under the same metric"""
    p = prep_nohess(layer, tensor, sample); hn = e3.ROUTE[tensor][1]
    HA = settle77.load_h(os.environ["W77_HA"], layer, hn); p.H = HA; p.Wo = p.Wo.double()
    K = float(Fraction(Ks)); s = get_scale(layer, tensor, sample, cb, Ks)["s_exl"]
    c = np.ascontiguousarray(trellis2.codebook(settle77.CBN[cb]), dtype=np.float32)
    Hq, _ = p.metric(sig)
    t0 = time.time(); Q1 = settle77.ldlq_cached(p.Wt * s, settle77.ldlq_prep(Hq), c, K) / s; t1 = time.time()
    Lout, hw, _ = hout_metric(layer, tensor, p, 0.0, ident=True)
    Q2 = ldlq2(p.Wt * s, block_ldl(Hq), Lout, vit_fun(c, K)) / s; t2 = time.time()
    Hs = {"B1": settle77.load_h(os.environ["W77_HB1"], layer, hn).float(), "B2": settle77.load_h(os.environ["W77_HB2"], layer, hn).float()}
    ref2, _ = refs(p, Hs, hw)
    e1 = scores(p, Q1, sig, HA, Hs, hw)["e2"]; e2 = scores(p, Q2, sig, HA, Hs, hw)["e2"]
    rB1 = (e1["B1"] + e1["B2"]) / (ref2["B1"] + ref2["B2"]); rB2 = (e2["B1"] + e2["B2"]) / (ref2["B1"] + ref2["B2"])
    tiles_diff = int(((Q1 - Q2).abs().reshape(8, 16, -1, 16).amax(dim=(1, 3)) > 1e-9).sum())
    o = {"selftest": f"{layer}-{tensor}-s{sample}-{cb}-K{Ks}-d{sig}", "rel_B_cached": rB1, "rel_B_ldlq2": rB2,
         "rel_ratio": rB2 / rB1, "max_abs_dQ": float((Q1 - Q2).abs().max()), "tiles_differ": tiles_diff,
         "tiles": Q1.numel() // 256, "sec_cached": round(t1 - t0, 1), "sec_ldlq2": round(t2 - t1, 1),
         "PASS": abs(rB2 / rB1 - 1) <= 1e-5}
    print(json.dumps(o), flush=True)
    with open(out + ".part", "w") as f: json.dump(o, f)
    os.replace(out + ".part", out)


def main():
    if sys.argv[1] == "job":
        return job(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if sys.argv[1] == "selftest":
        return selftest(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], float(sys.argv[7]), sys.argv[8])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
