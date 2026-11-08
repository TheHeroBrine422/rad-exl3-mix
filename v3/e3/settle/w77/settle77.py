#!/usr/bin/env python3
"""settle77.py LAYER TENSOR SAMPLE SIGMA OUT.jsonl -- w77 (#114, ../w77.md): damping sweep, held out, on w64's
order-fixed proxy. settle64.py is imported unchanged (exllamav3's LDLQ block order, SETTLE_ORDER=rev; the fork's exact
mul1 / mul1_i8 values); this file adds:
  - codebook mul1x_i4: codebook.cuh EXL3_MUL1_I4 on the exact mul1 value v (fp16): fp16((clamp(rint(f32(v) * 7.5f/2.6f
    - 0.5f), -8, 7) + 0.5f) * 2.6f/7.5f) -- 16 levels (gate77.py checks it against exl3i_ref.c + w53's i4 patch);
    sub-1 K through trellis.py's pattern path (0-bit steps read cb[state ^ salt(i)], as qtf2 / viterbi.c);
  - ourq's LDLQ drift (log-linear between exllamav3's table points, 1.08 below K1, 1.0 from K4) in the scale search;
  - held-out Hessians: the quantizer (LDLQ metric, damping delta = SIGMA x mean diag, per-output-channel refit) sees
    H_A = H_pub - H_B1 - H_B2 (W77_HA); the error rel = tr(dW H dW^T) / tr(W H W^T) is reported under H_B1, H_B2
    (W77_HB1 / W77_HB2: documents the quantizer never saw), H_A (in-sample) and H_pub.
One job = (tensor, sample, SIGMA), every level of W77_LEVELS (default i4 K0.125 / 0.25 / 0.5 / 0.75 / 1 / 2, i8 K1.5 / 2 / 2.5)
in turn under the same damped metric. The global scale per (tensor, sample, level) is exllamav3's search (settle.py),
run once by `settle77.py scale LAYER TENSOR SAMPLE CB K` into W77_SCALES and read by every SIGMA job.
One JSON line per level; OUT written whole at the end (OUT.part, then renamed)."""
import json, math, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w64"), os.path.join(_h, ".."), os.path.join(_h, "..", "..")]
import settle64                                            # patches trellis.ldlq_trellis (rev order), trellis2.codebook
import settle, trellis, trellis2, e3
from safetensors import safe_open
torch.set_grad_enabled(False)

LEVELS = [tuple(x.split(":")) for x in os.environ.get(
    "W77_LEVELS", "i4:0.125,i4:0.25,i4:0.5,i4:0.75,i4:1,i4:2,i8:1.5,i8:2,i8:2.5").split(",")]
CBN = {"i4": "mul1x_i4", "i8": "mul1x_i8"}
INPUT = settle.INPUT                                       # ssm_qkv, ssm_z, attn_qg, attn_k, attn_v


def sigma_current(tensor, K):
    """the ladder's damping today: layer-input group 0.05 below K1 (user 11-07), 0.005 from K1; the rest 0.025"""
    return (0.05 if K < 1 else 0.005) if tensor in INPUT else 0.025


_cb_prev = trellis2.codebook
def codebook(name):
    if name != "mul1x_i4": return _cb_prev(name)
    v = settle64.codebook("mul1x").astype(np.float16).astype(np.float32)
    st, inv = np.float32(2.6) / np.float32(7.5), np.float32(7.5) / np.float32(2.6)
    q = np.clip(np.rint(v * inv - np.float32(0.5)), -8, 7).astype(np.float32)
    return ((q + np.float32(0.5)) * st).astype(np.float16).astype(np.float64)
trellis2.codebook = codebook


_DT = sorted({1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004, 4: 1.0}.items())
def drift(K):
    """ourq.py's _drift (OURQ_DRIFT_EXACT unset)"""
    K = float(K); d = dict(_DT)
    if K in d: return d[K]
    if K <= _DT[0][0]: return _DT[0][1]
    if K >= _DT[-1][0]: return 1.0
    (k0, d0), (k1, d1) = next((p, q) for p, q in zip(_DT, _DT[1:]) if p[0] < K < q[0])
    return math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * (K - k0) / (k1 - k0))
class _Drift(dict):
    def get(self, K, default=None): return drift(K)
settle.DRIFT = _Drift()                                    # settle.search_scale reads DRIFT.get(K, 1.0)


def load_h(d, layer, hn):
    with safe_open(f"{d}/{e3.P}{layer}.{hn}.safetensors", "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k)
    H[iu[0], iu[1]] = up; H = H + H.T - torch.diag(H.diagonal())
    return H


def prep(layer, tensor, sample):
    p = settle.P2(layer, tensor, 128, sample, sample)      # transformed rows / tile sample (H-independent), cached
    return p


def scale_path(layer, tensor, sample, cb, K):
    return os.path.join(os.environ.get("W77_SCALES", "scales"), f"{layer}-{tensor}-s{sample}-{cb}-K{K}.json")


def do_scale(layer, tensor, sample, cb, K):
    out = scale_path(layer, tensor, sample, cb, K)
    if os.path.exists(out): return
    p = prep(layer, tensor, sample)
    c = np.ascontiguousarray(trellis2.codebook(CBN[cb]), dtype=np.float32)
    t0 = time.time(); s, sx = settle.search_scale(p.T, c, float(Fraction(K)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w") as f:
        json.dump({"s_exl": s, "search": sx, "drift": drift(float(Fraction(K))), "sec": round(time.time() - t0, 1)}, f)
    os.replace(out + ".part", out)


def quad(D, H):
    D = D.to(H.dtype)
    return float(torch.einsum("ij,jk,ik->", D, H, D))


def rels(p, Q, d, HA, Hs, ref):
    """per-output-channel refit under H_A + delta I (the quantizer's), then the error tr(dW H dW^T) under each
    scoring H (ref[k] = tr(W H W^T): rel = err / ref); Q in Wt units. Returns (err after refit, err before)"""
    q = p.back(Q); dl = d * HA.diagonal().mean(); hq = q @ HA
    c = ((hq * p.Wo).sum(1) + dl * (q * p.Wo).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
    D1, D0 = p.Wo - q * c[:, None], p.Wo - q
    return {k: quad(D1, H) for k, H in Hs.items()}, {k: quad(D0, H) for k, H in Hs.items()}


def ldlq_prep(Hq):
    """trellis.ldlq_trellis's factorisation, once per job (same operations, block order as settle64 ORDER): returns
    (column permutation, dead columns in that order, U = chol(H^-1) upper). Damping is already in Hq (dq = 0)."""
    perm = settle64.blockrev(Hq.shape[0]) if settle64.ORDER == "rev" else torch.arange(Hq.shape[0])
    H = Hq[perm][:, perm].contiguous(); dead = H.diagonal() == 0
    H[dead, dead] = 1
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True); del H
    Ui = [torch.linalg.inv(U[a:a + 16, a:a + 16]) for a in range(0, U.shape[0], 16)]
    return perm, dead, U, Ui


def ldlq_cached(W, F, cb, K):
    """trellis.ldlq_trellis(W, H, cb, K, 0) in settle64's order with the factorisation F = ldlq_prep(H)"""
    perm, dead, U, Ui = F
    W = W[:, perm].contiguous(); R, Kd = W.shape; W[:, dead] = 0
    Q = torch.zeros_like(W)
    for a in range(0, Kd, 16):
        blk = W[:, a:a + 16]
        tiles = blk.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(trellis.vit(tiles, cb, K).astype(np.float64)).reshape(R, 16)
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ Ui[a // 16]
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    out = torch.empty_like(Q); out[:, perm] = Q
    return out


def selftest(layer, tensor, sample, sig, cb, Ks):
    """with the published H as H_A: ldlq_cached == settle64's trellis.ldlq_trellis bit for bit, and the fp32 scoring
    + refit here == settle.rels (settle64's own refit and rel, fp64) to ~1e-6"""
    p = prep(layer, tensor, sample); K = float(Fraction(Ks))
    c = np.ascontiguousarray(trellis2.codebook(CBN[cb]), dtype=np.float32)
    s, _ = settle.search_scale(p.T, c, K)
    Hq, dq = p.metric(sig)
    t0 = time.time(); Q1 = trellis.ldlq_trellis(p.Wt * s, Hq, c, K, dq) / s; t1 = time.time()
    Q2 = ldlq_cached(p.Wt * s, ldlq_prep(Hq), c, K) / s; t2 = time.time()
    ref = {"pub": quad(p.Wo, p.H.float())}
    e1, e0 = rels(p, Q2, sig, p.H, {"pub": p.H.float()}, ref)
    r_settle = settle.rels(p, Q1, sig)[0]
    print(json.dumps({"selftest": f"{layer}-{tensor}-s{sample}-{cb}-K{Ks}-d{sig}", "max_abs_diff_Q": float((Q1 - Q2).abs().max()),
                      "rel_settle": r_settle[0], "rel_here": e1["pub"] / ref["pub"], "ratio": e1["pub"] / ref["pub"] / r_settle[0],
                      "rel_norefit_settle": r_settle[1], "rel_norefit_here": e0["pub"] / ref["pub"],
                      "sec_ldlq_settle": round(t1 - t0, 1), "sec_ldlq_cached": round(t2 - t1, 1), "s_exl": s}), flush=True)


def main():
    if sys.argv[1] == "selftest":
        return selftest(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7])
    if sys.argv[1] == "scale":
        layer, tensor, sample, cb, K = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
        return do_scale(layer, tensor, sample, cb, K)
    layer, tensor, sample, sig, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    t00 = time.time()
    p = prep(layer, tensor, sample)
    hn = e3.ROUTE[tensor][1]
    Hpub = p.H                                             # P2 loaded the published H (E3_HESS)
    HA = load_h(os.environ["W77_HA"], layer, hn)
    p.H = HA                                               # P2.metric / delta use p.H: the quantizer sees H_A
    t0 = time.time(); Hq, dq = p.metric(sig); assert dq == 0
    F = ldlq_prep(Hq); del Hq; sec_metric = round(time.time() - t0, 1)
    # scoring Hessians in fp32 (memory: ffn_down's are 17408^2); the refit uses H_A in fp64 as the quantizer would
    Hs = {"B1": load_h(os.environ["W77_HB1"], layer, hn).float(), "B2": load_h(os.environ["W77_HB2"], layer, hn).float(),
          "A": HA.float(), "pub": Hpub.float()}
    del Hpub; p.Wo = p.Wo.double()
    Wf = p.Wo.float(); ref = {k: quad(Wf, H) for k, H in Hs.items()}; del Wf
    lines = []
    for cb, Ks in LEVELS:
        K = float(Fraction(Ks)); sc = json.load(open(scale_path(layer, tensor, sample, cb, Ks)))
        c = np.ascontiguousarray(trellis2.codebook(CBN[cb]), dtype=np.float32)
        t0 = time.time(); s = sc["s_exl"]
        Q = ldlq_cached(p.Wt * s, F, c, K) / s
        e1, e0 = rels(p, Q, sig, HA, {k: H for k, H in Hs.items()}, ref); del Q
        r = {k: e1[k] / ref[k] for k in e1}; r0 = {k: e0[k] / ref[k] for k in e0}
        r["B"] = (e1["B1"] + e1["B2"]) / (ref["B1"] + ref["B2"]); r0["B"] = (e0["B1"] + e0["B2"]) / (ref["B1"] + ref["B2"])
        lines.append({"exp": "w77", "layer": layer, "w": tensor, "group": "input" if tensor in INPUT else "rest",
                      "sample": sample, "cb": cb, "bits": K, "sigma": sig, "sigma_current": sigma_current(tensor, K),
                      "s_exl": s, "drift": sc["drift"], "rel": r, "rel_norefit": r0, "err": e1, "ref": ref,
                      "sec": round(time.time() - t0, 1),
                      "sec_metric": sec_metric, "order": settle64.ORDER, "N": p.N, "Kd": p.Kd, **p.extra})
        print(json.dumps({k: v for k, v in lines[-1].items() if k != "blocks"}), flush=True)
    for o in lines: o["sec_job"] = round(time.time() - t00, 1)
    with open(out + ".part", "w") as fh:
        for o in lines: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main()
