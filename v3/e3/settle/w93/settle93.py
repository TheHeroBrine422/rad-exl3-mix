#!/usr/bin/env python3
"""settle93.py -- w93 (#39-01, ../w93.md item 2): the #39 gain re-run on today's stack, held out (CPU).

  settle93.py job LAYER TENSOR SAMPLE OUTDIR [DONE]   every gain of W93_GAINS x every level of W93_LEVELS; one file per
                                                  level, OUTDIR/LAYER-TENSOR-s{SAMPLE}-{cb}-K{K}.jsonl (skipped if
                                                  present: restartable), then OUTDIR/LAYER-TENSOR-s{SAMPLE}.DONE
  settle93.py selftest LAYER TENSOR SAMPLE OUT.json   gate: the identity block order reproduces settle87's natural
                                                  order bit for bit (same Q), at i8 K2.5, m 1

w87's settle87.py (w77's held-out two-sided proxy: H_A quantizes, H_B1 / H_B2 score, H_out = bn1920 diagonal, sigma_out
0.01, exllamav3's last-to-first LDLQ, mul1x_i8, per-output-channel refit) is imported unchanged. Added here:
  - gain m: the trellis input scale is m x s_exl (exllamav3's own g_scale choice, ourq --gain), the codes divided back
    by the same scale (settle87's Q = ldlq2(Wt s) / s with s = m s_exl);
  - #117 block order (v3.toml quant.block_order, = w84/order82.json): for residual readers (H_in = layers.N.input or
    post_attention.input) the LDLQ visits the input 128-blocks in the stored order (route B: Wt and the rotated metric
    permuted by 128-blocks, the proxy's input Hadamard is block-diagonal by 128 so this is exact; codes permuted back).
    Every reader row also gets an "noorder" twin at m = 1 (natural order), reported only;
  - sigma = the current damping (settle87.sigma_current: inputs 0.005, rest 0.025 at K >= 1).
Not emulated: tile CD (attn_qg in production; this proxy's attn_qg rows are pre-CD), the su refit (as w43 / w87)."""
import json, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w87")]
import settle87 as S87                                    # also puts w77, settle, settle64, trellis, e3 on the path
import settle77, trellis2, e3
torch.set_grad_enabled(False)

GAINS = [float(x) for x in os.environ.get("W93_GAINS", "0.9,0.95,1.0,1.05,1.1").split(",")]
LEVELS = [tuple(x.split(":")) for x in os.environ.get("W93_LEVELS", "i8:2.5,i8:4").split(",")]
ORDER = json.load(open(os.environ.get("W93_ORDER", os.path.join(_h, "..", "w84", "order82.json"))))["stored_order"]
assert sorted(ORDER) == list(range(40))


def is_reader(tensor):
    return e3.ROUTE[tensor][1] in ("input", "post_attention.input")


def perm(order):
    idx = torch.cat([torch.arange(b * 128, (b + 1) * 128) for b in order])
    return idx, torch.argsort(idx)


def quant(p, Lin, Lout, s, cb, K, idx=None, inv=None):
    c = np.ascontiguousarray(trellis2.codebook(settle77.CBN[cb]), dtype=np.float32)
    X = p.Wt * s
    if idx is None: return S87.ldlq2(X, Lin, Lout, S87.vit_fun(c, K)) / s
    return S87.ldlq2(X[:, idx].contiguous(), Lin, Lout, S87.vit_fun(c, K))[:, inv] / s


def lin_of(p, sig, idx=None):
    Hq, dq = p.metric(sig); assert dq == 0
    assert int((p.H.diagonal() == 0).sum()) == 0, "dead input columns"
    if idx is not None: Hq = Hq.index_select(0, idx).index_select(1, idx).contiguous()
    return S87.block_ldl(Hq)


def setup(layer, tensor, sample):
    p = S87.prep_nohess(layer, tensor, sample)
    hn = e3.ROUTE[tensor][1]
    Lout, hw, hinfo = S87.hout_metric(layer, tensor, p, S87.SIGMA_OUT)
    HA = settle77.load_h(os.environ["W77_HA"], layer, hn); p.H = HA
    Hs = {"B1": settle77.load_h(os.environ["W77_HB1"], layer, hn).float(),
          "B2": settle77.load_h(os.environ["W77_HB2"], layer, hn).float(), "A": HA}
    p.Wo = p.Wo.double()
    return p, Lout, hw, hinfo, Hs


def rels(p, Q, sig, Hs, hw, ref2, ref1):
    e = S87.scores(p, Q, sig, p.H, Hs, hw)
    rel = {k: e["e2"][k] / ref2[k] for k in Hs}; rel["B"] = (e["e2"]["B1"] + e["e2"]["B2"]) / (ref2["B1"] + ref2["B2"])
    rel1 = {k: e["e1"][k] / ref1[k] for k in Hs}; rel1["B"] = (e["e1"]["B1"] + e["e1"]["B2"]) / (ref1["B1"] + ref1["B2"])
    reln = {k: e["e2_norefit"][k] / ref2[k] for k in Hs}
    return rel, rel1, reln


def job(layer, tensor, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{layer}-{tensor}-s{sample}")
    fn = lambda cb, Ks: f"{stem}-{cb}-K{Ks}.jsonl"
    todo = [(cb, Ks) for cb, Ks in LEVELS if not os.path.exists(fn(cb, Ks))]
    if todo:
        t00 = time.time()
        p, Lout, hw, hinfo, Hs = setup(layer, tensor, sample)
        ref2, ref1 = S87.refs(p, Hs, hw)
        rd = is_reader(tensor)
        sig = S87.sigma_current(tensor, float(Fraction(todo[0][1])))
        assert all(S87.sigma_current(tensor, float(Fraction(Ks))) == sig for _, Ks in todo)
        idx, inv = perm(ORDER) if rd else (None, None)
        t0 = time.time(); Lin = lin_of(p, sig, idx); sec_metric = round(time.time() - t0, 1)
        Lnat = lin_of(p, sig) if rd else None
        for cb, Ks in todo:
            K = float(Fraction(Ks)); sc = S87.get_scale(layer, tensor, sample, cb, Ks); s0 = sc["s_exl"]
            arms = [("order" if rd else "natural", m) for m in GAINS] + ([("noorder", 1.0)] if rd else [])
            lines = []
            for arm, m in arms:
                t0 = time.time()
                Q = quant(p, Lnat, Lout, s0 * m, cb, K) if arm == "noorder" else quant(p, Lin, Lout, s0 * m, cb, K, idx, inv)
                rel, rel1, reln = rels(p, Q, sig, Hs, hw, ref2, ref1); del Q
                lines.append({"exp": "w93", "layer": layer, "w": tensor, "group": "input" if tensor in S87.INPUT else "rest",
                              "sample": sample, "cb": cb, "bits": K, "gain": m, "arm": arm, "reader": rd, "sigma": sig,
                              "s_exl": s0, "drift": sc["drift"], "rel": rel, "rel1": rel1, "rel_norefit": reln,
                              "sec": round(time.time() - t0, 1), "sec_metric": sec_metric, "N": p.N, "Kd": p.Kd, **hinfo,
                              **{k: v for k, v in p.extra.items()}})
                print(json.dumps({k: v for k, v in lines[-1].items() if k not in ("blocks",)}), flush=True)
            for o in lines: o["sec_job"] = round(time.time() - t00, 1)
            with open(fn(cb, Ks) + ".part", "w") as fh:
                for o in lines: fh.write(json.dumps(o) + "\n")
            os.replace(fn(cb, Ks) + ".part", fn(cb, Ks))
    if all(os.path.exists(fn(cb, Ks)) for cb, Ks in LEVELS):
        open(stem + ".DONE", "w").close()


def selftest(layer, tensor, sample, out):
    """identity order through the permuting path == the natural path (same Q bit for bit); the 117 order != natural"""
    p, Lout, hw, hinfo, Hs = setup(layer, tensor, sample)
    sig = S87.sigma_current(tensor, 2.5); s0 = S87.get_scale(layer, tensor, sample, "i8", "2.5")["s_exl"]
    ident = list(range(40)); i0, v0 = perm(ident)
    Qn = quant(p, lin_of(p, sig), Lout, s0, "i8", 2.5)
    Qi = quant(p, lin_of(p, sig, i0), Lout, s0, "i8", 2.5, i0, v0)
    i1, v1 = perm(ORDER)
    Qo = quant(p, lin_of(p, sig, i1), Lout, s0, "i8", 2.5, i1, v1)
    ref2, ref1 = S87.refs(p, Hs, hw)
    rn = rels(p, Qn, sig, Hs, hw, ref2, ref1)[0]; ro = rels(p, Qo, sig, Hs, hw, ref2, ref1)[0]
    o = {"selftest": f"{layer}-{tensor}-s{sample}-i8-K2.5", "identity_max_abs_dQ": float((Qn - Qi).abs().max()),
         "order117_tiles_differ": int(((Qn - Qo).abs().reshape(8, 16, -1, 16).amax(dim=(1, 3)) > 1e-12).sum()),
         "rel_B_natural": rn["B"], "rel_B_order117": ro["B"], "order117_over_natural": ro["B"] / rn["B"]}
    o["PASS"] = o["identity_max_abs_dQ"] == 0.0 and o["order117_tiles_differ"] > 0
    print(json.dumps(o), flush=True)
    with open(out + ".part", "w") as f: json.dump(o, f)
    os.replace(out + ".part", out)


def main():
    if sys.argv[1] == "job": return job(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    if sys.argv[1] == "selftest": return selftest(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
