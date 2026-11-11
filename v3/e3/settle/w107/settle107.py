#!/usr/bin/env python3
"""settle107.py -- w107 (../w107.md) on w100's / w102's proxies (their code imported unchanged).

  settle107.py drift-a|drift-b LAYER TENSOR SAMPLE OUTDIR [DONE]   #141-01: one file per (K, arm),
        OUTDIR/LAYER-TENSOR-sS-K{K}-{arm}.jsonl; phase a = cur at the new K + tab at the old K, phase b = tab at the new K
        cur  at the new K (W107_NEW_K, default 0.25,0.375,0.625,0.75,0.875): w100's sub1 cur arm (exllamav3's search with
             today's drift 1.08, quantized; measured d_all, rel2 / rel1)
        tab  at every K whose T(K) is known: the search with drift T(K) (W107_TABLE json {K: T}; K0.125 / 0.5 / 1 are
             built in), quantized only if the scale moves ("same": true otherwise). At the old K the cur scale is the
             stored one (w100's), at the new K the cur file's.
        DONE is touched when every file of the phase exists.
  settle107.py g1 LAYER TENSOR SAMPLE OUTDIR            gate G1: cur at K0.125 / 0.5 / 1 (as w100's), one file
        OUTDIR/g1-LAYER-TENSOR-sS.jsonl (compared with w100's stored rows by report107.py)
  settle107.py ntry LAYER TENSOR SAMPLE OUTDIR [DONE]   #141-03: per K of W107_NTRY_KS (0.125,0.5,1), w102's d3 setup,
        arms T1 and T4 (two-sided), one file OUTDIR/LAYER-TENSOR-sS-K{K}-ntry.jsonl
Env: w100's env.sh (sub1 / g1 / drift), w102's (ntry)."""
import json, os, sys, time
from fractions import Fraction
import numpy as np

_h = os.path.dirname(os.path.abspath(__file__))
NEW_K = os.environ.get("W107_NEW_K", "0.25,0.375,0.625,0.75,0.875").split(",")
OLD_K = ["0.125", "0.5", "1"]
TABLE0 = {"0.125": 1.348, "0.5": 1.184, "1": 1.089}         # w100's D rounded to 3 decimals (../w107.md)


def table():
    t = dict(TABLE0); f = os.environ.get("W107_TABLE")
    if f and os.path.exists(f): t.update({k: float(v) for k, v in json.load(open(f)).items()})
    return t


def s100(layer, tensor, sample, Ks):
    """the scale w100's cur arm used (its stored row's "s"; W107_S100 = json {"L-T-sS-K": s} made from data/w100)"""
    return json.load(open(os.environ["W107_S100"]))[f"{layer}-{tensor}-s{sample}-K{Ks}"]


def write(fn, lines):
    with open(fn + ".part", "w") as fh:
        for o in lines: fh.write(json.dumps(o) + "\n")
    os.replace(fn + ".part", fn)


class Sub1:
    """w100's sub1 job state for one tensor x sample (settle100's pieces, same order of operations)"""
    def __init__(s, layer, tensor, sample):
        sys.path.insert(0, os.path.join(_h, "..", "w100"))
        import settle100 as S
        s.S = S; s.layer, s.tensor, s.sample = layer, tensor, sample
        s.p, s.Hs, r, idx = S.setup(layer, tensor, sample)
        hF = S.hvec(tensor, layer, s.p, "F"); s.hw = hF[idx]
        s.Lout = S.lout_of(s.p, hF, r, 1.0, S.S87.SIGMA_OUT)
        s.ref = S.refrows(s.p, s.Hs); s.lc = {}
        s.cbv = np.ascontiguousarray(S.trellis2.codebook(S.settle77.CBN[S.CB]), dtype=np.float32)

    def arm(s, arm, K, sc, rec, dr, extra):
        """quantize at scale sc and score as settle100.sub1's one() (rel2 / rel1 after the refit and before)"""
        S, p, Hs, hw = s.S, s.p, s.Hs, s.hw
        sig = S.S87.sigma_current(s.tensor, K); S.lin_for(p, s.tensor, sig, s.lc)
        t0 = time.time()
        Q, ds = S.quant(p, s.lc, s.Lout, sc, K)
        e = S.rowerr(p, Q, sig, Hs); del Q
        o = S.base(s.layer, s.tensor, s.sample, K, p, "w107-drift")
        o.update(arm=arm, sigma=sig, sigma_out=S.S87.SIGMA_OUT, s=sc, search=rec, drift_used=dr, measured=ds, **extra)
        for tag in ("e", "e0"):
            rel2 = {k: float((hw * e[tag][k]).sum() / (hw * s.ref[k]).sum()) for k in Hs}
            rel1 = {k: float(e[tag][k].sum() / s.ref[k].sum()) for k in Hs}
            for rr, w in ((rel2, hw), (rel1, None)):
                ww = (lambda v: w * v) if w is not None else (lambda v: v)
                rr["B"] = sum(float(ww(e[tag][k]).sum()) for k in ("B1", "B2")) / sum(float(ww(s.ref[k]).sum()) for k in ("B1", "B2"))
                rr["pub"] = sum(float(ww(e[tag][k]).sum()) for k in Hs) / sum(float(ww(s.ref[k]).sum()) for k in Hs)
            o["rel2" if tag == "e" else "rel2_norefit"] = rel2; o["rel1" if tag == "e" else "rel1_norefit"] = rel1
        o["sec"] = round(time.time() - t0, 1)
        print(json.dumps({k: v for k, v in o.items() if k not in ("blocks", "search")}), flush=True)
        return o


def drift(layer, tensor, sample, outdir, phase, done=None):
    os.makedirs(outdir, exist_ok=True)
    fn = lambda Ks, arm: os.path.join(outdir, f"{layer}-{tensor}-s{sample}-K{Ks}-{arm}.jsonl")
    T = table()
    want = [(Ks, "cur") for Ks in NEW_K] + [(Ks, "tab") for Ks in OLD_K] if phase == "a" else [(Ks, "tab") for Ks in NEW_K]
    if phase == "b": assert all(Ks in T for Ks in NEW_K), "phase b needs W107_TABLE with every new K"
    todo = [(Ks, a) for Ks, a in want if not os.path.exists(fn(Ks, a))]
    if todo:
        c = Sub1(layer, tensor, sample); S = c.S
        for Ks, arm in todo:
            K = float(Fraction(Ks)); t0 = time.time()
            if arm == "cur":
                d0 = S.settle77.drift(K)
                s0, rec = S.search(c.p.T, c.cbv, K, d0, S.COARSE)
                o = c.arm("cur", K, s0, rec, d0, {"sec_search": round(time.time() - t0, 1)})
                write(fn(Ks, "cur"), [o]); continue
            if Ks in OLD_K:
                s0 = s100(layer, tensor, sample, Ks)                      # w100's cur arm's scale
            else:
                if not os.path.exists(fn(Ks, "cur")): continue
                s0 = json.loads(open(fn(Ks, "cur")).readline())["s"]
            s1, rec = S.search(c.p.T, c.cbv, K, T[Ks], S.COARSE)
            ex = {"sec_search": round(time.time() - t0, 1), "s_cur": s0, "table": T[Ks]}
            if abs(s1 / s0 - 1) > 1e-12:
                o = c.arm("tab", K, s1, rec, T[Ks], ex)
            else:
                o = {"exp": "w107-drift", "layer": layer, "w": tensor, "sample": sample, "bits": K, "arm": "tab",
                     "same": True, "s": s1, "search": rec, "drift_used": T[Ks], **ex}
                print(json.dumps(o), flush=True)
            write(fn(Ks, "tab"), [o])
    if done and all(os.path.exists(fn(Ks, a)) for Ks, a in want): open(done, "w").close()


def g1(layer, tensor, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    f = os.path.join(outdir, f"g1-{layer}-{tensor}-s{sample}.jsonl")
    if os.path.exists(f): return
    c = Sub1(layer, tensor, sample); S = c.S; lines = []
    for Ks in OLD_K:
        K = float(Fraction(Ks)); s0 = s100(layer, tensor, sample, Ks)
        lines.append(c.arm("cur", K, s0, None, S.settle77.drift(K), {"gate": "G1"}))
    write(f, lines)


def ntry(layer, tensor, sample, outdir, done=None):
    sys.path.insert(0, os.path.join(_h, "..", "w102"))
    import job102 as J
    from job102 import settle87, settle77, trellis, trellis2
    os.makedirs(outdir, exist_ok=True)
    Ks_all = os.environ.get("W107_NTRY_KS", "0.125,0.5,1").split(",")
    fn = lambda Ks: os.path.join(outdir, f"{layer}-{tensor}-s{sample}-K{Ks}-ntry.jsonl")
    todo = [k for k in Ks_all if not os.path.exists(fn(k))]
    if todo:
        ctx = J.Ctx(layer, tensor, sample); p = ctx.p
        for Ks in todo:
            K = float(Fraction(Ks)); sig = settle87.sigma_current(tensor, K)
            sc = settle87.get_scale(layer, tensor, sample, "i8", Ks); s = sc["s_exl"]
            cb = np.ascontiguousarray(trellis2.codebook(settle77.CBN["i8"]), dtype=np.float32)
            lines = []
            for arm, n in (("T1", 1), ("T4", 4)):
                Lin, _ = ctx.metric(sig, False)
                t0 = time.time()
                Q = settle87.ldlq2(p.Wt * s, Lin, ctx.Lout, lambda t: J.ring(t, cb, K, 1, n)) / s
                tq = time.time() - t0
                e = settle87.scores(p, Q, sig, p.H, ctx.Hs, ctx.hw); del Q
                lines.append({**ctx.head(K, sig, sc), "exp": "w107", "item": "ntry", "arm": arm, "ntry": n,
                              "rel": ctx.rel(e["e2"]), "rel1": ctx.rel1(e["e1"]), "sec": round(tq, 1)})
                print(json.dumps(lines[-1]), flush=True)
            write(fn(Ks), lines)
    if done and all(os.path.exists(fn(k)) for k in Ks_all): open(done, "w").close()


def main():
    m, layer, tensor, sample, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    done = sys.argv[6] if len(sys.argv) > 6 else None
    if m in ("drift-a", "drift-b"): return drift(layer, tensor, sample, outdir, m[-1], done)
    if m == "g1": return g1(layer, tensor, sample, outdir)
    if m == "ntry": return ntry(layer, tensor, sample, outdir, done)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
