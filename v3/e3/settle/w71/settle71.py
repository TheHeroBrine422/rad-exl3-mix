#!/usr/bin/env python3
"""settle71.py LAYER TENSOR K OUT.jsonl [ROWS] -- w71 (#113, ../w71.md): the trellis state window L (16 / 18 / 20) and
a beam (M-algorithm) search on the settle proxy, exllamav3's LDLQ order (w64's settle64.py, SETTLE_ORDER=rev).

Everything except the per-tile trellis search is settle.py / settle64.py unchanged: P2's transforms on 128 aligned
rows (E3_SEED signs, E3_ROWSEED block), adopted damping, exllamav3's g_scale search, block LDLQ last block first, rel
after the per-output-channel refit (rel_norefit before) under the undamped calib H. The search inside (trellis.vit)
is swapped per arm (trellisL.py / beamL.c), all arms paired on the same rows, signs, metric and codes path:
  V<L>        full Viterbi, L-bit window, exllamav3's tail-biting heuristic. V16 = today's EXL3 (bit for bit
              viterbi.c's viterbi_tiles_tb ntry 1). Every V / B arm quantizes at V16's g_scale (x m with "@m"), the
              search run once per (tensor, sample, K) by the first job that needs it and shared through
              W71_SCALES/LAYER-TENSOR-sSEED-KK.json (flock), so the arms can run as separate jobs
  B<L>-<M>    the beam with M survivors, L-bit window, the same tail-biting heuristic
  F16         full Viterbi, free start, its own g_scale search with it (gate: == S16)
  S16         settle.py's own path: trellis.vit / libviterbi.so free start, its own scale search ("today's numbers")
  T16         settle.py's own path with E3_TB=1 (viterbi_tiles_tb ntry 1), its own scale search (gate: == V16)
  <arm>r      a V / B arm with an iid Gaussian table on the codebook's grid instead of the mul1 hash
              (trellisL.rand_table: the "ideal code" ceiling, no cheap decoder), at V16's scale
Codebook (W71_CB, default the ladder's: mul1_i8 below K6, mul1 from K6) in its L-bit form (trellisL.codebook_L: the
same byte-sum value with a 2^L - 1 window).
W71_ARMS: comma list (default below). Arms already in OUT.arms (one JSON line each) are skipped, so a killed job
resumes at its next arm; OUT (one JSON line with every arm) is written whole at the end (.part, renamed)."""
import json, os, sys, time
from fractions import Fraction
os.environ.setdefault("SETTLE_ORDER", "rev")
_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, "../w64"), os.path.join(_h, ".."), os.path.join(_h, "../..")]
import numpy as np
import torch
import settle64                         # noqa: F401  (patches trellis.ldlq_trellis to exllamav3's block order)
import settle, trellis, trellis2
import trellisL as TL
torch.set_grad_enabled(False)

ARMS_LO = "V16,V18,B16-256,B16-1024,B16-4096,B18-1024,B18-4096,B20-256,B20-1024,B20-4096"   # K <= 4
ARMS_HI = "V16,V18,B16-4096,B18-4096,B20-1024,B20-4096"                                      # K5 / K6


def parse(arm):
    """-> (kind, L, M, m)"""
    m = 1.0
    if "@" in arm: arm, ms = arm.split("@"); m = float(ms)
    if arm in ("S16", "T16"): return arm, 16, 0, m
    kind, rest = arm[0], arm[1:]
    if "-" in rest: L, M = rest.split("-"); return kind, int(L), int(M), m
    return kind, int(rest), 0, m


def main():
    layer, tensor, K, out = int(sys.argv[1]), sys.argv[2], float(Fraction(sys.argv[3])), sys.argv[4]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    assert K.is_integer(), "integer K only (tail-biting)"
    seed, rseed = int(os.environ.get("E3_SEED", 0)), int(os.environ.get("E3_ROWSEED", 0))
    cbn = os.environ.get("W71_CB") or ("mul1_i8" if K < 6 else "mul1")
    arms = os.environ.get("W71_ARMS") or (ARMS_LO if K <= 4 else ARMS_HI)
    arms = arms.split(",")
    t00 = time.time()
    cb16 = np.ascontiguousarray(trellis2.codebook(cbn), dtype=np.float32)
    tabs = {}
    def tab(L, code):
        if (L, code) not in tabs:
            tabs[L, code] = TL.codebook_L(cbn, L, cb16) if code == cbn else TL.rand_table(L, cbn)
        return tabs[L, code]
    p = settle.P2(layer, tensor, rows, seed, rseed)
    fam_in = tensor in settle.INPUT; damp0 = 0.005 if fam_in else 0.025
    Hq, dq = p.metric(damp0)
    base = {"exp": "w71", "order": settle64.ORDER, "layer": layer, "w": tensor, "group": "input" if fam_in else "rest",
            "bits": K, "cb": cbn, "damp": damp0, "seed": seed, "rowseed": rseed, "rows": rows, "N": p.N, "K": p.Kd,
            **p.extra}
    done = {}
    if os.path.exists(out + ".arms"):
        for l in open(out + ".arms"):
            try: o = json.loads(l); done[o["arm"]] = o
            except ValueError: pass                         # a line cut by a kill
    orig_vit = trellis.vit
    acc = {"sse": 0.0, "n": 0}
    def mk(kind, L, M, code=cbn):
        if kind == "S16": return orig_vit
        if kind == "T16":
            def v(x, cb, KK):
                os.environ["E3_TB"] = "1"
                try: return orig_vit(x, cb, KK)
                finally: os.environ.pop("E3_TB", None)
            return v
        tb = 0 if kind == "F" else 1
        def v(x, cb, KK):
            q, c = TL.search(x, tab(L, code), L, KK, M, tb, want_cost=True)
            acc["sse"] += float(c.astype(np.float64).sum()); acc["n"] += x.size
            return q
        return v
    sdir = os.environ.get("W71_SCALES", "scales")
    def scale_tb():
        """V16's g_scale search for this (tensor, sample, K): computed once (flock), shared by every job"""
        import fcntl
        os.makedirs(sdir, exist_ok=True)
        f = os.path.join(sdir, f"{layer}-{tensor}-s{seed}-r{rseed}-{rows}-K{sys.argv[3]}-{cbn}.json")
        with open(f + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(f): o = json.load(open(f)); return o["s"], o["search"], 0.0
            trellis.vit = mk("V", 16, 0); t = time.time()
            s, sx = settle.search_scale(p.T, cb16, K); dt = time.time() - t
            with open(f + ".part", "w") as fh: json.dump({"s": s, "search": sx, "sec": dt}, fh)
            os.replace(f + ".part", f)
            return s, sx, dt
    for arm in arms:
        if arm in done: continue
        code = "rand" if arm.endswith("r") else cbn
        kind, L, M, m = parse(arm[:-1] if arm.endswith("r") else arm)
        t0 = time.time(); sx = None
        if kind in ("S16", "T16", "F"):
            trellis.vit = mk(kind, L, M)
            s, sx = settle.search_scale(p.T, cb16, K)
        else:
            s, sx, _ = scale_tb(); s *= m
            if arm != "V16": sx = None
        trellis.vit = mk(kind, L, M, code)
        t1 = time.time(); acc["sse"] = 0.0; acc["n"] = 0
        Q = trellis.ldlq_trellis(p.Wt * s, Hq, cb16, K, dq)
        t2 = time.time()
        r, r0 = settle.rels(p, Q / s, damp0)[0]; del Q
        o = {"arm": arm, "L": L, "M": M, "m": m, "s": s, "rel": r, "rel_norefit": r0, "sec_ldlq": round(t2 - t1, 1),
             "sec_search": round(t1 - t0, 1), "search_mse": acc["sse"] / acc["n"] if acc["n"] else None,
             **({"search": sx} if sx else {})}
        with open(out + ".arms", "a") as fh: fh.write(json.dumps(o) + "\n")
        print(json.dumps({k: v for k, v in o.items() if k != "search"}), flush=True)
        done[arm] = o
    trellis.vit = orig_vit
    res = {**base, "arms": [a for a in arms], "rel": {a: done[a]["rel"] for a in arms},
           "rel_norefit": {a: done[a]["rel_norefit"] for a in arms}, "s": {a: done[a]["s"] for a in arms},
           "sec_ldlq": {a: done[a]["sec_ldlq"] for a in arms}, "sec_search": {a: done[a]["sec_search"] for a in arms},
           "search_mse": {a: done[a]["search_mse"] for a in arms},
           "search": {a: done[a]["search"] for a in arms if "search" in done[a]}, "sec_job": round(time.time() - t00, 1)}
    with open(out + ".part", "w") as fh: fh.write(json.dumps(res) + "\n")
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main()
