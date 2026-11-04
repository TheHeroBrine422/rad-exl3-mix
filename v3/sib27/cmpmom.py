#!/usr/bin/env python3
"""cmpmom.py -- compare moment dirs (moments.py / sibmom27.py: <group>.safetensors "H", "C", "N" + metadata n).
  cmpmom.py diff A B                 per group and stat: ||A - B||_F / ||B||_F and max |A - B| / max |B|; n equal
  cmpmom.py verdict REF REF2 OURS    the GPU equivalence rule (test_gpu.sh): floor = REF2 vs REF (moments.py run twice);
      PASS if OURS vs REF is bit-exact-level (max rel <= 1e-6) when the floor is 0, else if per stat (H, C, N) the
      worst-group relative Frobenius difference of OURS vs REF <= 2 x the floor's (and n equal everywhere).
  cmpmom.py fdose DN DNHALF          per wpre file of DNHALF (mkdenoise on the N/2 snapshots): ||F_half - F|| / ||F - I||
      (F^T as stored, f32 on DEV): how much of the filter's correction is still sampling noise at N/2 (reported)."""
import glob, os, sys
import torch
from safetensors import safe_open


def load(d):
    out = {}
    return {os.path.basename(f)[:-12]: f for f in sorted(glob.glob(f"{d}/*.input.safetensors"))}


def diff(A, B):
    a, b = load(A), load(B)
    assert a.keys() == b.keys() and a, (len(a), len(b), sorted(set(a) ^ set(b))[:3])
    res = {}
    for g in a:
        with safe_open(a[g], "pt") as x, safe_open(b[g], "pt") as y:
            r = {"n": x.metadata()["n"] == y.metadata()["n"]}
            for s in "HCN":
                u, v = x.get_tensor(s).double(), y.get_tensor(s).double()
                r[s] = (float((u - v).norm() / v.norm().clamp_min(1e-300)), float((u - v).abs().max() / v.abs().max().clamp_min(1e-300)))
        res[g] = r
    return res


def worst(res, s, j): return max(r[s][j] for r in res.values())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fdose":
        dev = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu"); out = []
        for f in sorted(glob.glob(f"{sys.argv[3]}/wpre/*.safetensors")):
            g = os.path.basename(f)
            with safe_open(f"{sys.argv[2]}/wpre/{g}", "pt", device=dev) as x, safe_open(f, "pt", device=dev) as y:
                F, Fh = x.get_tensor("Ft").float(), y.get_tensor("Ft").float()
            num = float((Fh - F).norm()); F.diagonal().sub_(1)
            out.append(f"{g.split('layers.')[1].rsplit('.', 2)[0]} {num / float(F.norm()):.2f}"); del F, Fh
        print("||F_half - F|| / ||F - I||: " + ", ".join(out)); sys.exit(0)
    if cmd == "diff":
        res = diff(*sys.argv[2:4])
        for g, r in res.items():
            print(f"{g.split('layers.')[1]:32s} n {'=' if r['n'] else 'DIFFERS'}  " + "  ".join(f"{s} {r[s][0]:.1e} / {r[s][1]:.1e}" for s in "HCN"))
        print("worst: " + "  ".join(f"{s} {worst(res, s, 0):.1e} / {worst(res, s, 1):.1e}" for s in "HCN"))
    elif cmd == "verdict":
        REF, REF2, OURS = sys.argv[2:5]
        fl, us = diff(REF2, REF), diff(OURS, REF)
        nok = all(r["n"] for r in list(fl.values()) + list(us.values()))
        f0 = max(worst(fl, s, 1) for s in "HCN")
        line = "  ".join(f"{s} ours {worst(us, s, 0):.1e} floor {worst(fl, s, 0):.1e}" for s in "HCN")
        if f0 == 0: ok = nok and max(worst(us, s, 1) for s in "HCN") <= 1e-6; rule = "floor 0: ours must be <= 1e-6 (max rel)"
        else: ok = nok and all(worst(us, s, 0) <= 2 * worst(fl, s, 0) for s in "HCN"); rule = "ours <= 2 x floor (rel Frobenius, worst group)"
        print(f"MOMENT EQUIVALENCE {'PASS' if ok else 'FAIL'} ({rule}; n equal: {nok}): {line}")
        sys.exit(0 if ok else 1)
