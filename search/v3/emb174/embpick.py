#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""embpick.py MENU CAL.json... [--b fit|v2] [--json OUT] -- the per-size embedding pick from the measured swaps (w135,
#174), not from a fitted embedding term (leave-one-out: neither kfit3's P_0 offset nor an offset after the
compounding extrapolates across sizes). CAL: calobs.py --json outputs (f16 and A8 anchors; per eval the pmean log
ratio vs l3 ("base") and its SE; several evals of one (option, size): inverse-variance mean). Model per size: upgrades
(l4..bf16) = - k_option x F(size), F log-linear in size between the measured sizes (flat outside), k fitted (e5 = 1);
l2 = its own log-linear cost curve. Score = that log ratio plus b(size) x the option's extra bpw (the size-adjusted score of the search: what the same bytes buy in the trunk). b: the driver's last
fitted slopes ("fit", 11-13 08:38 CST status; depressed at >= 5.5 by the l3 floor itself) or V2's board slopes ("v2",
driver.B_DEFAULT). Pick = the lowest score among options with a value; "base" scores 0."""
import argparse, json, math, sys

B_FIT = {1.5: 2.721, 1.75: 1.79, 2.0: 1.65, 2.5: 1.50, 3.0: 1.40, 3.5: 1.29, 4.0: 1.17, 4.5: 1.00, 5.0: 0.79, 5.5: 0.54, 6.0: 0.32, 6.5: 0.13, 7.0: 0.70}
B_V2 = [(2.5, 1.6), (3.0, 1.6), (4.0, 1.1), (5.5, 1.1), (6.0, 0.7), (7.0, 0.7)]
SIZES = [1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
PARAMS = 26895998464


def bv2(s):
    if s <= B_V2[0][0]: return B_V2[0][1]
    for (x0, y0), (x1, y1) in zip(B_V2, B_V2[1:]):
        if s <= x1: return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return B_V2[-1][1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embpick.py")
    ap.add_argument("menu"); ap.add_argument("cal", nargs="+"); ap.add_argument("--b", default="v2"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    M = json.load(open(a.menu)); eo = next(t for t in M["tensors"] if t["w"] == "token_embd.weight")["opts"]
    by = {o["o"]: o["bytes"] for o in eo}; dbpw = {o: (b - by["base"]) * 8 / PARAMS for o, b in by.items()}
    meas = {}
    for p in a.cal:
        for t, r in json.load(open(p)).items():
            meas.setdefault(r["emb"], {}).setdefault(float(r["size"]), []).append((r["pmean"], r["se"], t))
    mv = {}
    for o, d in meas.items():
        for s, v in d.items():
            w = [1 / max(se, 1e-4) ** 2 for _, se, _ in v]
            mv.setdefault(o, {})[s] = (sum(x * wi for (x, _, _), wi in zip(v, w)) / sum(w), 1 / math.sqrt(sum(w)), [t for *_, t in v])
    # upgrades (l4 and up) share one curve F(s) = the l3 floor's share, log-linear between the measured sizes and flat
    # outside them, times a factor per option; l2 (a downgrade) has its own log-linear cost curve, extrapolated with
    # its last slope above the measured sizes (it only gets worse) and flat below
    up = [o for o in by if o not in ("base", "e2") and o in mv]
    k = {o: 1.0 for o in up}
    for _ in range(20):
        F = {}
        for o in up:
            for s, (lr, se, _) in mv[o].items(): F.setdefault(s, []).append((-lr / k[o], 1 / max(se, 1e-4) ** 2))
        F = {s: max(sum(x * w for x, w in v) / sum(w for _, w in v), 1e-4) for s, v in F.items()}
        k = {o: sum(-lr / F[s] / max(se, 1e-4) ** 2 for s, (lr, se, _) in mv[o].items()) / sum(1 / max(se, 1e-4) ** 2 for (lr, se, _) in mv[o].values()) for o in up}
        n = k.get("e5", 1.0); k = {o: v / n for o, v in k.items()}
    Fk = sorted(F.items())

    def logint(pts, s, extrap_up=False):
        if s <= pts[0][0]: return pts[0][1]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if s <= s1: return math.exp(math.log(v0) + (s - s0) / (s1 - s0) * (math.log(v1) - math.log(v0)))
        if extrap_up and len(pts) > 1:
            (s0, v0), (s1, v1) = pts[-2], pts[-1]; return math.exp(math.log(v1) + (s - s1) / (s1 - s0) * (math.log(v1) - math.log(v0)))
        return pts[-1][1]
    l2 = sorted((s, max(v[0], 1e-4)) for s, v in mv.get("e2", {}).items())
    out = {"curve": {"F": F, "k": k, "l2": dict(l2)}}
    print("upgrade curve F(s) (l3 -> e5-equivalent, -log ratio): " + ", ".join(f"{s:g} {v:.4f}" for s, v in Fk)
          + " | option factors " + ", ".join(f"{o} {v:.2f}" for o, v in k.items()) + " | l2 cost " + ", ".join(f"{s:g} {v:+.4f}" for s, v in l2))
    print(f"b = {a.b}; per option: model log ratio vs l3 (m = measured at this size) + b x extra bpw = score (lower is better)")
    for s in SIZES:
        b = B_FIT[s] if a.b == "fit" else bv2(s); row = {"b": b, "opts": {"base": {"score": 0.0}}}
        for o in by:
            if o == "base" or (o not in up and o != "e2"): continue
            lr = logint(l2, s, True) if o == "e2" else -k[o] * logint(Fk, s)
            if o == "e2" and not l2: continue
            m = mv.get(o, {}).get(s)
            row["opts"][o] = {"logr": lr, "meas": m[0] if m else None, "dbpw": dbpw[o], "score": lr + b * dbpw[o]}
        row["pick"] = min(row["opts"], key=lambda o: row["opts"][o]["score"])
        out[s] = row
        print(f"{s:4g} b {b:.2f} | " + " ".join(f"{o} {r['logr']:+.3f}{'m' if r['meas'] is not None else ''}={r['score']:+.3f}" for o, r in row["opts"].items() if o != "base")
              + f" | pick {row['pick']}")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
