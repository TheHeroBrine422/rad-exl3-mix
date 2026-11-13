#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""loosum.py DIR [--json OUT] -- loo174.sh's leave-one-out results per arm (w135, #174): for every held-out calibration
eval, predicted vs measured per-part log ratio vs its anchor; per arm the RMSE and the mean error (bias) over the
held-out rows, plain and in SE units, plus the in-sample ("none") rows. The arm with the lowest held-out RMSE wins."""
import argparse, glob, json, math, os, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="loosum.py"); ap.add_argument("dir"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    arms = {}
    for p in sorted(glob.glob(f"{a.dir}/*.json")):
        b = os.path.basename(p)[:-5]
        if b.startswith("arm-"): continue
        arm, ho = b.split("-", 1); d = json.load(open(p))
        e = arms.setdefault(arm, {"held": {}, "in": {}, "emb": None})
        if ho == "none": e["in"] = d["rows"]; e["emb"] = d["emb"]
        else: e["held"].update(d["rows"])
    out = {}
    for arm, e in sorted(arms.items()):
        r = {}
        for k in ("held", "in"):
            v = list(e[k].values())
            if not v: continue
            err = [x["pred"] - x["meas"] for x in v]; z = [(x["pred"] - x["meas"]) / x["se"] for x in v]
            r[k] = {"n": len(v), "rmse": math.sqrt(sum(t * t for t in err) / len(v)), "bias": sum(err) / len(v),
                    "rmse_z": math.sqrt(sum(t * t for t in z) / len(v))}
        out[arm] = dict(r, emb=e["emb"], rows=e["held"])
        print(f"{arm:5s} " + " | ".join(f"{k} n {s['n']} rmse {s['rmse']:.4f} bias {s['bias']:+.4f} rmse/se {s['rmse_z']:.2f}" for k, s in r.items()))
    print("held-out rows (meas / pred per arm):")
    tags = sorted({t for e in arms.values() for t in e["held"]})
    for t in tags:
        m = next(e["held"][t]["meas"] for e in arms.values() if t in e["held"])
        print(f"  {t:40s} meas {m:+.4f} " + " ".join(f"{arm} {e['held'][t]['pred']:+.4f}" for arm, e in sorted(arms.items()) if t in e["held"]))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
