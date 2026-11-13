# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sol133.py -- #173 step 3 (RULES-173.md): re-solve the knapsack at every search size with a fitted model (the driver's
`solve` move: moves.mckp from the size's incumbent, window [size - 0.02, size] bpw with the driver's byte offset) and
compare the pick with the current incumbent and w129's fit129h2 pick (the run's init assignments).

  python3 sol133.py OUT.json DS CANDS FITDIR CUT MODEL[,MODEL..] --state STATE --assign ASSIGNDIR --init INITDIR
                    [--sizes 1.5,..] [--bytes-off 8255936] [--tol 0.02] [--save-dir DIR]
Per size and model: the pick's bpw, tensors changed and bytes moved (sum |d bytes| over tensors) vs the incumbent and vs
the init pick, and every model's predicted log ratio (c, x, part mean) of the pick vs the incumbent (cross-predictions:
each pick under each model). Picks are written as assignment JSON (--save-dir) for the lead; nothing is submitted."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import ev133 as E             # noqa: E402
import moves as MV            # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("ds"); ap.add_argument("cands"); ap.add_argument("fitdir"); ap.add_argument("cut", type=int)
    ap.add_argument("models"); ap.add_argument("--state", required=True); ap.add_argument("--assign", required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--sizes", default=None)
    ap.add_argument("--bytes-off", type=int, default=8255936); ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args(argv)
    ds = E.DS(a.ds); cands = json.load(open(a.cands)); M = ds.M
    M.fixed = M.fixed + a.bytes_off
    inc = {}
    for l in open(a.state):
        try: e = json.loads(l)
        except ValueError: continue
        if e["ev"] == "incumbent": inc[e["size"]] = e["tag"]
    sizes = [float(s) for s in a.sizes.split(",")] if a.sizes else sorted(inc)
    names = a.models.split(",")
    mods = {n: E.get_model(ds, cands, n, a.cut, a.fitdir, refit=False) for n in names}
    preds = {n: MV.Pred(Mo, x) for n, (Mo, x) in mods.items()}

    def diff(a0, a1):
        ch = [w for w in a0 if a0[w] != a1.get(w)]
        mb = sum(abs(M.opt(w, a1[w])["bytes"] - M.opt(w, a0[w])["bytes"]) for w in ch)
        return len(ch), int(mb)

    res = {"cut": a.cut, "models": names, "bytes_off": a.bytes_off, "sizes": {}}
    for size in sizes:
        t = inc[size]; a0 = json.load(open(f"{a.assign}/{t}.json"))
        ip = f"{a.init}/init-{size:g}.json"; ai = json.load(open(ip)) if os.path.exists(ip) else None
        hi = size * M.P / 8; lo = (size - a.tol) * M.P / 8
        picks = {"incumbent": a0}
        if ai: picks["init"] = ai
        for n in names: picks[f"solve-{n}"] = MV.mckp(preds[n], a0, hi, lo)
        r = {"incumbent": t, "picks": {}}
        for k, asg in picks.items():
            if asg is None: r["picks"][k] = None; continue
            row = {"bpw": M.bpw(asg), "vs_inc": diff(a0, asg), "vs_init": diff(ai, asg) if ai else None, "pred": {}}
            for n in names:
                Mo, x = mods[n]
                c = Mo.logpart(x, [asg, a0], "c"); xx = Mo.logpart(x, [asg, a0], "x")
                row["pred"][n] = {"c": float(c[0] - c[1]), "x": float(xx[0] - xx[1]), "pmean": float(0.5 * (c[0] - c[1] + xx[0] - xx[1]))}
            r["picks"][k] = row
            if a.save_dir and k.startswith("solve"):
                os.makedirs(a.save_dir, exist_ok=True); json.dump(asg, open(f"{a.save_dir}/{k}-{size:g}.json", "w"))
        res["sizes"][str(size)] = r
        print(f"size {size}: " + " | ".join(f"{k} {v['bpw']:.4f} bpw ch {v['vs_inc'][0]} moved {v['vs_inc'][1] / 1e6:.0f} MB "
                                            + " ".join(f"{n}:{v['pred'][n]['pmean']:+.4f}" for n in names)
                                            for k, v in r["picks"].items() if v), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
