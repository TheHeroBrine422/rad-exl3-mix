# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""solve202.py -- RULES-floor202 §3: the driver's solve / thompson moves on a fitfloor prediction, as inbox decide files.

  solve202.py state DRV OUT.json            (VM) the incumbents' assignments (1.5-7.0) and every submitted hash
  solve202.py solve IN.json FIT.out.json STATE.json OUTDIR SIZES(csv) [floor|sat] [NAME_SUFFIX]
                                            (colo) IN = the fit's input (menu, obs, opts); FIT = fit202.py's output on it.
      per size: `opt` = moves.mckp(pred, incumbent, window) (kind solve); `ts` = mckp on a posterior draw (kind thompson;
      draws tried in a seeded order (seed 202) until the allocation is new). Window = driver.window: bytes in
      [size - 0.02, size] bpw. A candidate equal to the incumbent or to any submitted hash is not written (reported)."""
import json, math, os, random, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
TOL = 0.02


def state(drv, out):
    inc, seen = {}, set(); SIZES = [1.5, 1.75, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    for l in open(f"{drv}/state.jsonl"):
        e = json.loads(l)
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
        if e.get("ev") == "submit" and e.get("hash"): seen.add(e["hash"])
    json.dump({"inc": {str(s): {"tag": inc[s], "assign": json.load(open(f"{drv}/assign/{inc[s]}.json"))} for s in SIZES}, "seen": sorted(seen)}, open(out, "w"))
    print({s: inc[s] for s in SIZES}, len(seen), "hashes")


def solve(inp, fitp, statep, outdir, sizes, term="floor", suffix=""):
    SIZES = [float(z) for z in sizes.split(",")]
    import fit202 as F2, menu as MN, moves as MV
    d = json.load(open(inp)); ft = json.load(open(fitp)); S = json.load(open(statep))
    M = MN.Menu(d["menu"]) if isinstance(d["menu"], dict) else MN.Menu.load(d["menu"])
    Mo = F2.build(M, d["obs"], ft["opts"]); Mo.layout(); x = np.array(ft["x"]); assert len(x) == Mo.N
    pred = MV.Pred(Mo, x); draws = [MV.Pred(Mo, np.array(v)) for v in ft["draws"]]
    seen = set(S["seen"]); os.makedirs(outdir, exist_ok=True); rep = []
    for size in SIZES:
        a0 = S["inc"][str(size)]["assign"]; hi = size * M.P / 8; lo = (size - TOL) * M.P / 8
        assert set(a0) == set(M.W) if hasattr(M, "W") else True
        order = list(range(len(draws))); random.Random(202 + int(size * 100)).shuffle(order)
        cands = [("opt", "solve", lambda: MV.mckp(pred, a0, hi, lo))]
        got_ts = False
        for var, kind, f in cands + [("ts", "thompson", (lambda j=j: MV.mckp(draws[j], a0, hi, lo))) for j in order]:
            if var == "ts" and got_ts: break
            c = f()
            if c is None: rep.append((size, var, "no solution")); continue
            h = MV.ahash(c); b = M.bytes(c); bpw = M.bpw(c)
            nd = sum(1 for w in c if c[w] != a0[w]); plr = math.log(pred.kl(c) / pred.kl(a0))
            if h == MV.ahash(a0) or h in seen:
                rep.append((size, var, f"not new ({'= incumbent' if h == MV.ahash(a0) else 'already submitted'}; pred log r {plr:+.4f})")); continue
            if not (lo <= b <= hi): rep.append((size, var, f"bytes outside the window ({bpw:.4f})")); continue
            for t in M.T: assert c[t["w"]] in {o["o"] for o in t["opts"]}
            assert len(c) == len(M.T)
            name = f"w188-{term}-{size:g}-{var}{suffix}.json"
            json.dump({"size": size, "mode": "decide", "assign": c,
                       "note": f"#202 RULES-floor202 §3: {kind} from fit{term} (fit202.py, {'F' if term == 'floor' else 'S'} {ft['model']['F' if term == 'floor' else 'S']:.3g}) on {os.path.basename(inp)}; "
                               f"from incumbent {S['inc'][str(size)]['tag']}; {nd} tensors changed; model pred log r {plr:+.4f}; bpw {bpw:.4f}"},
                      open(os.path.join(outdir, name), "w"))
            seen.add(h); rep.append((size, var, f"{name}: bpw {bpw:.4f}, {nd} changed, pred log r {plr:+.4f}, hash {h}"))
            if var == "ts": got_ts = True
    for r in rep: print(*r)
    json.dump(rep, open(os.path.join(outdir, f"report-{term}-{SIZES[0]:g}-{SIZES[-1]:g}{suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "state": state(a[1], a[2])
    else: solve(*a[1:])
