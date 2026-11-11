# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.seedsd ROOT [--json OUT] -- the seed SD of the quantizer's proxy from rxrun.run's duplicate-seed jobs
(w105; user 11-11 shortlist item 5).

Each ROOT/<level>/dup/<job>/ holds one tensor quantized with seed_offset + 1; its partner is the same tensor in the
level's own job (ROOT/<level>/out/*/log.jsonl, seed_offset as specified). Per pair d = ln(proxy_1 / proxy_0) (the
proxy_err the job logs: relative, so pairs of different K and families pool); the seed SD of one run's ln proxy is
sqrt(mean d^2 / 2) (two independent draws), reported in % with a 90% bootstrap interval over pairs, per family, per K
band and per family x K band. Last line: "SEEDSD n pairs, pooled SD x%"."""
import argparse, glob, json, math, os, random

BANDS = [(0, 1, "K<1"), (1, 2, "K1-2"), (2, 3, "K2-3"), (3, 4, "K3-4"), (4, 5, "K4-5"), (5, 99, "K>=5")]


def band(K):
    return next(n for a, b, n in BANDS if a <= K < b)


def pairs(root):
    out = []
    for lvd in sorted(glob.glob(f"{root}/*/dup")):
        D = os.path.dirname(lvd); base = {}
        for lp in glob.glob(f"{D}/out/*/log.jsonl"):
            for x in open(lp):
                try: r = json.loads(x)
                except ValueError: continue
                base[r["key"]] = r
        for lp in glob.glob(f"{lvd}/*/log.jsonl"):
            sp = json.load(open(f"{os.path.dirname(lp)}/spec.json"))
            for x in open(lp):
                try: r = json.loads(x)
                except ValueError: continue
                b = base.get(r["key"])
                if b is None or not (r.get("proxy_err", 0) > 0 and b.get("proxy_err", 0) > 0): continue
                out.append({"level": os.path.basename(D), "K": float(sp["K"]), "key": r["key"], "fam": r["rxq"]["family"],
                            "p0": b["proxy_err"], "p1": r["proxy_err"], "d": math.log(r["proxy_err"] / b["proxy_err"])})
    return out


def sd(ds, boot=1000, seed=0):
    if not ds: return None
    v = math.sqrt(sum(d * d for d in ds) / len(ds) / 2) * 100
    if len(ds) < 3: return v, None, None
    rng = random.Random(seed); bs = []
    for _ in range(boot):
        s = [ds[rng.randrange(len(ds))] for _ in ds]; bs.append(math.sqrt(sum(d * d for d in s) / len(s) / 2) * 100)
    bs.sort()
    return v, bs[int(0.05 * boot)], bs[int(0.95 * boot) - 1]


def fmt(x, n):
    if x is None: return "-"
    v, lo, hi = x
    return f"{v:.3g}% (n {n})" if lo is None else f"{v:.3g}% [{lo:.3g}, {hi:.3g}] (n {n})"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.seedsd"); ap.add_argument("root"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    P = pairs(a.root)
    fams = sorted({p["fam"] for p in P}); bands = [n for _, _, n in BANDS if any(band(p["K"]) == n for p in P)]
    print(f"seed SD of ln proxy_err, % (90% bootstrap over pairs); {len(P)} pairs from {a.root}")
    print(f"{'family':12s} {'all K':>28s}  " + "  ".join(f"{b:>18s}" for b in bands))
    rows = {}
    for f in fams:
        ds = [p["d"] for p in P if p["fam"] == f]
        cells = []
        for b in bands:
            dd = [p["d"] for p in P if p["fam"] == f and band(p["K"]) == b]
            cells.append(f"{sd(dd, boot=0)[0]:.3g}% n{len(dd)}" if dd else "-")
        rows[f] = sd(ds)
        print(f"{f:12s} {fmt(rows[f], len(ds)):>28s}  " + "  ".join(f"{c:>18s}" for c in cells))
    for b in bands:
        ds = [p["d"] for p in P if band(p["K"]) == b]; print(f"{'band ' + b:12s} {fmt(sd(ds), len(ds)):>28s}")
    allsd = sd([p["d"] for p in P])
    if a.json: json.dump({"pairs": P, "by_family": rows}, open(a.json, "w"), indent=1)
    print(f"SEEDSD {len(P)} pairs, pooled SD {fmt(allsd, len(P))}")


if __name__ == "__main__":
    main()
