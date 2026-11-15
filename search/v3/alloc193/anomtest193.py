# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""anomtest193.py SNAP RDIR ANOM.json PAIRS.jsonl OUTDIR [--nflags 6] [--two 2] [--commit C] -- RULES-193 §3 tests.

Flags = anom193.flags (|z| >= 3), distinct (group, band), largest |z| first, at most --nflags. Per flag: the band's
size with the most paired rows first (then the next); the current incumbent with every tensor of the group moved one
own-family menu step in the flag's direction, bytes balanced outside the group by my fit.py refit (bal193). The first
--two flags get two sizes, the rest one (RULES-193 amendment 2: anom budget 8)."""
import argparse, collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import anom193 as A  # noqa: E402
import bal193 as BL  # noqa: E402
import fitopt193 as FO  # noqa: E402
import curv193 as CV  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("rdir"); ap.add_argument("anom"); ap.add_argument("pairs")
    ap.add_argument("out"); ap.add_argument("--nflags", type=int, default=6); ap.add_argument("--two", type=int, default=2)
    ap.add_argument("--commit", default="?"); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--min-bits", type=float, default=0.0, help="skip flags whose move shifts < this many bpw")
    ap.add_argument("--multi", action="store_true", help="amendment 6: up to 4 steps to reach --min-bits")
    ap.add_argument("--groups", help="combined test: comma list of groups (dir -1), sizes from --sizes")
    ap.add_argument("--sizes")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); st = C.load_state(a.snap); inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    res = json.load(open(a.anom)); F = []; seen = set()
    for f in A.flags(res):
        if (f["g"], f["band"]) in seen: continue
        seen.add((f["g"], f["band"])); F.append(f)
    F = F[a.start:a.start + a.nflags]
    if a.groups:                                                       # amendment 6: combined groups, given sizes
        F = []; os.makedirs(a.out, exist_ok=True); cache = {}
        for s in [float(x) for x in a.sizes.split(",")]:
            g = CV.band(s)
            if g not in cache: cache[g] = FO.load(M, a.rdir, g)[0]
            pred = cache[g]; ia = C.assign(a.snap, inc[s]); gs = a.groups.split(",")
            G = [w for w in km.W if C.gname(w) in gs]; b = dict(ia)
            for w in G: b[w] = BL.shift(M, b, w, -1)
            bb = BL.balance(pred, M, b, set(G), s)
            if bb is None: print(f"{s:g}: no balance"); continue
            name = f"anom-lateffndown-{s:g}"
            d = {"size": s, "mode": "decide", "assign": bb, "note": f"#193 anom | combined lead {a.groups} (fit.py overvalues late ffn_down, "
                 f"all bands, both halves): one step fewer beats the incumbent | code {a.commit}"}
            why = C.check_file(M, d)
            print(f"{name}: {len(G)} moved, {sum(bb[w] != ia[w] for w in km.W)} changed, fit.py log r {math.log(pred.kl(bb) / pred.kl(ia)):+.4f} {why or 'OK'}")
            if why is None: json.dump(d, open(f"{a.out}/{name}.json", "w"))
        return
    cnt = collections.Counter(json.loads(l)["size"] for l in open(a.pairs))
    os.makedirs(a.out, exist_ok=True); cache = {}; plan = []
    for k, f in enumerate(F):
        sizes = sorted([s for s in C.SIZES if C.BAND[s] == f["band"]], key=lambda s: -cnt[s])[:2 if k < a.two else 1]
        for s in sizes:
            g = CV.band(s)
            if g not in cache: cache[g] = FO.load(M, a.rdir, g)[0]
            pred = cache[g]; ia = C.assign(a.snap, inc[s])
            G = [w for w in km.W if C.gname(w) == f["g"]]
            b = dict(ia)
            nst = 1
            for nst in range(1, 5 if a.multi else 2):                  # amendment 6: <= 4 steps for small groups
                moved_bpw = abs(M.bytes({**ia, **{w: BL.shift(M, ia, w, nst * f["dir"]) for w in G}}) - M.bytes(ia)) * 8 / M.P
                if moved_bpw >= a.min_bits: break
            if moved_bpw < a.min_bits: print(f"{f['g']} {f['band']}: {nst} step(s) move {moved_bpw:.5f} bpw < {a.min_bits}: skipped"); break
            f = dict(f, steps=nst)
            for w in G: b[w] = BL.shift(M, b, w, f.get("steps", 1) * f["dir"])
            moved = sum(b[w] != ia[w] for w in G)
            bb = BL.balance(pred, M, b, set(G), s)
            if bb is None: print(f"{f['g']} {f['band']} {s:g}: no balance"); continue
            pm = math.log(pred.kl(bb) / pred.kl(ia))
            name = f"anom-{f['g'].replace('@', 'd')}-{s:g}"
            d = {"size": s, "mode": "decide", "assign": bb,
                 "note": f"#193 anom | flag {f['g']} {f['band']} (version {f['ver']}, beta {f['beta']:+.2f}, |z| {f['z']:.1f}): "
                         f"{'more' if f['dir'] > 0 else 'fewer'} bits in {f['g']} beat the incumbent | code {a.commit}"}
            why = C.check_file(M, d)
            print(f"{name}: {len(G)} tensors in group, {moved} moved {f['dir']:+d} step, {sum(bb[w] != ia[w] for w in km.W)} changed in all, "
                  f"bpw {M.bpw(bb):.4f}, fit.py log r {pm:+.4f}, {why or 'OK'}")
            if why is None: json.dump(d, open(f"{a.out}/{name}.json", "w")); plan.append(name)
    print(plan)


if __name__ == "__main__":
    main()
