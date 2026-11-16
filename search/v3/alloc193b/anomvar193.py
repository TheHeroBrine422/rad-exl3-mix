# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""anomvar193.py SNAP RDIR ANOM.json PAIRS.jsonl CONFIRMED.json OUTDIR [--nflags 12] [--commit C] [--no-base]
RULES-193 amendment 13 (w181): `anom` tests from the cut-A flag table + follow-up variants of the confirmed flags.

  base : every flag (|z| >= 3, top --nflags distinct group x band) at every size of its band, one own-family menu step
         in the flag's direction (amendment 11), name anom-<group>-<size>
  CONFIRMED.json = [{"groups": ["ffn_down@7"], "band": "b3", "dir": -1, "src": "..."}]: per entry
    s2  : two steps                              nbm / nbp : the type's depth band d-1 / d+1, one step
    nb2 : the flagged band + both neighbours     xb : the flagged group(s), one step, at the adjacent size bands' sizes
    rev : one step against the flag (control), at the band's size with most paired rows
SNAP = a fresh state snapshot (incumbents); RDIR = the cut-A refit (balancing: bal193). Writes OUTDIR/*.json and
OUTDIR/meta.jsonl (name -> groups, dir, steps, var, flag) for the per-variant report."""
import argparse, collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import anom193 as A  # noqa: E402
import bal193 as BL  # noqa: E402
import fitopt193 as FO  # noqa: E402
import curv193 as CV  # noqa: E402

ADJ = {"b1": ["b2"], "b2": ["b1", "b3"], "b3": ["b2"]}


def nb(g, d):
    t, k = g.split("@"); k = int(k) + d
    return f"{t}@{k}" if 0 <= k <= 7 else None


def main():
    ap = argparse.ArgumentParser()
    for x in ("snap", "rdir", "anom", "pairs", "confirmed", "out"): ap.add_argument(x)
    ap.add_argument("--nflags", type=int, default=12); ap.add_argument("--commit", default="?"); ap.add_argument("--no-base", action="store_true")
    a = ap.parse_args()
    M = C.load_menu(a.snap); km = C.KM(M); st = C.load_state(a.snap); inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    cnt = collections.Counter(json.loads(l)["size"] for l in open(a.pairs))
    os.makedirs(a.out, exist_ok=True); cache = {}; meta = open(f"{a.out}/meta.jsonl", "w")

    def make(name, s, groups, d, steps, var, flag, why_txt):
        g = CV.band(s)
        if g not in cache: cache[g] = FO.load(M, a.rdir, g)[0]
        pred = cache[g]; ia = C.assign(a.snap, inc[s]); G = [w for w in km.W if C.gname(w) in groups]; b = dict(ia)
        for w in G: b[w] = BL.shift(M, b, w, steps * (d[C.gname(w)] if isinstance(d, dict) else d))
        moved = sum(b[w] != ia[w] for w in G)
        if not moved: print(f"{name}: nothing to move"); return
        bb = BL.balance(pred, M, b, set(G), s)
        if bb is None: print(f"{name}: no balance"); return
        f = {"size": s, "mode": "decide", "assign": bb, "note": f"#193 anom | {why_txt} | code {a.commit}"}
        why = C.check_file(M, f)
        print(f"{name}: {moved}/{len(G)} moved {steps} step(s), {sum(bb[w] != ia[w] for w in km.W)} changed, bpw {M.bpw(bb):.4f}, "
              f"fit.py log r {math.log(pred.kl(bb) / pred.kl(ia)):+.4f} {why or 'OK'}", flush=True)
        if why is None:
            json.dump(f, open(f"{a.out}/{name}.json", "w"))
            meta.write(json.dumps({"name": name, "size": s, "groups": groups, "dir": d if not isinstance(d, dict) else 0, "steps": steps, "var": var, "flag": flag, "from": inc[s]}) + "\n"); meta.flush()

    def tok(gs):
        if gs == ["ffn_down@6", "ffn_down@7"]: return "lateffndown"
        if len(gs) > 1 and len({x.split("@")[0] for x in gs}) == 1: return gs[0].split("@")[0] + "d" + "".join(x.split("@")[1] for x in gs)
        return "+".join(x.replace("@", "d") for x in gs)
    bsz = lambda b: sorted([s for s in C.SIZES if C.BAND[s] == b], key=lambda s: -cnt[s])
    if not a.no_base:
        res = json.load(open(a.anom)); seen = set(); F = []
        for f in A.flags(res):
            if (f["g"], f["band"]) in seen: continue
            seen.add((f["g"], f["band"])); F.append(f)
        for f in F[:a.nflags]:
            for s in bsz(f["band"]):
                make(f"anom-{tok([f['g']])}-{s:g}", s, [f["g"]], f["dir"], 1, "base", f"{f['g']}|{f['band']}",
                     f"flag {f['g']} {f['band']} (cut A, version {f['ver']}, beta {f['beta']:+.2f}, |z| {f['z']:.1f}): "
                     f"{'more' if f['dir'] > 0 else 'fewer'} bits in {f['g']} beat the incumbent")
    for c in json.load(open(a.confirmed)):
        gs, b, d = c["groups"], c["band"], c["dir"]; fl = f"{'+'.join(gs)}|{b}"; t = tok(gs); way = "more" if d > 0 else "fewer"
        for s in bsz(b):
            if c.get("s2ok"): make(f"anom-{t}-{s:g}-s3", s, gs, d, 3, "s3", fl, f"follow-up of the confirmed two-step {fl}: three steps {way} (still short of the optimum)")
            make(f"anom-{t}-{s:g}-s2", s, gs, d, 2, "s2", fl, f"follow-up of the confirmed flag {fl}: two steps {way} (the confirmed step was short of the optimum)")
            if "@" not in gs[0]: continue                                 # head / emb: no depth neighbours
            lo = [x for x in (nb(gs[0], -1),) if x and x not in gs]; hi = [x for x in (nb(gs[-1], +1),) if x and x not in gs]
            if lo: make(f"anom-{t}-{s:g}-nbm", s, lo, d, 1, "nbm", fl, f"follow-up of the confirmed flag {fl}: the neighbouring depth band {lo[0]} shares the mis-valuation ({way} bits)")
            if hi: make(f"anom-{t}-{s:g}-nbp", s, hi, d, 1, "nbp", fl, f"follow-up of the confirmed flag {fl}: the neighbouring depth band {hi[0]} shares the mis-valuation ({way} bits)")
            if lo or hi: make(f"anom-{t}-{s:g}-nb2", s, lo + gs + hi, d, 1, "nb2", fl, f"follow-up of the confirmed flag {fl}: the flagged band and its neighbours together, one step {way}")
        for b2 in ADJ[b]:
            if any(x["groups"] == gs and x["band"] == b2 for x in json.load(open(a.confirmed))): continue
            for s in bsz(b2):
                make(f"anom-{t}-{s:g}-xb", s, gs, d, 1, "xb", fl, f"follow-up of the confirmed flag {fl}: the effect is not band-specific ({way} bits in {'+'.join(gs)} at {s:g})")
        s = bsz(b)[0]
        make(f"anom-{t}-{s:g}-rev", s, gs, -d, 1, "rev", fl, f"control for the confirmed flag {fl}: one step AGAINST the flag should lose")
    Cf = json.load(open(a.confirmed))                                  # amendment 16: cmb
    for s in C.SIZES:
        dirs = {}
        for c in Cf:
            if c["band"] == C.BAND[s]:
                for g in c["groups"]: dirs.setdefault(g, c["dir"])
        if len(dirs) >= 2:
            make(f"anom-cmb-{s:g}", s, sorted(dirs), dirs, 1, "cmb", f"cmb|{C.BAND[s]}",
                 f"confirmed effects add: every confirmed flag of {C.BAND[s]} one step at once ({', '.join(g + ('+' if v > 0 else '-') for g, v in sorted(dirs.items()))})")


if __name__ == "__main__":
    main()
