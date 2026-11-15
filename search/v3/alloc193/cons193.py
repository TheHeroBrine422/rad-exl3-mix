# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cons193.py SNAP OUTDIR [--sizes 2,2.5,...] [--min-diff 10] [--commit C] -- RULES-193 §2: refreshed consensus.

consensus192's local-linear per-tensor K(size) curve and rounding, with sources = the s186e re-selection models within
3 % of each size's best (weights as #192) + each size's current pool members scored whole on s190a (weight
e^(-log(kld / best member) / 0.01)). Allowed slices only (s186e, s190a). Library: sources(), target(), round_to().
Writes OUTDIR/cons-<size>.json (decide) for sizes whose rounding differs from the incumbent in >= --min-diff tensors."""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common193 as C  # noqa: E402

FULL = {"s186e": None, "s190a": 378300}


def full_klds(snap, slc):
    """tag -> whole-slice kld on slc (results.jsonl; positions = the slice's whole count, the largest seen)"""
    rows = [json.loads(l) for l in open(f"{snap}/results.jsonl")]
    rows = [r for r in rows if r.get("slice") == slc and r.get("kld") and C.ok_slice(slc)]
    full = max(r["positions"] for r in rows)
    return {r["tag"]: r["kld"] for r in rows if r["positions"] == full}


def sources(snap, st, within=0.03, top=5, use_pool=True):
    """[(size, weight, assignment, label)]"""
    out = []
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit" and e.get("kind") == "reselect" and e.get("slice") == "s186e"}
    ke = full_klds(snap, "s186e"); by = {}
    for t, e in sub.items():
        if t in ke: by.setdefault(float(e["size"]), []).append((ke[t], e["of"]))
    for s, L in sorted(by.items()):
        L.sort(); best = L[0][0]
        for k, of in [(k, of) for k, of in L if math.log(k / best) <= within][:top]:
            out.append((s, math.exp(-math.log(k / best) / 0.01), C.assign(snap, of), f"s186e:{of}"))
    if use_pool:
        ka = full_klds(snap, "s190a"); pools = {}
        hs = {}
        for e in st:
            if e.get("ev") == "pool": pools[float(e["size"])] = e["members"]
            if e.get("ev") == "submit" and e.get("slice") == "s190a": hs.setdefault(e["hash"], []).append(e["tag"])
        sh = {e["tag"]: e["hash"] for e in st if e.get("ev") == "submit"}
        for s, mem in sorted(pools.items()):
            if s not in C.SIZES: continue
            got = []
            for m in mem:
                k = ka.get(m) or next((ka[t] for t in hs.get(sh.get(m), []) if t in ka), None)
                if k: got.append((k, m))
            if not got: continue
            best = min(k for k, _ in got)
            for k, m in got: out.append((s, math.exp(-math.log(k / best) / 0.01), C.assign(snap, m), f"s190a:{m}"))
    return out


class Curve:
    """consensus192's per-tensor weighted local-linear fit of K on size"""

    def __init__(self, km, src, bw=1.0):
        self.km, self.src, self.bw = km, src, bw
        f0 = {w: km.fam(src[0][2][w]) for w in km.W}
        self.opts = {w: km.fam_opts(w, f0[w]) for w in km.W}

    def target(self, w, s0):
        sw = sx = sy = sxx = sxy = 0.0
        for s, wt, asg, _ in self.src:
            k = self.km.Kof[w].get(asg[w])
            if k is None or self.km.fam(asg[w]) != self.km.fam(self.opts[w][0][1]): continue
            g = wt * math.exp(-((s - s0) / self.bw) ** 2); x = s - s0
            sw += g; sx += g * x; sy += g * k; sxx += g * x * x; sxy += g * x * k
        d = sw * sxx - sx * sx
        return sy / sw if abs(d) < 1e-12 else (sy * sxx - sx * sxy) / d

    def targets(self, s0): return {w: self.target(w, s0) for w in self.km.W}


def round_to(km, opts, tk, M, s0, mode="decide", tol=0.02):
    """consensus192's rounding: one global shift (bisection) of nearest-option rounding into the window, then top-up
    (closest to the next step first)"""
    hi = s0 * M.P / 8
    rnd = lambda k, w: min(opts[w], key=lambda o: abs(o[0] - k))
    fill = lambda cc: {w: rnd(tk[w] + cc, w)[1] for w in km.W}
    l_, h_ = -3.0, 3.0
    for _ in range(60):
        m = (l_ + h_) / 2
        if M.bytes(fill(m)) <= hi: l_ = m
        else: h_ = m
    asg = fill(l_); b = M.bytes(asg)
    while True:
        best = None
        for w in km.W:
            names = [o[1] for o in opts[w]]; i = names.index(asg[w])
            if i + 1 >= len(opts[w]): continue
            nk, no, nb = opts[w][i + 1]; db = nb - opts[w][i][2]
            if b + db > hi: continue
            gap = nk - (tk[w] + l_)
            if best is None or gap < best[0]: best = (gap, w, no, db)
        if best is None: break
        asg[best[1]] = best[2]; b += best[3]
    return asg, l_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap"); ap.add_argument("out"); ap.add_argument("--sizes", default="2,2.5,3,3.5,4,4.5,5,6")
    ap.add_argument("--min-diff", type=int, default=10); ap.add_argument("--commit", default="?")
    a = ap.parse_args()
    st = C.load_state(a.snap); M = C.load_menu(a.snap); km = C.KM(M)
    src = sources(a.snap, st)
    for s in sorted({x[0] for x in src}): print(f"size {s:g}: " + ", ".join(f"{l} w{wt:.2f}" for ss, wt, _, l in src if ss == s))
    cv = Curve(km, src); inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    os.makedirs(a.out, exist_ok=True)
    for s0 in [float(x) for x in a.sizes.split(",")]:
        asg, sh = round_to(km, cv.opts, cv.targets(s0), M, s0)
        ia = C.assign(a.snap, inc[s0]); nd = sum(asg[w] != ia[w] for w in km.W)
        dk = sum(abs(km.Kof[w][asg[w]] - km.Kof[w][ia[w]]) for w in km.W) / len(km.W)
        d = {"size": s0, "mode": "decide", "assign": asg,
             "note": f"#193 cons | refreshed consensus (s186e top 3% + current pools on s190a) beats the incumbent | code {a.commit}"}
        why = C.check_file(M, d)
        keep = nd >= a.min_diff and why is None
        print(f"{s0:g}: bpw {M.bpw(asg):.4f} shift {sh:+.3f} vs inc {inc[s0]}: {nd} tensors changed, mean|dK| {dk:.3f} "
              f"{'WRITE' if keep else 'skip'} {why or ''}")
        if keep: json.dump(d, open(f"{a.out}/cons-{s0:g}.json", "w"))


if __name__ == "__main__":
    main()
