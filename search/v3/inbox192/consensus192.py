#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""consensus192.py DRIVER.json OUTDIR [--decide 2,2.5,...] [--probe 2.25,2.75,...] [--within 0.03] [--top 5] [--bw 1.0]
-- #192 consensus allocations: pool the sizes' good models into one smooth per-tensor curve K_t(size), then round it
to the menu at any size (user 11-15: "use the overall optimal tables across all of them").

Source models: the #189-02 re-selection on s186e (every model scored whole on one fresh slice): per size the best and
every model within --within (log ratio) of it, at most --top. Weight: exp(-log(kld / best) / 0.01) (a model 1 % worse
counts e^-1 as much). Per tensor (336 trunk + head + embedding), at a target size s0: weighted local linear fit of K on
size over all sources, kernel exp(-((s - s0) / bw)^2). Rounding: one global shift c (bisection) so that each tensor's
nearest menu option of its own family to K_t(s0) + c fills the byte window [s0 - size_tol, s0]; then the remaining
bytes go to the tensors closest to their next step up. Model-free: no KLD fit, only the measured allocations.
Writes OUTDIR/<size>.json in the inbox format (mode decide for --decide sizes, probe with a t183 read for --probe)."""
import argparse, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import menu as MN  # noqa: E402

T183 = "/data/q27/w131/slices/golden-t183"


def fam(o):
    """option family: trunk mad levels, mad head u<K>, mul1 head u<K>m, trellis embedding t<K>, t<K>m, bf16"""
    if o.startswith("q27v3"): return "mad"
    if o == "bf16": return "bf16"
    return o[0] + ("m" if o.endswith("m") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conf"); ap.add_argument("out")
    ap.add_argument("--decide", default="2,2.5,3,3.5,4,4.5,5,6"); ap.add_argument("--probe", default="2.25,2.75,3.25,3.75,4.25,4.75,5.5")
    ap.add_argument("--within", type=float, default=0.03); ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bw", type=float, default=1.0); ap.add_argument("--kmax", type=float, default=99.0, help="cap trunk K (w8a8 saturates above ~K7)"); ap.add_argument("--slice", default="s186e")
    ap.add_argument("--whole", action="store_true", help="(11-16) sources = every whole (not early-stopped) result on --slice in the driver state, any kind")
    a = ap.parse_args()
    c = json.load(open(a.conf)); M = MN.Menu.load(c["menu"]); R = c["root"]; tol = float(c.get("size_tol", 0.02))
    st = [json.loads(l) for l in open(f"{R}/state.jsonl")]
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit" and e.get("kind") == "reselect" and e.get("slice") == a.slice}
    kl = {}
    for l in open(c["results"]):
        r = json.loads(l)
        if r.get("tag") in sub and r.get("kld") and r.get("slice", a.slice) == a.slice: kl[r["tag"]] = r["kld"]
    by = {}
    if a.whole:                                                 # (11-16) every whole result on the slice, any kind
        sub2 = {e["tag"]: e for e in st if e.get("ev") == "submit"}
        for e in st:
            if e.get("ev") != "result": continue
            r = e["r"]; t = e["tag"]
            if r.get("slice") != a.slice or r.get("stopped") or not r.get("kld") or t not in sub2: continue
            if not os.path.exists(f"{R}/assign/{t}.json"): continue
            by.setdefault(float(sub2[t]["size"]), []).append((r["kld"], t))
    else:
      for t, e in sub.items():
        if t in kl: by.setdefault(float(e["size"]), []).append((kl[t], e["of"]))
    src = []                                                    # (size, weight, assignment)
    for s, L in sorted(by.items()):
        L.sort(); best = L[0][0]
        keep = [(k, of) for k, of in L if math.log(k / best) <= a.within][:a.top]
        for k, of in keep:
            src.append((s, math.exp(-math.log(k / best) / 0.01), json.load(open(f"{R}/assign/{of}.json"))))
        print(f"size {s:g}: {len(L)} scored, {len(keep)} used: " + ", ".join(f"{of.split('-', 2)[2]} {math.log(k / best):+.3f}" for k, of in keep))
    W = list(src[0][2]); opts = {}
    for w in W:
        f0 = fam(src[0][2][w])
        opts[w] = sorted(((o["K"], o["o"], o["bytes"]) for o in M.d_opts(w)) if hasattr(M, "d_opts") else
                         ((o["K"], o["o"], o["bytes"]) for o in next(t for t in M.T if t["w"] == w)["opts"] if fam(o["o"]) == f0))
    Kof = {w: {o: k for k, o, _ in opts[w]} for w in W}

    def target(w, s0):
        sw = sx = sy = sxx = sxy = 0.0
        for s, wt, asg in src:
            k = Kof[w].get(asg[w])
            if k is None: continue                              # a source in another family (head / embedding)
            g = wt * math.exp(-((s - s0) / a.bw) ** 2); x = s - s0
            sw += g; sx += g * x; sy += g * k; sxx += g * x * x; sxy += g * x * k
        d = sw * sxx - sx * sx
        return sy / sw if abs(d) < 1e-12 else (sy * sxx - sx * sxy) / d     # local linear intercept at s0

    def rnd(k, w):
        cand = [o for o in opts[w] if not (w.startswith("blk.") and o[0] > a.kmax)] or opts[w]
        return min(cand, key=lambda o: abs(o[0] - k))

    os.makedirs(a.out, exist_ok=True)
    for mode, lst in (("decide", a.decide), ("probe", a.probe)):
        for s0 in [float(x) for x in lst.split(",") if x]:
            tk = {w: target(w, s0) for w in W}
            hi = s0 * M.P / 8; lo = (s0 - tol) * M.P / 8
            fill = lambda cc: {w: rnd(tk[w] + cc, w)[1] for w in W}
            l_, h_ = -3.0, 3.0
            for _ in range(60):
                m = (l_ + h_) / 2
                if M.bytes(fill(m)) <= hi: l_ = m
                else: h_ = m
            asg = fill(l_); b = M.bytes(asg)
            while True:                                         # top up: closest to the next step first
                best = None
                for w in W:
                    i = [o[1] for o in opts[w]].index(asg[w])
                    if i + 1 >= len(opts[w]): continue
                    nk, no, nb = opts[w][i + 1]
                    db = nb - opts[w][i][2]
                    if w.startswith("blk.") and nk > a.kmax: continue
                    if b + db > hi: continue
                    gap = nk - (tk[w] + l_)
                    if best is None or gap < best[0]: best = (gap, w, no, db)
                if best is None: break
                asg[best[1]] = best[2]; b += best[3]
            bpw = M.bpw(asg)
            ok = lo <= b <= hi if mode == "decide" else abs(bpw - s0) <= 0.02
            d = {"size": s0, "assign": asg, "mode": mode, "note": f"#192 consensus (s186e top within {a.within}, bw {a.bw}), shift {l_:+.3f}, bpw {bpw:.4f}"}
            if mode == "probe": d["extra_slices"] = [{"name": "t183", "golden": T183}]
            json.dump(d, open(f"{a.out}/{s0:g}-{mode}.json", "w"))
            print(f"{mode} {s0:g}: bpw {bpw:.4f} {'OK' if ok else 'OUT OF WINDOW'}, shift {l_:+.3f}, head {asg.get('output.weight')}, emb {asg.get('token_embd.weight')}")


if __name__ == "__main__":
    main()
