#!/usr/bin/env python3
"""consensus197.py CONF OUTDIR [--probe LABELS] [--force TENSOR=OPTION] [--card-total BYTES] [--within 0.03] [--top 5]
   [--bw 1.0] [--slice s186e] [--tol 0.02] -- #197 (w185): copy of search/v3/inbox192/consensus192.py (the lead's #192
consensus allocations) with two additions for #197:
  --force TENSOR=OPTION   pin one tensor to an option (e.g. token_embd.weight=bf16: the embedding off the card); the
                          pinned tensor's bytes count in the probe's total, but the rounding works on the others only.
  --card-total BYTES      the budget the rest must fill: trunk + head + fixed bytes = BYTES - (pinned bytes) + fixed ...
                          concretely the probe's total container bytes = BYTES + the pinned tensor's bytes, so the
                          card's trunk + head + fixed bytes equal BYTES - (no pinned bytes on the card). Pass the
                          total container bytes of the size being matched (S): then the card holds what S's card held
                          plus S's embedding (the re-allocation of #197).
  --probe LABELS          probe labels (comma list); with --card-total the label is recomputed from the actual bytes.
Everything else is consensus192's: sources = s186e re-selection models within --within of each size's best (<= --top,
weight exp(-dlog/0.01)), per-tensor weighted local-linear fit in size (bw), one global shift + top-up to the window.
Writes OUTDIR/<label>-probe.json in the inbox format (mode probe, extra_slices t183). Run with `python3 -I`? no: plain
python3 (the menu module is imported from search/v3 by path)."""
import argparse, json, math, os, sys

sys.path.insert(0, "/workspace/rad-exl3-mix/search/v3")
import menu as MN  # noqa: E402

T183 = "/data/q27/w131/slices/golden-t183"


def fam(o):
    if o.startswith("q27v3"): return "mad"
    if o == "bf16": return "bf16"
    return o[0] + ("m" if o.endswith("m") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conf"); ap.add_argument("out")
    ap.add_argument("--probe", default="5.5")
    ap.add_argument("--force", action="append", default=[])
    ap.add_argument("--card-total", type=int, default=None)
    ap.add_argument("--within", type=float, default=0.03); ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bw", type=float, default=1.0); ap.add_argument("--slice", default="s186e")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=99.0)
    a = ap.parse_args()
    c = json.load(open(a.conf)); M = MN.Menu.load(c["menu"]); R = c["root"]
    tol = float(c.get("size_tol", 0.02)) if a.tol is None else a.tol
    force = dict(f.split("=", 1) for f in a.force)
    st = [json.loads(l) for l in open(f"{R}/state.jsonl")]
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit" and e.get("kind") == "reselect" and e.get("slice") == a.slice}
    kl = {}
    for l in open(c["results"]):
        r = json.loads(l)
        if r.get("tag") in sub and r.get("kld") and r.get("slice", a.slice) == a.slice: kl[r["tag"]] = r["kld"]
    by = {}
    for t, e in sub.items():
        if t in kl: by.setdefault(float(e["size"]), []).append((kl[t], e["of"]))
    src = []
    for s, L in sorted(by.items()):
        L.sort(); best = L[0][0]
        keep = [(k, of) for k, of in L if math.log(k / best) <= a.within][:a.top]
        for k, of in keep:
            src.append((s, math.exp(-math.log(k / best) / 0.01), json.load(open(f"{R}/assign/{of}.json"))))
        print(f"size {s:g}: {len(L)} scored, {len(keep)} used")
    W = [w for w in src[0][2] if w not in force]                 # rounded tensors
    opts = {}
    for w in W:
        f0 = fam(src[0][2][w])
        opts[w] = sorted((o["K"], o["o"], o["bytes"]) for o in next(t for t in M.T if t["w"] == w)["opts"] if fam(o["o"]) == f0)
    Kof = {w: {o: k for k, o, _ in opts[w]} for w in W}
    pinned = {}
    for w, o in force.items():
        pinned[w] = o
    fixed_bytes = M.fixed

    def target(w, s0):
        sw = sx = sy = sxx = sxy = 0.0
        for s, wt, asg in src:
            k = Kof[w].get(asg[w])
            if k is None: continue
            g = wt * math.exp(-((s - s0) / a.bw) ** 2); x = s - s0
            sw += g; sx += g * x; sy += g * k; sxx += g * x * x; sxy += g * x * k
        d = sw * sxx - sx * sx
        return sy / sw if abs(d) < 1e-12 else (sy * sxx - sx * sxy) / d

    def rnd(k, w):
        cand = [o for o in opts[w] if not (w.startswith("blk.") and o[0] > a.kmax)] or opts[w]
        return min(cand, key=lambda o: abs(o[0] - k))

    os.makedirs(a.out, exist_ok=True)
    labels = [float(x) for x in a.probe.split(",") if x]
    for s0 in labels:
        pin_bytes = sum(next(o["bytes"] for o in next(t for t in M.T if t["w"] == w)["opts"] if o["o"] == p) for w, p in pinned.items())
        if a.card_total is not None:
            # the rest (trunk + head + fixed) must hold card_total minus nothing: the probe's total = card_total + pinned
            hi = a.card_total + pin_bytes
        else:
            hi = s0 * M.P / 8
        lo = hi - tol * M.P / 8
        tk = {w: target(w, s0) for w in W}
        fill = lambda cc: {w: rnd(tk[w] + cc, w)[1] for w in W}
        l_, h_ = -3.0, 3.0
        base = {w: p for w, p in pinned.items()}
        for _ in range(60):
            m = (l_ + h_) / 2
            asg0 = fill(m); asg0.update(base)
            if M.bytes(asg0) <= hi: l_ = m
            else: h_ = m
        asg = fill(l_); asg.update(base); b = M.bytes(asg)
        while True:
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
        label = f"{s0:g}" if a.card_total is None else f"{bpw:.4f}"
        ok = lo <= b <= hi
        d = {"size": round(bpw, 6) if a.card_total is not None else s0, "assign": asg, "mode": "probe",
             "note": f"#197 consensus (s186e top within {a.within}, bw {a.bw}), label {s0:g}, bpw {bpw:.4f}, "
                     f"card_total {a.card_total}, force {force}, window {'OK' if ok else 'OUT'}",
             "extra_slices": [{"name": "t183", "golden": T183}]}
        fn = f"{a.out}/{label}-probe.json"
        json.dump(d, open(fn, "w"))
        print(f"{fn}: total {b} (window {lo}..{hi}) bpw {bpw:.4f} {'OK' if ok else 'OUT OF WINDOW'} head {asg['output.weight']} emb {asg['token_embd.weight']}")


if __name__ == "__main__":
    main()
