# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""own193.py OUTDIR [--sizes ...] [--allocs xplant,pavg,smooth,spread,cap,floor,mom] [--commit C] [--run RUN]
RULES-193 amendment 7: model-free allocators on the LIVE state (read-only): the size's incumbent, pool, neighbours'
incumbents and incumbent chain -> a target K per trunk tensor -> cons193.round_to (own family, global shift, top-up).
Head / embedding keep the incumbent's options. Later amendments add allocators here (see ALLOCS)."""
import argparse, hashlib, json, math, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import common193 as C  # noqa: E402
import cons193 as CS  # noqa: E402
import menu as MN  # noqa: E402

RUN = "/workspace/rad-local/workers/lead-srch/run"
HYP = {"xplant": "a neighbouring size's incumbent shape, shifted to this size, beats the size's own incumbent",
       "pavg": "the per-tensor mean of the pool members cancels each member's noise moves",
       "smooth": "the optimum is smooth in depth: the incumbent smoothed over layers within type beats it",
       "spread": "the allocation's overall contrast is off: K scaled about the mean beats the incumbent",
       "cap": "the top tensors sit on the flat part of the error curve: K clipped at the 90th percentile beats the incumbent",
       "floor": "the lowest tensors sit on the steep part: K raised to the 10th percentile beats the incumbent",
       "dxp": "a step that confirmed at a neighbouring size (its latest incumbent change) also helps this size",
       "mom": "accepted moves keep pointing the same way: one extrapolated step along the incumbent chain beats the incumbent"}


class Live:
    def __init__(self, run=RUN):
        self.run = run; conf = json.load(open(f"{run}/driver.json")); self.M = MN.Menu.load(conf["menu"]); self.km = C.KM(self.M)
        self.st = [json.loads(l) for l in open(f"{run}/drv/state.jsonl")]
        self.chain, self.pool = {}, {}
        for e in self.st:
            if e.get("ev") == "incumbent": self.chain.setdefault(float(e["size"]), []).append(e["tag"])
            if e.get("ev") == "pool": self.pool[float(e["size"])] = e["members"]
        self.inc = {s: c[-1] for s, c in self.chain.items()}
        self.W = self.km.W; self.trunk = [w for w in self.W if w not in (MN.HEAD, MN.EMB)]
        nper = {}
        for t in self.M.T:
            o = sorted(t["opts"], key=lambda x: x["bytes"]); nper[t["w"]] = (o[-1]["bytes"] - o[0]["bytes"]) / max(o[-1]["K"] - o[0]["K"], 1e-9)
        self.n = nper
        self.known = None

    def asg(self, tag): return json.load(open(f"{self.run}/drv/assign/{tag}.json"))

    def K(self, a): return {w: self.km.Kof[w][a[w]] for w in self.W}

    def s190a(self):
        """tag -> whole s190a kld (allowed slice), via the assignment hash"""
        res = {}; sub = {}
        for e in self.st:
            if e.get("ev") == "submit": sub[e["tag"]] = e
            if e.get("ev") == "result":
                r = e["r"]
                if r.get("slice") == "s190a" and not r.get("stopped") and r.get("kld"): res[e["tag"]] = r["kld"]
        hk = {}
        for t, k in res.items():
            if t in sub: hk.setdefault(sub[t]["hash"], k)
        return lambda tag: hk.get(sub[tag]["hash"]) if tag in sub else None

    def distinct_chain(self, s):
        out, seen = [], set()
        for t in reversed(self.chain[s]):
            h = hashlib.sha1(json.dumps(self.asg(t), sort_keys=True).encode()).hexdigest()
            if h not in seen: seen.add(h); out.append(t)
        return out          # newest first

    def round(self, tk, s, ia):
        km, M = self.km, self.M
        opts = {w: km.fam_opts(w, km.fam(ia[w])) for w in self.W}
        for w in (MN.HEAD, MN.EMB):
            opts[w] = [o for o in opts[w] if o[1] == ia[w]]
        tk = dict(tk)
        for w in (MN.HEAD, MN.EMB): tk[w] = km.Kof[w][ia[w]]
        return CS.round_to(km, opts, tk, M, s)[0]

    def is_known(self, a):
        if self.known is None:
            import glob
            self.known = set()
            for p in glob.glob(f"{self.run}/drv/assign/*.json"):
                self.known.add(hashlib.sha1(json.dumps(json.load(open(p)), sort_keys=True).encode()).hexdigest())
        return hashlib.sha1(json.dumps(a, sort_keys=True).encode()).hexdigest() in self.known


def layer_type(w):
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", w); return int(m.group(1)), m.group(2)


def smooth(L, K, kern):
    by = {}
    for w in L.trunk:
        l, t = layer_type(w); by.setdefault(t, {})[l] = w
    out = dict(K); h = len(kern) // 2
    for t, d in by.items():
        for l, w in d.items():
            num = den = 0.0
            for j, c in enumerate(kern):
                ww = d.get(l + j - h)
                if ww is not None: num += c * K[ww]; den += c
            out[w] = num / den
    return out


def targets(L, alloc, s, sizes):
    """[(variant, target K dict)] for one allocator at one size"""
    ia = L.asg(L.inc[s]); K = L.K(ia); out = []
    kv = np.array([K[w] for w in L.trunk]); nv = np.array([L.n[w] for w in L.trunk])
    if alloc == "xplant":
        i = sizes.index(s); lo = sizes[i - 1] if i > 0 else None; hi = sizes[i + 1] if i + 1 < len(sizes) else None
        Kl = L.K(L.asg(L.inc[lo])) if lo else None; Kh = L.K(L.asg(L.inc[hi])) if hi else None
        if Kl: out.append(("lo", Kl))
        if Kh: out.append(("hi", Kh))
        if Kl and Kh: out.append(("mid", {w: (Kl[w] - lo + Kh[w] - hi) / 2 + s for w in L.W}))
    elif alloc == "pavg":
        f = L.s190a(); mem = L.pool.get(s, [L.inc[s]]); ks = [f(m) for m in mem]
        if len(mem) < 2: return []
        wt = [math.exp(-math.log(k / min(ks)) / 0.01) for k in ks] if all(ks) else [1.0] * len(mem)
        Ks = [L.K(L.asg(m)) for m in mem]
        out.append(("", {w: sum(a * k[w] for a, k in zip(wt, Ks)) / sum(wt) for w in L.W}))
    elif alloc == "smooth":
        out.append(("", smooth(L, K, [1, 2, 1]))); out.append(("w5", smooth(L, K, [1, 2, 3, 2, 1])))
    elif alloc == "spread":
        m = float((kv * nv).sum() / nv.sum())
        for lam in (0.85, 1.15): out.append((f"l{lam:g}", {w: (m + lam * (K[w] - m)) if w in L.n and w in set(L.trunk) else K[w] for w in L.W}))
    elif alloc == "cap":
        q = float(np.percentile(kv, 90)); out.append(("", {w: min(K[w], q) for w in L.W}))
    elif alloc == "floor":
        q = float(np.percentile(kv, 10)); out.append(("", {w: max(K[w], q) for w in L.W}))
    elif alloc == "mom":
        ch = L.distinct_chain(s)
        for var, k, al in (("", 3, 1.0), ("k6", 6, 0.5)):
            if len(ch) <= 1: continue
            Kb = L.K(L.asg(ch[min(k, len(ch) - 1)])); out.append((var, {w: K[w] + al * (K[w] - Kb[w]) for w in L.W}))
    elif alloc == "dxp":                                        # amendment 9
        i = sizes.index(s); ds = {}
        for var, nb in (("lo", sizes[i - 1] if i > 0 else None), ("hi", sizes[i + 1] if i + 1 < len(sizes) else None)):
            ch = L.distinct_chain(nb) if nb in L.chain else []
            if len(ch) < 2: continue
            Kn, Kp = L.K(L.asg(ch[0])), L.K(L.asg(ch[1])); ds[var] = {w: Kn[w] - Kp[w] for w in L.W}
            out.append((var, {w: K[w] + ds[var][w] for w in L.W}))
        if len(ds) == 2: out.append(("both", {w: K[w] + ds["lo"][w] + ds["hi"][w] for w in L.W}))
    else:
        raise SystemExit(f"unknown allocator {alloc}")
    return out


def emit(L, alloc, s, var, a, out, commit, hyp=None, extra=""):
    ia = L.asg(L.inc[s]); M = L.M
    nd = sum(a[w] != ia[w] for w in L.W); dk = float(np.mean([abs(L.km.Kof[w][a[w]] - L.km.Kof[w][ia[w]]) for w in L.W]))
    d = {"size": s, "mode": "decide", "assign": a,
         "note": f"#193 {alloc} | {hyp or HYP[alloc]}{' (' + var + ')' if var else ''}{extra}; from {L.inc[s]} | code {commit}"}
    why = C.check_file(M, d)
    if why is None and nd < 4: why = "under 4 tensors changed"
    if why is None and L.is_known(a): why = "already in the run"
    name = f"{alloc}-{s:g}{'-' + var if var else ''}.json"
    print(f"{name}: bpw {M.bpw(a):.4f}, {nd} changed, mean|dK| {dk:.3f} {why or 'OK'}", flush=True)
    if why is None:
        os.makedirs(out, exist_ok=True); json.dump(d, open(f"{out}/{name}", "w")); return f"{out}/{name}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--sizes", default="2,2.5,3,3.5,4,4.5,5,6")
    ap.add_argument("--allocs", default="xplant,pavg,smooth,spread,cap,floor,mom"); ap.add_argument("--commit", default="?")
    ap.add_argument("--only-var")
    a = ap.parse_args(); L = Live()
    for s in [float(x) for x in a.sizes.split(",")]:
        for al in a.allocs.split(","):
            for var, tk in targets(L, al, s, C.SIZES):
                if a.only_var is not None and var not in a.only_var.split(","): continue
                asg = L.round(tk, s, L.asg(L.inc[s])); emit(L, al, s, var, asg, a.out, a.commit)


if __name__ == "__main__":
    main()
