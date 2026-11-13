# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ds133.py -- #173 (w133): the search's measurements as one dataset for the sensitivity-model comparison.

  python3 ds133.py OUT.json.gz --state DRV/state.jsonl --assign DRV/assign --registry REG.json --menu MENU.json
                   --prior FIT129H2.obs.jsonl --w129 DATA/refit/w129 [--a8-slices s171c,..] [--until 'MM-DD HH:MM:SS']

Replays the driver's state (search/v3/driver.py: replay + on_result + add_obs, not imported: the VM copy is live) and
writes
  obs     the fit rows exactly as the driver builds them (one per (tag, label); "S|p|full" / "S|p|pre<k>" labels, the
          incumbent's anchor row on the same documents, se = paired delta-method SE vs the anchor), each with the result
          index ri at which it became available (prior rows: ri 0) and the act mode of its slice;
  events  every decided candidate (accept / reject; stale ones dropped) with the incumbent it was judged against (the
          driver's rule: stopped and audit-stopped runs vs the submit's ref, full runs vs the size's incumbent at the
          time), per part measured log r and its paired SE on the scored documents, the size offset off = b dbpw, the
          call, its result index and time;
  s129    w129's held-out swaps vs their base on s129 (per part log r + SE, from the per-document sums);
  assign  option index per menu tensor for every tag used.
Option indices refer to the menu's tensors in file order (menu.py sorts each tensor's options by bytes)."""
import argparse, gzip, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "v3"))
import seqrule as SR          # noqa: E402
import menu as MN             # noqa: E402


def paired_parts(A, B, parts, k):
    """per part (log r, SE, docs) of B vs A over the first k documents (both scored)"""
    acc = {}
    for i in range(k):
        a, b = A[i], B[i]
        if a is None or b is None: break
        acc.setdefault(parts[i], []).append((a, b))
    out = {}
    for p, v in acc.items():
        P = SR.Paired(len(v), {"kind": "full"})
        for a, b in v: P.add(a, b)
        se = P.se()
        out[p] = {"logr": P.d_pooled(), "se": se if se != SR.INF else None, "docs": len(v)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--state", required=True); ap.add_argument("--assign", required=True)
    ap.add_argument("--registry", required=True); ap.add_argument("--menu", required=True)
    ap.add_argument("--prior", required=True); ap.add_argument("--w129", required=True)
    ap.add_argument("--a8-slices", default="", help="slices scored with A8 containers (w132's switch)")
    ap.add_argument("--until", default=None, help="ignore state lines after this CST time stamp ('11-13 08:39:01')")
    a = ap.parse_args(argv)
    M = MN.Menu.load(a.menu); reg = json.load(open(a.registry))["slices"]
    a8 = set(x for x in a.a8_slices.split(",") if x)
    parts_of = lambda s: [d["part"] for d in reg[s]["docs"]] if s in reg else None
    mode_of = lambda s: "a8" if s.split("|")[0] in a8 else "f16"
    sub, res, inc, order = {}, {}, {}, []
    events, obs, assign = [], {}, {}
    pend = {}                                                           # tag -> ref used by its decision
    last_t = None

    def amap(tag):
        if tag not in assign:
            d = json.load(open(f"{a.assign}/{tag}.json"))
            assign[tag] = [M.oi[i][d[w]] for i, w in enumerate(M.W)]
        return tag

    def add_obs(t, r, ri):                                              # driver.add_obs
        s = sub.get(t)
        if s is None or s["kind"] in ("monitor", "monitor_ref") or r.get("fail") or not r.get("docs"): return
        docs, pos = r["docs"], r["pos"]; N = len(docs); k = sum(1 for x in docs if x is not None)
        if k == 0: return
        g = max(1, math.ceil(N / 20)); full = k == N; kq = N if full else (k // g) * g
        if kq < 3: return
        parts = parts_of(r["slice"]) or ["m"] * N
        pset = sorted(set(parts)) if len(set(parts)) > 1 else [None]
        ref = s.get("ref") if s.get("ref") in res else None
        rr = res[ref] if ref and ref != t else None
        for p in pset:
            ix = [i for i in range(kq) if p is None or parts[i] == p]
            if len(ix) < 3: continue
            label = f"{r['slice']}|" + (f"{p}|" if p else "") + ("full" if full else f"pre{kq}")
            kl = sum(docs[i] for i in ix) / max(sum(pos[i] for i in ix), 1)
            se = 0.0
            if rr is not None:
                P = SR.Paired(len(ix), {"kind": "full"})
                for i in ix: P.add(rr["docs"][i], docs[i])
                se = P.se() if P.se() != SR.INF else 0.5
                if all(rr["docs"][i] is not None for i in ix) and (ref, label) not in obs:     # first creation's ri
                    obs[(ref, label)] = {"tag": f"{ref}@{label}", "a": amap(ref), "label": label, "size": sub[ref]["size"],
                                         "kld": sum(rr["docs"][i] for i in ix) / max(sum(rr["pos"][i] for i in ix), 1), "se": 0.0,
                                         "ri": ri, "mode": mode_of(label), "part": p}
            obs[(t, label)] = {"tag": f"{t}@{label}", "a": amap(t), "label": label, "kld": kl, "se": se, "size": s["size"],
                               "ri": ri, "mode": mode_of(label), "part": p}

    ri = 0
    for l in open(a.state):
        try: e = json.loads(l)
        except ValueError: continue
        if a.until and e.get("t") and e["t"] > a.until + " CST": break
        last_t = e.get("t", last_t)
        ev = e["ev"]
        if ev == "submit": sub[e["tag"]] = e
        elif ev == "incumbent": inc[e["size"]] = e["tag"]
        elif ev == "result":
            t, r = e["tag"], e["r"]; res[t] = r; ri += 1; order.append(t)
            s = sub[t]; size = s["size"]
            if r.get("fail") or s["kind"] in ("monitor", "monitor_ref", "incumbent"): pend[t] = None
            elif r.get("stopped"): pend[t] = s["ref"] if s.get("ref") in res else inc.get(size)
            else: pend[t] = (inc.get(size), s.get("ref"), s.get("audit"))
            add_obs(t, r, ri)
            e["_ri"] = ri; sub[t]["_ri"] = ri; sub[t]["_rt"] = e.get("t")
        elif ev == "decision":
            t = e["tag"]
            if e["decision"] not in ("accept", "reject") or t not in res: continue
            s = sub[t]; r = res[t]; pr = pend.get(t)
            if isinstance(pr, tuple):                                    # full run: vs the incumbent; audit stopped: vs ref
                cur, sref, audit = pr
                ref = sref if (audit and sref in res and e.get("k", 0) < len(r["docs"])) else cur
            else: ref = pr
            if ref is None or ref not in res: continue
            parts = parts_of(r["slice"]) or ["m"] * len(r["docs"])
            pp = paired_parts(res[ref]["docs"], r["docs"], parts, e["k"])
            events.append({"tag": amap(t), "ref": amap(ref), "size": s["size"], "kind": s["kind"], "ri": s["_ri"], "t": e["t"],
                           "slice": r["slice"], "mode": mode_of(r["slice"]), "decision": e["decision"], "k": e["k"],
                           "n": len(r["docs"]), "stopped": bool(r.get("stopped")), "audit": bool(s.get("audit")),
                           "off": e.get("off", 0.0), "b": s.get("b"), "bpw": s["bpw"], "bpw_ref": sub[ref]["bpw"],
                           "logr": e["logr"], "z": e["z"], "parts": pp,
                           "parts_driver": {p: v["logr"] for p, v in (e.get("parts") or {}).items()}})
    # prior rows (w129's fit129, the driver's prior_obs) and the s129 held-out swaps
    prior = []
    for l in open(a.prior):
        o = json.loads(l); tg = o["tag"].split("@")[0]
        if tg not in assign: assign[tg] = [M.oi[i][o["assign"][w]] for i, w in enumerate(M.W)]
        prior.append({"tag": o["tag"], "a": tg, "label": o["label"], "kld": o["kld"], "se": o.get("se", 0.0), "size": o.get("size"),
                      "ri": 0, "mode": "f16", "part": None, "kind": o.get("kind")})
    s129 = []
    dd = f"{a.w129}/docs"
    for f in sorted(os.listdir(dd)):
        if not f.endswith("-s129.json") or f.startswith("w129-u-"): continue
        tg = f[:-len("-s129.json")]
        base = "w129-u-k2.5" if tg.startswith("w129-g2.5") else "w129-u-k4"
        A, B = json.load(open(f"{dd}/{base}-s129.json")), json.load(open(f"{dd}/{f}"))
        tgc = tg if tg in assign else tg.replace(".r1", "")
        if tgc not in assign: print(f"ds133: no assignment for {tg}", file=sys.stderr); continue
        pp = {}
        for p in ("c", "x"):
            P = SR.Paired(len(A[p]), {"kind": "full"})
            for (sa, na), (sb, nb) in zip(A[p], B[p]): P.add(sa, sb)
            pp[p] = {"logr": P.d_pooled(), "se": P.se(), "docs": len(A[p])}
        s129.append({"tag": tgc, "ref": base, "parts": pp})
    out = {"menu": os.path.abspath(a.menu), "W": M.W, "n_results": ri, "last_t": last_t, "a8_slices": sorted(a8),
           "obs": sorted(obs.values(), key=lambda o: (o["ri"], o["tag"])), "prior": prior, "events": events, "s129": s129,
           "assign": assign}
    with gzip.open(a.out, "wt") as f: json.dump(out, f)
    na = sum(e["decision"] == "accept" for e in events)
    print(f"ds133: {ri} results, {len(obs)} search rows + {len(prior)} prior, {len(events)} decided events ({na} accepts), "
          f"{len(s129)} s129 swaps, {len(assign)} assignments, last state line {last_t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
