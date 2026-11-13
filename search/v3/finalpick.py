# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""finalpick.py SPEC.json [--out OUT.json] [--boot 10000] [--seed 0] -- the final pick of one release band
(w127, #167; rewritten for RULES-picks.md by w145, 11-13).

The top ~5 candidates of a band (driver frontier.json "top") are scored once each, whole, on the final-pick slice:
a calib part c (ver4 + reused verify documents) and a fresh xcorp part x, ~750k scored positions together. Per part
and candidate pair (pickstat.py): (a) pooled log ratio, (b) capped (a document <= 20 % of its part), (c) per-document
mean; CIs by paired bootstrap over units (a document; a long trace's chunks are one unit), the same draws for every
pair, 10,000 draws, 95 %.

Score (the decision, RULES-picks "Final picks"): (a), size-adjusted with the band's frozen b:
    S = log KLD + b (bpw - T),  KLD = sum of per-document KL sums / positions over BOTH parts' documents together
(the board's mean over positions; b from driver.py --freeze-b, T the release size, bpw = VRAM bytes x 8 / 26,895,998,464).
The best S wins, unless the GUARD fails: the winner's (c) must not be significantly worse than any other finalist's in
either part (95 % CI of its size-adjusted (c) difference wholly > 0); a guarded-out candidate is passed over and the
next best S is checked the same way. CHECK: any document holding > 20 % of a part's KL (any candidate) is flagged:
look at it before trusting the pick (expected never).
Reported: per candidate S pooled and per part; for the pick vs the runner-up and vs the default pick (SPEC "default",
the re-solve's), and for every candidate vs the pick: (a) pooled, and (a) / (b) / (c) per part with CIs.

SPEC: {"band": T, "b": number | "FROZEN.json" (its bands[T].b), "parts": ["c", "x"], "default": TAG (optional),
       "units": {"c": UNITS, ...} (optional; UNITS = a JSON list with one label per document, or a golden's kld.json:
                 label = source:orig of each document, so a trace's chunks share one unit; absent = each document),
       "cands": [{"tag", "bpw", "docs": {"c": DOCS.json, "x": DOCS.json}}, ...]}
DOCS.json: rxrun.multi's docs/<tag>-<slice>.json ({"m": [[kl_sum, positions], ...]}; the first list in the file) or a
plain list of [kl_sum, positions]; every candidate's file of a part must list the same documents in the same order."""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickstat as PS  # noqa: E402

FLAG = 0.2


def load_docs(p):
    d = json.load(open(p)) if isinstance(p, str) else p
    if isinstance(d, dict): d = next(iter(d.values()))
    if any(x is None for x in d): raise SystemExit(f"{p if isinstance(p, str) else 'docs'}: a stopped / partial run (None documents): final picks need whole runs")
    a = np.array(d, float)
    return a[:, 0], a[:, 1]


def load_units(u, n):
    if u is None: return None
    d = json.load(open(u)) if isinstance(u, str) else u
    if isinstance(d, dict): d = [f"{x.get('source', '')}:{x.get('orig', i)}" for i, x in enumerate(d["docs"])]
    if len(d) != n: raise SystemExit(f"units: {len(d)} labels for {n} documents")
    return d


def frozen_b(b, band):
    if isinstance(b, (int, float)): return float(b), "spec"
    d = json.load(open(b)); bd = d["bands"]
    k = next((k for k in bd if abs(float(k) - band) < 1e-9), None)
    if k is None: raise SystemExit(f"{b}: no band {band}")
    return float(bd[k]["b"]), f"{b} (frozen {d.get('frozen')})"


def pick(spec, boot=PS.BOOT, seed=0):
    T = float(spec["band"]); b, bsrc = frozen_b(spec["b"], T); parts = spec.get("parts", ["c", "x"]); C = spec["cands"]
    tags = [c["tag"] for c in C]; nc = len(C)
    S, N, U = {}, {}, {}
    for p in parts:
        rows = [load_docs(c["docs"][p]) for c in C]
        n0 = rows[0][1]
        for c, (_, n) in zip(C, rows):
            if len(n) != len(n0) or not np.array_equal(n, n0): raise SystemExit(f"part {p}: {c['tag']} scored other documents / positions than {C[0]['tag']}")
        S[p] = np.stack([s for s, _ in rows]); N[p] = n0                        # S[p]: cands x docs
        U[p] = load_units((spec.get("units") or {}).get(p), len(n0))
    adj = np.array([b * (float(c["bpw"]) - T) for c in C])
    Cn = {p: PS.draws(len(N[p]), U[p], boot, seed + k) for k, p in enumerate(parts)}   # shared copy counts per part

    # (a) pooled over both parts: point and bootstrap per candidate
    def pooled(w):
        s = sum(S[p] @ (np.ones(len(N[p])) if w is None else w[p]) for p in parts)
        n = sum(N[p] @ (np.ones(len(N[p])) if w is None else w[p]) for p in parts)
        return np.log(s / n) + adj
    pt = pooled(None)
    bsp = []
    for i in range(0, boot, 1000):
        s = sum(Cn[p][i:i + 1000].astype(float) @ S[p].T for p in parts); n = sum(Cn[p][i:i + 1000].astype(float) @ N[p] for p in parts)
        bsp.append(np.log(s / n[:, None]) + adj)
    bsp = np.concatenate(bsp)                                                     # boot x cands
    part_pt = {p: np.log(S[p].sum(1) / N[p].sum()) + adj for p in parts}

    cache = {}

    def pair(i, j):
        """candidate i vs reference j: pooled (a) and per part (a) / (b) / (c), size-adjusted, with CIs"""
        if (i, j) in cache: return cache[(i, j)]
        off = float(adj[i] - adj[j])
        r = {"pooled_a": {"d": float(pt[i] - pt[j]), "ci": PS.ci(bsp[:, i] - bsp[:, j])}}
        for p in parts: r[p] = PS.compare(S[p][j], S[p][i], cap=FLAG, off=off, C=Cn[p])
        cache[(i, j)] = r; return r

    order = [int(i) for i in np.argsort(pt)]
    guard = {}
    for i in order:                                                               # the guard: (c) not significantly worse than any other finalist
        bad = [f"{tags[j]} ({p})" for j in range(nc) if j != i for p in parts if pair(i, j)[p]["c"]["ci"][0] > 0]
        guard[tags[i]] = bad
        if not bad: break
    chosen = next((i for i in order if tags[i] in guard and not guard[tags[i]]), None)
    flags = []
    for p in parts:
        for i in range(nc):
            sh = S[p][i] / S[p][i].sum(); k = int(np.argmax(sh))
            if sh[k] > FLAG: flags.append({"part": p, "cand": tags[i], "doc": k, "unit": U[p][k] if U[p] else None, "share": float(sh[k])})
    share = {p: max(PS.max_share(S[p][i]) for i in range(nc)) for p in parts}
    rows = []
    for r, i in enumerate(order):
        rows.append({"rank": r + 1, "tag": tags[i], "bpw": float(C[i]["bpw"]), "adj": float(adj[i]),
                     "kld": {**{p: float(math.exp(part_pt[p][i] - adj[i])) for p in parts}, "pooled": float(math.exp(pt[i] - adj[i]))},
                     "score": {**{p: float(part_pt[p][i]) for p in parts}, "pooled": float(pt[i])},
                     "guard_fail": guard.get(tags[i]) or None,
                     "vs_pick": pair(i, chosen) if chosen is not None and i != chosen else None})
    out = {"band": T, "b": b, "b_src": bsrc, "boot": boot, "cap": FLAG,
           "parts": {p: {"docs": int(len(N[p])), "units": len(set(U[p])) if U[p] else int(len(N[p])), "positions": int(N[p].sum()),
                         "max_doc_share": share[p]} for p in parts},
           "best_a": tags[order[0]], "pick": tags[chosen] if chosen is not None else None, "flags": flags, "rows": rows, "vs": {}}
    if chosen is not None:
        ru = next((i for i in order if i != chosen), None)
        if ru is not None: out["vs"]["runner_up"] = {"tag": tags[ru], **pair(chosen, ru)}
        dt = spec.get("default")
        if dt is not None:
            if dt not in tags: raise SystemExit(f"default {dt} is not a candidate")
            j = tags.index(dt)
            out["vs"]["default"] = {"tag": dt, **(pair(chosen, j) if j != chosen else {"same": True})}
    return out


def fmt(x): return f"{x['d']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]"


def show(R):
    print(f"band {R['band']:g}: b {R['b']:.3f} ({R['b_src']}); " + ", ".join(
        f"{p} {v['docs']} docs / {v['units']} units / {v['positions']} pos (max doc share {100 * v['max_doc_share']:.1f}%)" for p, v in R["parts"].items()))
    ps = list(R["parts"])
    print(f"{'#':>2} {'tag':34s} {'bpw':>7} " + " ".join(f"{'KLD ' + p:>10}" for p in ps + ["pooled"]) + "  " +
          f"{'dS pooled (a) vs pick':>27}  " + "  ".join(f"{'(' + s + ') ' + p + ' vs pick':>27}" for p in ps for s in "abc"))
    for r in R["rows"]:
        v = r["vs_pick"]
        cis = (f"{fmt(v['pooled_a']):>27}  " + "  ".join(f"{fmt(v[p][s]):>27}" for p in ps for s in "abc")) if v else f"{'(pick)' if r['tag'] == R['pick'] else '':>27}"
        print(f"{r['rank']:>2} {r['tag']:34s} {r['bpw']:7.4f} " + " ".join(f"{r['kld'][p]:10.5f}" for p in ps + ["pooled"]) + "  " + cis +
              (f"  GUARD: (c) worse than {', '.join(r['guard_fail'])}" if r["guard_fail"] else ""))
    for k, v in R["vs"].items():
        if v.get("same"): print(f"pick vs {k}: the same candidate"); continue
        print(f"pick vs {k} {v['tag']}: pooled (a) {fmt(v['pooled_a'])}; " + "; ".join(f"{p}: (a) {fmt(v[p]['a'])} (b) {fmt(v[p]['b'])} (c) {fmt(v[p]['c'])}" for p in ps))
    for f in R["flags"]: print(f"FLAG: part {f['part']} doc {f['doc']} (unit {f['unit']}) holds {100 * f['share']:.1f}% of {f['cand']}'s KL: look before trusting the pick")
    print(f"best (a) {R['best_a']}; PICK {R['pick']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="finalpick.py")
    ap.add_argument("spec"); ap.add_argument("--out"); ap.add_argument("--boot", type=int, default=PS.BOOT); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    R = pick(json.load(open(a.spec)), a.boot, a.seed); show(R)
    if a.out: json.dump(R, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
