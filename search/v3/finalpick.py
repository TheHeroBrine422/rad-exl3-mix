# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""finalpick.py SPEC.json [--out OUT.json] [--boot 4000] [--seed 0] -- the final pick of one release band (w127, #167;
docs/design/eval-data.md "Final picks", TODO "Final pick score").

The top ~5 candidates of a band (driver frontier.json "top") are scored once each, whole, on the calib part (golden-ver4,
91 verify docs, frozen) and a fresh xcorp part (several hundred docs, 8k cap). Score = size-adjusted log KLD:
    S = log KLD + b (bpw - T),  KLD = sum of per-document KL sums / positions (the challenge's mean over positions),
b = the band's slope frozen before the ver4 reads (driver.py CONF --freeze-b), T = the release size, bpw = VRAM bytes x
8 / 26,895,998,464. The adjustment is a constant per candidate, so the bootstrap only resamples documents.
Pick = the best pooled S (both parts' documents together), unless the calib part VETOES it: a candidate is vetoed when
its calib-part S is significantly worse (95% paired bootstrap CI wholly > 0) than another finalist's; the pick is the
best pooled non-vetoed one. The calib part never ranks (it is small).
Reported per candidate vs the pooled best: S per part and pooled, paired bootstrap CIs (documents resampled within each
part, the same draw for every candidate) of the S difference per part, pooled, and the part gap (calib diff - xcorp
diff: do the parts disagree?).

SPEC: {"band": T, "b": number | "FROZEN.json" (its bands[T].b), "parts": ["c", "x"],
       "cands": [{"tag", "bpw", "docs": {"c": DOCS.json, "x": DOCS.json}}, ...]}
DOCS.json: rxrun.multi's docs/<tag>-<slice>.json ({"m": [[kl_sum, positions], ...]}; the first list in the file) or a
plain list of [kl_sum, positions]; every candidate's file of a part must list the same documents in the same order."""
import argparse, json, math, sys

import numpy as np


def load_docs(p):
    d = json.load(open(p))
    if isinstance(d, dict): d = next(iter(d.values()))
    if any(x is None for x in d): raise SystemExit(f"{p}: a stopped / partial run (None documents): final picks need whole runs")
    a = np.array(d, float)
    return a[:, 0], a[:, 1]


def frozen_b(b, band):
    if isinstance(b, (int, float)): return float(b), "spec"
    d = json.load(open(b)); bd = d["bands"]
    k = next((k for k in bd if abs(float(k) - band) < 1e-9), None)
    if k is None: raise SystemExit(f"{b}: no band {band}")
    return float(bd[k]["b"]), f"{b} (frozen {d.get('frozen')})"


def pick(spec, boot=4000, seed=0):
    T = float(spec["band"]); b, bsrc = frozen_b(spec["b"], T); parts = spec.get("parts", ["c", "x"]); C = spec["cands"]
    S, N = {}, {}
    for p in parts:
        rows = [load_docs(c["docs"][p]) for c in C]
        n0 = rows[0][1]
        for c, (_, n) in zip(C, rows):
            if len(n) != len(n0) or not np.array_equal(n, n0): raise SystemExit(f"part {p}: {c['tag']} scored other documents / positions than {C[0]['tag']}")
        S[p] = np.stack([s for s, _ in rows]); N[p] = n0                        # S[p]: cands x docs
    adj = np.array([b * (float(c["bpw"]) - T) for c in C])

    def scores(w):
        """w: {part: doc weights (boot counts) or None} -> {part: cand scores, "pooled": ...}"""
        out = {}; tot_s = 0; tot_n = 0
        for p in parts:
            wp = np.ones(len(N[p])) if w is None else w[p]
            s = S[p] @ wp; n = N[p] @ wp; out[p] = np.log(s / n) + adj; tot_s = tot_s + s; tot_n = tot_n + n
        out["pooled"] = np.log(tot_s / tot_n) + adj
        return out

    pt = scores(None)
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in list(parts) + ["pooled"]}
    for _ in range(boot):
        w = {p: rng.multinomial(len(N[p]), np.full(len(N[p]), 1.0 / len(N[p]))).astype(float) for p in parts}
        sc = scores(w)
        for k in bs: bs[k].append(sc[k])
    bs = {k: np.array(v) for k, v in bs.items()}                                  # boot x cands
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    order = list(np.argsort(pt["pooled"]))
    cpart = parts[0]                                                               # the calib part (vetoes)
    veto = {}
    for i in range(len(C)):
        worse = [C[j]["tag"] for j in range(len(C)) if j != i and np.percentile(bs[cpart][:, i] - bs[cpart][:, j], 2.5) > 0]
        if worse: veto[C[i]["tag"]] = worse
    chosen = next((i for i in order if C[i]["tag"] not in veto), None)
    best = order[0]
    rows = []
    for r, i in enumerate(order):
        row = {"rank": r + 1, "tag": C[i]["tag"], "bpw": float(C[i]["bpw"]), "adj": float(adj[i]),
               "kld": {p: float(math.exp(pt[p][i] - adj[i])) for p in list(parts) + ["pooled"]},
               "score": {p: float(pt[p][i]) for p in list(parts) + ["pooled"]}, "veto": veto.get(C[i]["tag"])}
        if i != best:
            row["vs_best"] = {p: {"d": float(pt[p][i] - pt[p][best]), "ci": ci(bs[p][:, i] - bs[p][:, best])} for p in list(parts) + ["pooled"]}
            if len(parts) == 2:
                g = (bs[parts[0]][:, i] - bs[parts[0]][:, best]) - (bs[parts[1]][:, i] - bs[parts[1]][:, best])
                row["vs_best"]["part_gap"] = {"d": float((pt[parts[0]][i] - pt[parts[0]][best]) - (pt[parts[1]][i] - pt[parts[1]][best])), "ci": ci(g)}
        rows.append(row)
    return {"band": T, "b": b, "b_src": bsrc, "parts": {p: {"docs": int(len(N[p])), "positions": int(N[p].sum())} for p in parts},
            "boot": boot, "pooled_best": C[best]["tag"], "pick": C[chosen]["tag"] if chosen is not None else None, "vetoed": veto, "rows": rows}


def show(R):
    print(f"band {R['band']:g}: b {R['b']:.3f} ({R['b_src']}); parts " + ", ".join(f"{p} {v['docs']} docs / {v['positions']} pos" for p, v in R["parts"].items()))
    ps = list(R["parts"]) + ["pooled"]
    print(f"{'#':>2} {'tag':34s} {'bpw':>7} " + " ".join(f"{'KLD ' + p:>10}" for p in ps) + "  " + "  ".join(f"{'dS ' + p + ' [95% CI]':>26}" for p in ps + ["part_gap"]))
    for r in R["rows"]:
        vb = r.get("vs_best")
        cis = "  ".join((f"{vb[p]['d']:+.4f} [{vb[p]['ci'][0]:+.4f}, {vb[p]['ci'][1]:+.4f}]" if vb and p in vb else f"{'(best)' if not vb else '':>26}").rjust(26) for p in ps + ["part_gap"])
        print(f"{r['rank']:>2} {r['tag']:34s} {r['bpw']:7.4f} " + " ".join(f"{r['kld'][p]:10.5f}" for p in ps) + "  " + cis + (f"  VETO (calib worse than {', '.join(r['veto'])})" if r["veto"] else ""))
    print(f"pooled best {R['pooled_best']}; PICK {R['pick']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="finalpick.py")
    ap.add_argument("spec"); ap.add_argument("--out"); ap.add_argument("--boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    R = pick(json.load(open(a.spec)), a.boot, a.seed); show(R)
    if a.out: json.dump(R, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
