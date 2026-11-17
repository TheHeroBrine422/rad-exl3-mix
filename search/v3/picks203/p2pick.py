# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p2pick.py CANDS-P2.json OUT.json [--report OUT.md] -- the final pick per size on phase 2 (RULES-picks am. 11-17 3:35 AM).
Per size, among candidates with a full phase-2 read (final750 + s195c6-c10):
  trim (am. 11-17 4:30 AM): documents whose median per-token KL across the size's candidates is >= 8x the slice median,
    at most 5 -- fixed from the reads alone, before any comparison;
  pick: best size-adjusted pooled KLD on the trimmed slice (b frozen: CANDS-P2["_b"], the phase-1 measured curve);
  guards: the pick vs the runner-up, paired bootstrap 2,000: not significantly worse (95 % CI wholly above 0) on (i) the
    full untrimmed slice and (ii) the calib-verify documents (final750 c1 / c3). A failed guard: the runner-up is checked
    the same way against the next, and so on.
OUT: per size the ranking (trimmed / untrimmed / verify, each vs the pick with CI), the trimmed documents and their
share of the pick's KL, the guard results, and whether the trim changed the pick."""
import json, math, sys

import numpy as np

import p2lib as L

cp, outp = sys.argv[1:3]; rep = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None
C = json.load(open(cp)); B = {float(k): v for k, v in C.get("_b", {}).items()}
R = L.Reads(2); VM = L.verify_mask(); IDS = L.doc_ids(2)
res, lines = {}, []
for s in sorted((k for k in C if not k.startswith("_")), key=float):
    S = float(s); b = B.get(S, 0.0)
    v = [c["tag"] for c in C[s] if R.full(c["tag"])]
    if not v: continue
    V = {t: R.vec(t) for t in v}
    keep, trim, rel = L.trim_mask([V[t] for t in v], R.pos(v[0]))
    adj = lambda t: b * (R.bpw[t] - S)
    tr = {t: math.log(V[t][keep].sum()) + adj(t) for t in v}
    un = {t: math.log(V[t].sum()) + adj(t) for t in v}
    order = sorted(v, key=lambda t: tr[t]); pick, guards = None, []
    for i, t in enumerate(order):
        rest = [u for u in order if u != t]
        if not rest: pick = t; break
        ru = rest[0]
        g_full = L.boot(V[t], V[ru], adj(t) - adj(ru))
        g_ver = L.boot(V[t][VM], V[ru][VM], adj(t) - adj(ru))
        ok = g_full[1] <= 0 and g_ver[1] <= 0
        guards.append({"cand": t, "vs": ru, "full": [L.pct(x) for x in g_full], "verify": [L.pct(x) for x in g_ver], "ok": ok})
        if ok: pick = t; break
    pick = pick or order[0]
    pick_un = min(v, key=lambda t: un[t])
    rows = []
    for t in order:
        if t == pick: rows.append({"tag": t, "trim": [0, 0, 0], "full": [0, 0, 0], "verify": [0, 0, 0]}); continue
        d = adj(t) - adj(pick)
        rows.append({"tag": t, "trim": [L.pct(x) for x in L.boot(V[t][keep], V[pick][keep], d)],
                     "full": [L.pct(x) for x in L.boot(V[t], V[pick], d)],
                     "verify": [L.pct(x) for x in L.boot(V[t][VM], V[pick][VM], d)]})
    res[s] = {"pick": pick, "pick_untrimmed": pick_un, "trim_changed_pick": pick != pick_un, "b": b,
              "trimmed_docs": [{"id": IDS[i], "per_token_x_median": float(rel[i])} for i in trim], "trimmed_share_of_pick": float(1 - V[pick][keep].sum() / V[pick].sum()),
              "kld_trim": float(V[pick][keep].sum() / R.pos(pick)[keep].sum()), "rows": rows, "guards": guards,
              "kld_full": float(V[pick].sum() / R.pos(pick).sum()), "n": len(v), "n_cands": len(C[s])}
    ru = rows[1] if len(rows) > 1 else None
    lines.append(f"| {S:g} | {len(v)}/{len(C[s])} | {pick.split('-', 2)[-1]} | {res[s]['kld_full']:.6f} | "
                 + (f"{ru['tag'].split('-', 2)[-1]} {ru['trim'][0]:+.2f} [{ru['trim'][1]:+.2f}, {ru['trim'][2]:+.2f}]" if ru else "—")
                 + f" | {100 * res[s]['trimmed_share_of_pick']:.1f} % | {'YES' if res[s]['trim_changed_pick'] else 'no'} |")
json.dump(res, open(outp, "w"), indent=1)
txt = ("| size | scored | pick | pick KLD (full) | runner-up (trimmed, vs pick) | trimmed docs' KL share | trim changed pick? |\n"
       "|---|---|---|---|---|---|---|\n" + "\n".join(lines))
print(txt)
if rep: open(rep, "w").write(txt + "\n")
