#!/usr/bin/env python3
"""cmp205.py DOCSDIR -- mad-only swap check (RULES-picks am. 11-17 5:10 AM): each frozen pick vs its mad swap, paired per document
on the 6 phase-2 slices. Pick rows = the coordinator's phase-2 docs; mad rows = DOCSDIR/<size>-mad-<slice>.json (run205.sh, box2).
Statistic = the phase-2 one: the size's trim mask from its phase-2 candidates (p2lib.trim_mask, as p2pick.py), bootstrap 2,000.
Same bytes (mad tables / heads have the i8 / mul1 sizes), so no size adjustment. Adopt mad-only iff no size > +1 % (trimmed point)
and none significantly worse (CI wholly above 0) below 6.0."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2lib as L
D = sys.argv[1]; R = L.Reads(2)
F = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/picks203/picks-frozen.json")))
C = json.load(open("/workspace/rad-local/workers/lead-srch/picks203/cands-p2.json"))
f = lambda e: f"{L.pct(e[0]):+.2f} [{L.pct(e[1]):+.2f}, {L.pct(e[2]):+.2f}]"
ok_all, rows = True, []
print("| size | pick | mad swap vs pick, trimmed | untrimmed | calib-verify | ok |\n|---|---|---|---|---|---|")
for s, x in F.items():
    sz = s[:-2] if s.endswith(".0") else s
    fs = [f"{D}/{sz}-mad-{sl}.json" for sl in L.P2]
    if not all(os.path.exists(p) for p in fs): print(f"| {s} | (incomplete: {sum(map(os.path.exists, fs))}/6) |"); ok_all = None if ok_all else ok_all; continue
    m = np.concatenate([np.array(json.load(open(p))["m"], float)[:, 0] for p in fs])
    p = R.vec(x["pick"]); tags = [c["tag"] for c in C[s if s in C else sz] if R.full(c["tag"])]
    keep, _, _ = L.trim_mask([R.vec(t) for t in tags], R.pos(tags[0]))
    et, eu, ev = L.boot(m[keep], p[keep]), L.boot(m, p), L.boot(m[L.verify_mask()], p[L.verify_mask()])
    ok = L.pct(et[0]) <= 1.0 and not (float(s) < 6.0 and et[1] > 0)
    ok_all = ok_all and ok
    print(f"| {s} | {x['pick'].split('-', 2)[2]} | {f(et)} | {f(eu)} | {f(ev)} | {'yes' if ok else 'NO'} |")
print("\nmad-only:", {True: "ADOPT (all 16 pass)", False: "REJECT (a size fails)", None: "incomplete"}[ok_all])
