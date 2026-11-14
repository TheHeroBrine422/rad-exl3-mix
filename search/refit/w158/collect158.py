#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""collect158.py MENU OUT.jsonl SPECS.jsonl.. --docs DOCSUMS.json.. [--anchor TAG] -- (w158, #173-03) scored rows as fit obs:
{tag, assign, label "fit129", mode a8 | f16, kld (positions-weighted mean over the slice's scored positions), se (delta-method
SE of log KLD / KLD(anchor) on the shared documents, w143's lse; the anchor's own 0), base_tag + se_vs_base (swap rows: the U
row they move from), size (menu container bpw incl. a/b), block, meta}. Rows without a score are listed on stderr."""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import menu as MN  # noqa: E402


def lse(D, terms):
    z = 0.0
    for k, c in terms:
        a = D[k][:, 0]; z = z + c * a / a.sum()
    return float(np.sqrt(np.sum((z - np.mean(z)) ** 2) * len(z) / max(len(z) - 1, 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("menu"); ap.add_argument("out"); ap.add_argument("specs", nargs="+")
    ap.add_argument("--docs", nargs="+", required=True); ap.add_argument("--anchor", default="w158-u-k2.5"); ap.add_argument("--label", default="fit129")
    a = ap.parse_args(argv)
    M = MN.Menu.load(a.menu); ext = M.d.get("ab_t5_extra") or 0
    S = {}
    for p in a.specs:
        for l in open(p):
            if l.strip(): s = json.loads(l); S[s["tag"]] = s
    R = {}
    for p in a.docs: R.update(json.load(open(p)))
    D = {t: np.array(r["docs"], float) for t, r in R.items()}
    n0 = next(iter(D.values()))[:, 1]
    for t, d in D.items(): assert np.array_equal(d[:, 1], n0), t
    rows, miss = [], []
    for t, s in S.items():
        if t not in R: miss.append(t); continue
        m = s["meta"]; d = D[t]
        kld = float(d[:, 0].sum() / d[:, 1].sum())
        base = None
        if m["block"] in ("G", "H", "E", "B", "S2"): base = f"w158-u-k{m['base']:g}"
        o = {"tag": t, "assign": s["assign"], "label": a.label, "mode": "a8" if s.get("act", True) else "f16", "kld": kld,
             "se": 0.0 if t == a.anchor else lse(D, [(t, 1), (a.anchor, -1)]), "block": m["block"],
             "size": round(M.bpw(s["assign"]) + (ext * 8 / M.P if s["ab"] == "t5" else 0), 4), "meta": m}
        if base and base in D: o["base_tag"] = base; o["se_vs_base"] = lse(D, [(t, 1), (base, -1)]); o["log_vs_base"] = math.log(kld / float(D[base][:, 0].sum() / D[base][:, 1].sum()))
        rows.append(o)
    with open(a.out, "w") as f:
        for o in rows: f.write(json.dumps(o) + "\n")
    print(f"{a.out}: {len(rows)} rows, {len(miss)} without a score{': ' + ', '.join(miss[:8]) if miss else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
