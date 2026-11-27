# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lrtab223.py QROOT [OUT.json] -- #222 / #223 lr screens: one row per arm with every read that exists (j217 from results.tsv; jc228 /
gap paired vs the unhealed release on the same judge: mean ratio + per-document bootstrap CI (cmp214, 2,000 draws), median doc
ratio, ratio without the 2 worst-delta docs). Arms: name -> (size, lr label, schedule, final ckpt)."""
import json, os, sys

import numpy as np

ROOT = sys.argv[1]; sys.path.insert(0, f"{ROOT}/code/v3/qad214")
import cmp214  # noqa: E402

ARMS = {"L30-b": (3.0, 1e-4, "flat"), "L35-a": (3.5, 1e-4, "flat"), "L40-a": (4.0, 1e-4, "flat"), "L40-b": (4.0, 5e-5, "flat"),
        "L20-as": (2.0, 3e-4, "3e-4>1.5M,1e-4"), "L20-bs": (2.0, 3e-4, "3e-4>2.4M"), "L30-as": (3.0, 3e-4, "3e-4>0.5M,1e-4"),
        "L35-bs": (3.5, 3e-4, "3e-4>0.5M,1e-4"),
        "G20-1e4": (2.0, 1e-4, "flat"), "G20-2e4": (2.0, 2e-4, "flat"), "G20-4e4": (2.0, 4e-4, "flat"),
        "G25-1e4": (2.5, 1e-4, "flat"), "G25-2e4": (2.5, 2e-4, "flat r6"), "G25-4e4": (2.5, 4e-4, "flat"),
        "G30-5e5": (3.0, 5e-5, "flat"), "G30-2e4": (3.0, 2e-4, "flat r6"), "G35-5e5": (3.5, 5e-5, "flat"), "G35-25e6": (3.5, 2.5e-5, "flat"),
        "G40-25e6": (4.0, 2.5e-5, "flat"), "G40-125e6": (4.0, 1.25e-5, "flat"), "R40-5e5": (4.0, 5e-5, "flat r6")}
JUDGE = {"JC": "golden-jc228", "JG": "golden-jgap", "": "golden-j217"}


def ckpt(arm): return "ckpt-3e+06" if arm.startswith("L20") else "ckpt-2e+06"


def pair(j, size, arm):
    a, b = f"{ROOT}/scores/{j}{'-' if j else ''}UN{size:.2f}.json.rows", f"{ROOT}/scores/{j}{'-' if j else ''}{arm}-{ckpt(arm)}.json.rows"
    if not os.path.exists(a) and not j: a = f"/data/q27/w217/scores/ctl-{size:.2f}.json.rows"     # j217 control of the release (w217)
    if not (os.path.exists(a) and os.path.exists(b)): return None
    G = f"{ROOT}/slices/{JUDGE[j]}" if os.path.isdir(f"{ROOT}/slices/{JUDGE[j]}") else f"/data/q27/w217/slices/{JUDGE[j]}"
    xa, n = cmp214.doc_sums(G, a); xb, _ = cmp214.doc_sums(G, b)
    r = xb.sum() / xa.sum(); rng = np.random.default_rng(0); idx = rng.integers(0, len(xa), (2000, len(xa)))
    bs = xb[idx].sum(1) / xa[idx].sum(1); m = (xa > 0) & (n > 0)
    d = xb - xa; keep = np.ones(len(d), bool); keep[np.argsort(-d)[:2]] = False
    return {"r": float(r), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "med": float(np.median(xb[m] / xa[m])), "wo2": float(xb[keep].sum() / xa[keep].sum()), "better": int((d < 0).sum()), "n": len(d)}


rows = []
for arm, (size, lr, sch) in ARMS.items():
    row = {"arm": arm, "size": size, "lr": lr, "sched": sch}
    for j in ("", "JC", "JG"):
        try: row[j or "J217"] = pair(j, size, arm)
        except Exception as e: row[j or "J217"] = {"err": str(e)[:80]}
    rows.append(row)
f = lambda v: "—" if not v else ("err" if "err" in v else f"{v['r']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] med {v['med']:.3f} wo2 {v['wo2']:.3f}")
print("| arm | size | lr | schedule | j217 | jc228 | gap |\n|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["size"], r["lr"])): print(f"| {r['arm']} | {r['size']} | {r['lr']:g} | {r['sched']} | {f(r['J217'])} | {f(r['JC'])} | {f(r['JG'])} |")
if len(sys.argv) > 2: json.dump(rows, open(sys.argv[2], "w"), indent=1)
