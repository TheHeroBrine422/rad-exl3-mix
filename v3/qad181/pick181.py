#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pick181.py REF RUN [RUN ...] [--runs DIR] [--what "ckpt 1e+06"] -- #181 amendment 4's screen readout: per run the fixed
eval KL at its decayed end and the paired per-document ratio vs REF (95 % document bootstrap, 2,000 resamples, seed 0),
the monitor ratio vs REF, and the rule's pick (lowest fixed-eval KL; it replaces REF only if its ratio's CI is below 1
and its monitor ratio is below 1). JSON on stdout, a table on stderr."""
import argparse, json, os, sys

import numpy as np


def last(path, what):
    L = [json.loads(l) for l in open(path)]
    L = [x for x in L if x.get("what") == what]
    return L[-1] if L else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default="runs"); ap.add_argument("--what", default="ckpt 1e+06")
    a = ap.parse_args()
    rows = {}
    for r in [a.ref] + a.runs:
        f = last(os.path.join(a.dir, r, "fixed.jsonl"), a.what); m = last(os.path.join(a.dir, r, "mon.jsonl"), a.what)
        if f is None: print(f"{r}: no '{a.what}' fixed eval yet", file=sys.stderr); continue
        rows[r] = {"kl": f["kl"], "sums": np.array(f["sums"]), "n": np.array(f["n"]), "mon": m["mon_kl"] if m else None}
    if a.ref not in rows: raise SystemExit("REF has no result")
    R = rows[a.ref]; out = {"ref": a.ref, "runs": {}}
    for r, v in rows.items():
        assert (v["n"] == R["n"]).all(), r
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(R["n"]), len(R["n"])); bs.append(v["sums"][i].sum() / R["sums"][i].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["runs"][r] = {"fixed_kl": v["kl"], "ratio": float(v["sums"].sum() / R["sums"].sum()), "ci95": [float(lo), float(hi)],
                          "docs_better": int((v["sums"] < R["sums"]).sum()), "docs_worse": int((v["sums"] > R["sums"]).sum()),
                          "monitor": v["mon"], "mon_ratio": (v["mon"] / R["mon"]) if (v["mon"] and R["mon"]) else None}
    best = min(out["runs"], key=lambda r: out["runs"][r]["fixed_kl"]); b = out["runs"][best]
    adopt = best != a.ref and b["ci95"][1] < 1 and (b["mon_ratio"] or 2) < 1
    out["best"] = best; out["pick"] = best if adopt else a.ref
    for r, v in sorted(out["runs"].items(), key=lambda x: x[1]["fixed_kl"]):
        print(f"{r:14s} fixed {v['fixed_kl']:.5f}  vs ref {v['ratio']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] "
              f"({v['docs_better']}/{v['docs_worse']})  monitor {v['monitor']}  ({v['mon_ratio']})", file=sys.stderr)
    print(f"lowest: {best}; pick (rule): {out['pick']}", file=sys.stderr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
