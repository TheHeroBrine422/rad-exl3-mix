#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw129.py LEDGER POOLS CALIB_KLD.json SEED -- #169's calib documents (RULES-prior.md "Data"): the merged fit + search
pool in slices.py's least-used-first order (draw_merged's sort: ledger decision counts, ties shuffled by SEED), cut into
the fit slice (>= 140,000 scored positions at cap 8,192) and then the held-out slice's c part (>= 45,000 at cap 4,096).
Prints two golden_pick argument lines (idx:cap), then a summary on stderr."""
import json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3"))
import slices as SL  # noqa: E402


def main(led, pools, calib, seed):
    pool = json.load(open(pools))["pool"]; docs = json.load(open(calib))["docs"]
    lu = SL.ledger_use(led)
    cand = [i for i, p in enumerate(pool) if p in SL.MERGED]
    use = {i: lu.get(i, 0) for i in cand}
    rng = random.Random(seed); cand.sort(key=lambda i: (use[i], rng.random()))
    out, k = [], 0
    for name, target, cap in (("fit129", 140000, 8192), ("s129-c", 45000, 4096)):
        pick, n = [], 0
        while n < target:
            i = cand[k]; k += 1; d = docs[i]
            r = min(d["tokens"], cap) - 1 - d["score_from"]
            if r < 16: continue                                   # too short to score: skipped (not used)
            pick.append(i); n += r
        out.append(" ".join(f"{i}:{cap}" for i in sorted(pick)))
        print(f"{name}: {len(pick)} docs, {n} scored positions (cap {cap}), uses min {min(use[i] for i in pick)} max "
              f"{max(use[i] for i in pick)}, pools {sorted({pool[i] for i in pick})}", file=sys.stderr)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
