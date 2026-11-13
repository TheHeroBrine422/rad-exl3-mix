#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""draw171.py REG LEDGER POOLS CALIB_KLD.json SEED NAME [NAME ..] -- the c parts of the first search slices (#171):
the merged fit + search pool in slices.py's least-used-first order (draw_merged's sort: ledger decision counts, the
head-input collections count 0, + this registry's own decisions; never a doc of a ready / active slice; ties shuffled by
SEED), cut into consecutive c parts of >= TARGET scored positions at cap CAP (env, default 45,000 at 4,096 = s129-c's
size). Prints one line per NAME: "NAME idx:cap ..." (golden_pick's arguments); a summary per part on stderr."""
import json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import slices as SL  # noqa: E402

TARGET, CAP = int(os.environ.get("TARGET", 45000)), int(os.environ.get("CAP", 4096))


def main(reg, led, pools, calib, seed, names):
    R = SL.Registry(reg); pool = json.load(open(pools))["pool"]; docs = json.load(open(calib))["docs"]
    mine = {f"golden-{k}-c" for k in R.d["slices"]}; lu = SL.ledger_use(led, mine); dd = R.d["doc_decisions"]
    busy = {int(d["id"][2:]) for s in R.d["slices"].values() if s["status"] in ("ready", "active") for d in s["docs"] if d["id"].startswith("c:")}
    cand = [i for i, p in enumerate(pool) if p in SL.MERGED and i not in busy]
    use = {i: lu.get(i, 0) + dd.get(f"c:{i}", 0) for i in cand}
    rng = random.Random(seed); cand.sort(key=lambda i: (use[i], rng.random()))
    k = 0
    for name in names:
        pick, n = [], 0
        while n < TARGET:
            i = cand[k]; k += 1; d = docs[i]
            r = min(d["tokens"], CAP) - 1 - d["score_from"]
            if r < 16: continue                                   # too short to score: skipped (not used)
            pick.append(i); n += r
        print(f"{name} " + " ".join(f"{i}:{CAP}" for i in sorted(pick)))
        print(f"{name}: {len(pick)} docs, {n} scored positions (cap {CAP}), uses min {min(use[i] for i in pick)} max "
              f"{max(use[i] for i in pick)}, pools {sorted({pool[i] for i in pick})}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6:])
