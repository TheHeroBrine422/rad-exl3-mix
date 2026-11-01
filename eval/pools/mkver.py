#!/usr/bin/env python3
"""mkver.py POOLS_V3.json CALIB.jsonl PROVENANCE.json OUT.json -- freeze ver3 / ver4 / the V4 reserve (2026-11-01, user).

  ver4 (V3's final check): only CLEAN docs (never in any slice: unused verify docs + the docs moved from qad by
        amend-2026-11-01), category-matched to the corpus, TOK tokens, docs cut to CAP; long part: trace 4723 whole,
        scored from 16384 (its first 16k were only in calB / calBbig checks).
  ver3 (V3's monitor, reused often, so it need not be clean): LIGHTLY used verify docs (in 1..MAXUSE slices, none of
        them a fit or search slice), category-matched, TOK tokens, cut to CAP; long part: trace 2300 from 8192.
  reserve: every clean doc ver4 did not take -- held for V4's final check; do not use before then.
Category shares are the corpus's token shares with long-agentic left out (it comes in as the separate long part).
Writes OUT.json (lists + golden_pick.py arguments) and prints the arguments."""
import collections, json, random, sys
PV, CAL, PROV, OUT = sys.argv[1:5]
TOK, CAP, MAXUSE, SEED = 150_000, 16384, 2, 20261101
pool = json.load(open(PV))["pool"]; cal = [json.loads(l) for l in open(CAL)]
use = {int(k): v for k, v in json.load(open(PROV))["doc_use"].items()}
TRACE3, TRACE4 = (2300, 8192), (4723, 16384)
LONGCAT = "long-agentic"
share = collections.Counter()
for d in cal:
    if d["category"] != LONGCAT: share[d["category"]] += int(d["tokens"])
tot = sum(share.values())
TUNED = ("calA", "fit", "search", "ml")       # slices that drove fits / selections, not checks
def light(i):
    u = use.get(i, [])
    return 1 <= len(u) <= MAXUSE and not any(t in g for g in u for t in TUNED)
traces = {TRACE3[0], TRACE4[0]}
clean = [i for i, p in enumerate(pool) if p == "verify" and i not in use and i not in traces]
lightd = [i for i, p in enumerate(pool) if p == "verify" and light(i) and i not in traces]
def pick(cands, seed):
    rng = random.Random(seed); by = collections.defaultdict(list); got = collections.Counter(); out = []
    for i in cands: by[cal[i]["category"]].append(i)
    for c in sorted(share):
        want = TOK * share[c] / tot; idx = by.get(c, [])[:]; rng.shuffle(idx)
        for i in idx:
            if got[c] >= want: break
            got[c] += min(int(cal[i]["tokens"]), CAP); out.append(i)
    short = {c: round(TOK * share[c] / tot - got[c]) for c in share if got[c] < 0.8 * TOK * share[c] / tot}
    return sorted(out), got, short
def args(ids, trace):
    a = [f"{i}:{CAP}" if int(cal[i]["tokens"]) > CAP else str(i) for i in ids]
    return a, [f"{trace[0]}:0:{trace[1]}"]
v4, g4, s4 = pick(clean, SEED)
v3, g3, s3 = pick(lightd, SEED + 1)
reserve = sorted(set(clean) - set(v4))
assert not set(v3) & set(v4) and not set(v4) & set(reserve)
rec = {"date": "2026-11-01", "pools": "pools-v3.json", "tokens": TOK, "cap": CAP, "seed": SEED, "maxuse": MAXUSE}
for name, ids, got, short, tr in (("ver4", v4, g4, s4, TRACE4), ("ver3", v3, g3, s3, TRACE3)):
    a, la = args(ids, tr)
    rec[name] = {"docs": ids, "tokens_by_cat": dict(got), "short_cats": short, "pick_args": a,
                 "long_args": la, "long_trace": {"doc": tr[0], "score_from": tr[1], "tokens": int(cal[tr[0]]["tokens"])}}
    print(f"{name}: {len(ids)} docs, {sum(got.values())} tok; short of target: {short or 'none'}; long {la[0]}")
rec["reserve"] = {"docs": reserve, "tokens": sum(int(cal[i]["tokens"]) for i in reserve),
                  "by_cat": dict(collections.Counter(cal[i]["category"] for i in reserve))}
print(f"reserve (V4 final check): {len(reserve)} docs, {rec['reserve']['tokens']} tok, {rec['reserve']['by_cat']}")
json.dump(rec, open(OUT, "w"), indent=1)
