#!/usr/bin/env python3
"""provenance.py POOLS.json CALIB.jsonl OUT.json GOLDEN_DIR... -- which calib documents every eval slice (golden) has used.

Each golden's kld.json lists its documents (docs[].orig = index into the corpus it was cut from; the corpus is
identified by corpus_fnv1a64). Goldens cut from the calib corpus are mapped onto the pools (pools.json: fit / search /
qad / verify). Output: per golden its pool mix and its overlap with every other golden; per calib document the goldens
that used it; and the "clean" verify documents (in no golden at all), split into long (>= LONG tokens) and normal --
the only documents ver3 / ver4 may be drawn from (calib_slice.py EXCLUDE_GOLDEN=<all goldens> does the same filter).
Goldens on other corpora (public, long traces' own files) are listed by corpus hash only. Env LONG (default 32768),
CAL_FNV (the calib corpus's FNV-1a 64 as hex, skips hashing it)."""
import collections, json, os, sys

POOLS, CALIB, OUT = sys.argv[1:4]
GOLD = sys.argv[4:]
LONG = int(os.environ.get("LONG", 32768))
pool = json.load(open(POOLS))["pool"]
cal = [json.loads(l) for l in open(CALIB)]
assert len(cal) == len(pool)


def fnv(path):           # FNV-1a 64 of the corpus file, as the goldens record it
    h = 0xcbf29ce484222325
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            for c in b: h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


cal_hash = int(os.environ["CAL_FNV"], 16) if os.environ.get("CAL_FNV") else fnv(CALIB)
used = collections.defaultdict(list); gold = {}
for g in GOLD:
    k = json.load(open(f"{g}/kld.json")); name = os.path.basename(g.rstrip("/"))
    h = int(k["corpus_fnv1a64"], 16) if isinstance(k["corpus_fnv1a64"], str) else k["corpus_fnv1a64"]
    docs = sorted({d.get("orig", j) for j, d in enumerate(k["docs"])})   # no orig: the corpus in order (full golden)
    if h != cal_hash:
        gold[name] = {"corpus": k.get("corpus"), "calib": False, "docs": len(docs)}; continue
    pc = collections.Counter(pool[i] for i in docs)
    gold[name] = {"corpus": k.get("corpus"), "calib": True, "docs": len(docs), "pools": dict(pc),
                  "tokens": sum(int(cal[i]["tokens"]) for i in docs)}
    for i in docs: used[i].append(name)
names = [n for n in gold if gold[n]["calib"]]
docsets = {n: {i for i, gs in used.items() if n in gs} for n in names}
overlap = {a: {b: len(docsets[a] & docsets[b]) for b in names if b != a and docsets[a] & docsets[b]} for a in names}
ver = [i for i, p in enumerate(pool) if p == "verify"]
clean = [i for i in ver if i not in used]
clean_long = [i for i in clean if int(cal[i]["tokens"]) >= LONG]
spent_long = sorted((i, int(cal[i]["tokens"]), used[i]) for i in ver if i in used and int(cal[i]["tokens"]) >= LONG)
res = {"calib_fnv1a64": cal_hash, "goldens": gold, "overlap": overlap,
       "verify_docs": len(ver), "verify_clean": len(clean), "verify_clean_tokens": sum(int(cal[i]["tokens"]) for i in clean),
       "verify_clean_long": [(i, int(cal[i]["tokens"]), cal[i]["category"]) for i in clean_long],
       "verify_spent_long": spent_long, "clean_ids": clean,
       "doc_use": {str(i): gs for i, gs in sorted(used.items())}}
json.dump(res, open(OUT, "w"), indent=1)
print(f"calib corpus fnv {cal_hash:016x}")
for n, g in sorted(gold.items()):
    print(f"  {n:24s} " + (f"calib  {g['docs']:4d} docs {g['tokens']:9d} tok  pools {g['pools']}" if g["calib"] else f"other corpus ({g['docs']} docs)"))
print("overlaps between calib goldens (shared docs):")
for a in names:
    if overlap[a]: print(f"  {a:24s} " + ", ".join(f"{b} {n}" for b, n in sorted(overlap[a].items())))
print(f"verify pool: {len(ver)} docs, {len(clean)} never used ({res['verify_clean_tokens']} tokens), "
      f"{len(clean_long)} of them >= {LONG} tokens; long verify docs already spent: {len(spent_long)}")
for i, t, gs in spent_long: print(f"  spent: doc {i} ({t} tok) in {gs}")
for i, t, c in res["verify_clean_long"]: print(f"  clean long: doc {i} ({t} tok, {c})")
