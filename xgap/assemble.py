#!/usr/bin/env python3
"""assemble.py -- finish the corpus: global dedup of the xgap sources against each other, seeded 10 % holdout, final
corpus files, manifest.jsonl, sources.json, summary.md.

  in : work/src/*.jsonl|.meta.jsonl|.stats.json (build_source.py), work/tags-calib.jsonl, work/tags-xcorp.jsonl (tag.py)
  out: corpus/xgap-v1/<cell>/<src_id>.jsonl (lines {id, source, category, kind, tokens, prompt}), manifest.jsonl,
       sources.json, summary.md, dedup.json
Dedup inside xgap: exact (normalized content hash), then the line rule (>= 20 % or >= 25 of a document's long lines, not
boilerplate, already in an earlier document of ANOTHER session / source record). Sources are visited in registry order, so
a duplicate is dropped from the later source. Holdout: per (cell, length band) stratum, documents ranked by
sha256(seed:id); the first round(10 %) are `holdout: true` (>= 1 when the stratum has >= 6 documents)."""
import collections, glob, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import sources

HOLDOUT_FRAC = 0.10
HOLDOUT_SEED = "xgap-v1-holdout-20261125"
SKIPPED = [   # (source, why) -- filled in by the lead's review of the pilot / full build
    ("Salesforce/xlam-function-calling-60k", "gated dataset (CC BY 4.0, auto-approval after accepting terms): the colo login has no access; not worked around"),
    ("fineweb-2 'HQ' languages beyond epfml/FineWeb2-HQ's 20", "FineWeb2-HQ only ships 20 languages (all already in calib); ko/hi/th/he/uk/bn/ta/... come from HuggingFaceFW/fineweb-2 (odc-by) with language_score >= 0.95"),
    ("whole small repos (the-stack-smol-xl / stackv2-edu)", "neither dataset groups files by repository (files are shuffled; stack-edu shards are per language): whole files only"),
]


def main():
    srcs = [s for s in sources.S if os.path.exists(f"{SRC_DIR}/{s['id']}.stats.json")]
    # pass 1: boilerplate lines (in >= 3 xgap documents)
    cnt = collections.Counter(); ndocs = 0
    for s in srcs:
        for l in open(f"{SRC_DIR}/{s['id']}.jsonl"):
            ndocs += 1
            cnt.update({lhash(x) for x in long_lines(json.loads(l)["prompt"])})
    boiler = {h for h, c in cnt.items() if c >= 3}
    del cnt
    # pass 2: dedup + collect
    seen, owner = {}, {}
    kept = []; drops = collections.Counter(); per_src_drop = collections.defaultdict(collections.Counter)
    for s in srcs:
        L = open(f"{SRC_DIR}/{s['id']}.jsonl"); M = open(f"{SRC_DIR}/{s['id']}.meta.jsonl")
        for l, m in zip(L, M):
            d = json.loads(l); meta = json.loads(m); p = d["prompt"]
            h = norm_hash(p)
            if h in seen: drops["exact"] += 1; per_src_drop[s["id"]]["exact"] += 1; continue
            key = meta.get("session") or meta["id"]
            ll = [x for x in map(lhash, long_lines(p)) if x not in boiler]
            hits = sum(1 for x in ll if x in owner and owner[x] != key)
            if hits >= 2 and (hits >= 0.2 * len(ll) or hits >= 25): drops["lines"] += 1; per_src_drop[s["id"]]["lines"] += 1; continue
            seen[h] = 1
            for x in ll: owner.setdefault(x, key)
            kept.append((s, d, meta))
    # holdout
    strata = collections.defaultdict(list)
    for i, (s, d, meta) in enumerate(kept): strata[(meta["cell"], meta["length_band"])].append(i)
    hold = set()
    for k, ids in strata.items():
        ids.sort(key=lambda i: hashlib.sha256(f"{HOLDOUT_SEED}:{kept[i][2]['id']}".encode()).hexdigest())
        n = round(HOLDOUT_FRAC * len(ids))
        if n == 0 and len(ids) >= 6: n = 1
        hold.update(ids[:n])
    # write corpus + manifest
    rows = []
    for fn in ("tags-calib.jsonl", "tags-xcorp.jsonl"):
        p = f"{ROOT}/work/{fn}"
        if os.path.exists(p): rows += [json.loads(l) for l in open(p)]
    byfile = collections.defaultdict(list)
    for i, (s, d, meta) in enumerate(kept): byfile[(meta["cell"], s["id"])].append(i)
    for (cell, sid), ids in sorted(byfile.items()):
        rel = f"corpus/xgap-v1/{cell}/{sid}.jsonl"; os.makedirs(os.path.dirname(f"{ROOT}/{rel}"), exist_ok=True)
        with open(f"{ROOT}/{rel}.part", "w") as fo:
            for ln, i in enumerate(ids):
                s, d, meta = kept[i]
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                r = dict(meta); r["holdout"] = i in hold; r["file"] = rel; r["line"] = ln
                rows.append(r)
        os.replace(f"{ROOT}/{rel}.part", f"{ROOT}/{rel}")
    with open(f"{ROOT}/manifest.jsonl.part", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{ROOT}/manifest.jsonl.part", f"{ROOT}/manifest.jsonl")
    # sources.json
    sj = {"calib": {"dataset": "StillDeadcode/qwen3.8-27b-quant-challenge", "file": "corpus/calib.jsonl", "revision": os.environ.get("CALIB_REVISION"),
                    "note": "the public split is never included (hash-only dedup)"},
          "xcorp-v1": {"dir": XCORP, "files": sorted({r["xcorp_file"] for r in rows if r["origin"] == "xcorp-v1"}), "note": "every pisess* file refused"},
          "xgap-v1": [], "skipped": [{"source": a, "why": b} for a, b in SKIPPED]}
    for s in srcs:
        st = json.load(open(f"{SRC_DIR}/{s['id']}.stats.json"))
        n = [r for r in rows if r.get("src_id") == s["id"]]
        sj["xgap-v1"].append({"id": s["id"], "source": s["source"], "cell": s["cell"], "dataset": st["dataset"], "config": st["config"], "split": st["split"],
                              "revision": st["revision"], "licence": sorted({r["licence"] for r in n if r["licence"]})[:12],
                              "budgets": st["budgets"], "built": st["tokens"], "docs_final": len(n), "tokens_final": sum(r["tokens"] for r in n),
                              "stopped": st["stopped"], "dups": st["dups"], "skips": st["skips"]})
    json.dump(sj, open(f"{ROOT}/sources.json", "w"), indent=1)
    json.dump({"xgap_sources_docs": ndocs, "dropped_in_assembly": dict(drops), "per_source": {k: dict(v) for k, v in per_src_drop.items()},
               "dups_at_build_vs_calib_public_xcorp": {x["id"]: x["dups"] for x in sj["xgap-v1"] if x["dups"]}}, open(f"{ROOT}/dedup.json", "w"), indent=1)
    summary(rows)
    print("manifest rows", len(rows), "xgap docs", len(kept), "holdout", len(hold), "dropped", dict(drops))


def summary(rows):
    def tab(keyf, title, rows):
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            a = agg[keyf(r)]; a[0] += 1; a[1] += r["tokens"]
        out = [f"### {title}", "", "| key | docs | tokens |", "|---|---:|---:|"]
        for k, (n, t) in sorted(agg.items(), key=lambda kv: str(kv[0])): out.append(f"| {k} | {n} | {t:,} |")
        return "\n".join(out) + "\n"
    xg = [r for r in rows if r["origin"] == "xgap-v1"]
    parts = [tab(lambda r: r["origin"], "by origin", rows), tab(lambda r: (r["origin"], r.get("cell") or "-", r["category"]), "origin x cell x category", rows)]
    bands = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows: bands[(r["origin"], r.get("cell") or "-")][r["length_band"]] += r["tokens"]
    hdr = "| origin / cell | " + " | ".join(BAND_ORDER) + " |\n|---|" + "---:|" * len(BAND_ORDER)
    parts.append("### tokens by origin x cell x length band\n\n" + hdr + "\n" + "\n".join(
        f"| {k[0]} / {k[1]} | " + " | ".join(f"{v.get(b, 0):,}" for b in BAND_ORDER) + " |" for k, v in sorted(bands.items())) + "\n")
    parts.append(tab(lambda r: (r.get("cell"), "holdout" if r["holdout"] else "train"), "xgap holdout by cell", xg))
    parts.append(tab(lambda r: (r.get("cell"), r.get("lang")), "xgap cell x lang", [r for r in xg if r.get("cell") in ("code-langs", "multilingual-text", "multilingual-chat")]))
    parts.append(tab(lambda r: (r["origin"], r.get("thinking"), r.get("has_system")), "chat docs: thinking x has_system", [r for r in rows if r["kind"] == "chat"]))
    parts.append(tab(lambda r: (r.get("length") or "-"), "xgap length tag (xl = whole session > 128k)", xg))
    open(f"{ROOT}/summary.md", "w").write("\n".join(parts))


if __name__ == "__main__":
    main()
