# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""stage190.py LIVE_REG A_KLD A_DIR B_KLD B_DIR OUT_REG LIVE_DRIVER OUT_DRIVER -- the staged registry and the candidate driver config
for the s190a / s190b pair (#190, RULES-cooldown amendment 11-14 11:40 PM CST). Reads only; writes OUT_REG and OUT_DRIVER.
  A_KLD / B_KLD: the mix golden's kld.json (a local copy); A_DIR / B_DIR: the golden's directory on ds8 (the path the registry
           records as mix / alias / parts: the hosts read it there).
  OUT_REG: a copy of the live registry (LIVE_REG) plus s190a (status staged) and s190b (status staged, "role": "confirm"), with
           the doc ids as slices.py's Registry.add writes them (c:<orig>, x:<source>:<orig>). doc_decisions as in LIVE_REG.
           Refuses when a pair doc id already sits in another registry slice (the cross-check of the exclusion set).
  OUT_DRIVER: LIVE_DRIVER with confirm.pairs + [["s190a", "s190b"]] (nothing else changed).
The driver is never touched; the lead merges OUT_REG's s190 keys and swaps the config (SWITCH-190.md)."""
import json, os, sys, time

LIVE_REG, MIX_A, DIR_A, MIX_B, DIR_B, OUT_REG, LIVE_DRV, OUT_DRV = sys.argv[1:9]
NOTE = ("#190 RULES-cooldown amendment (w169, 11-14 11:40 PM CST): one 2x-Latin pair matched to s186a / s186b. "
        "c: calib merged fit + search docs (least used first; N = the 15 free non-Latin docs split between the pair), cut at 4,096; "
        "x: xc8k-b1 unused pieces (N split between the pair, L 2x s186's L gen); distinct within and across the pair. Staged, not ready.")


def doc_id(d):
    if d["part"] == "c": return f"c:{d.get('orig', d['src'])}"
    return f"x:{d.get('source', '')}:{d.get('orig', d['src'])}"


def entry(mix_kld, mixdir, role=None):
    m = json.load(open(mix_kld))
    docs = [dict({"part": d["part"], "src": d["src"], "id": doc_id(d), "tokens": d["tokens"], "rows": d["tokens"] - 1 - d["score_from"]},
                 **({"stratum": d["stratum"]} if "stratum" in d else {})) for d in m["docs"]]
    e = {"parts": m["goldmix"]["parts"], "mix": mixdir, "host_dirs": {}, "docs": docs, "status": "staged", "decisions": {},
         "built": m["goldmix"]["built"], "alias": mixdir, "note": NOTE}
    if role: e["role"] = role
    return e, docs


R = json.load(open(LIVE_REG))
assert not any(k.startswith("s190") for k in R["slices"]), "s190 keys already in the live registry"
ea, da = entry(MIX_A, DIR_A); eb, db = entry(MIX_B, DIR_B, role="confirm")
# c ids are the calib index (exact identity). x ids are "x:<source>:<orig>" with orig = the index in the piece's own xcorp batch, so
# two batches can give the same string for different pieces (live: s173c / s173e have x:fineweb-edu:474 for xdoc 41799 and
# s190a has the same string for xdoc 2281). The x identity is the global xdoc, checked by check190.py (exclusion()); here the x
# strings are only counted.
ids_new = {d["id"] for d in da + db if d["part"] == "c"}
clash = {(k, d["id"]) for k, s in R["slices"].items() for d in s["docs"] if d["id"] in ids_new}
if clash:
    raise SystemExit(f"calib doc ids already in other slices (exclusion cross-check failed): {sorted(clash)[:10]}")
assert len({d["id"] for d in da} & {d["id"] for d in db}) == 0, "s190a and s190b share a document (id)"
xs_live = {d["id"] for k, s in R["slices"].items() for d in s["docs"] if d["part"] == "x"}
x_strings = sum(1 for d in da + db if d["part"] == "x" and d["id"] in xs_live)
print(f"x doc ids that also occur as a string in other slices (different pieces, see note): {x_strings}", file=sys.stderr)
R["slices"]["s190a"] = ea; R["slices"]["s190b"] = eb
tmp = OUT_REG + ".part"; json.dump(R, open(tmp, "w"), indent=1); os.replace(tmp, OUT_REG)
D = json.load(open(LIVE_DRV))
pairs = D["confirm"]["pairs"]
assert ["s190a", "s190b"] not in pairs
D["confirm"]["pairs"] = pairs + [["s190a", "s190b"]]
tmp = OUT_DRV + ".part"; json.dump(D, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, OUT_DRV)
print(json.dumps({"s190a": {"docs": len(da), "c": sum(d["part"] == "c" for d in da), "x": sum(d["part"] == "x" for d in da)},
                  "s190b": {"docs": len(db), "c": sum(d["part"] == "c" for d in db), "x": sum(d["part"] == "x" for d in db)},
                  "pairs": D["confirm"]["pairs"], "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
