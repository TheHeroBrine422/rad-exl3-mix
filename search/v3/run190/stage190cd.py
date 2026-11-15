# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""stage190cd.py LIVE_REG C_KLD C_DIR D_KLD D_DIR OUT_REG LIVE_DRIVER OUT_DRIVER -- the staged registry and the candidate driver config
for the s190c / s190d pair (#190-02, w176; RULES-cooldown amendments 5:15 / 5:40 AM CST). stage190.py's pattern, with:
  - the calib N documents of the pair may sit in the RETIRED slices s173a-e, s186a-f (5:40 amendment: s186b / d / e / f count as retired);
    any other clash of a pair document with a registry slice refuses (Latin documents must be fresh, the N documents of a non-retired
    slice are never used);
  - the x identity: the pair's x pieces are checked by xdoc (disjoint within the pair; the x strings are only counted, as in stage190).
  C_KLD / D_KLD: the mix golden's kld.json (a local copy); C_DIR / D_DIR: the golden's directory on ds8 (the paths the hosts read).
  OUT_REG: a copy of LIVE_REG plus s190c (status staged) and s190d (status staged, "role": "confirm"). Other keys unchanged.
  OUT_DRIVER: LIVE_DRIVER with confirm.pairs + [["s190c", "s190d"]] (nothing else changed).
Reads only; writes OUT_REG and OUT_DRIVER. The live registry and driver are never touched; the lead merges the s190c / s190d keys
(as SWITCH-190.md) and swaps the config."""
import json, os, sys, time

LIVE_REG, C_KLD, C_DIR, D_KLD, D_DIR, OUT_REG, LIVE_DRV, OUT_DRV = sys.argv[1:9]
RETIRED_OK = {"s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186b", "s186c", "s186d", "s186e", "s186f"}
NOTE = ("#190-02 (w176, 11-15, RULES-cooldown amendments 5:15 / 5:40 AM CST): the pair s190c (search) / s190d (confirm), built like "
        "s190a/b with reused N. c: calib merged fit + search docs; Latin 2x s186a / s186b's L gen, fresh, least used first; N documents "
        "(5 per part) from the retired slices (s173a-e, s186a-f), distinct within the pair; cut at 4,096. x: unused xc8k-b1 Latin pieces; "
        "N pieces from the unused b2 batches (11) and the retired slices' pieces (5:15 rule); distinct within and across the pair. Staged, not ready.")


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
    return e, docs, m


R = json.load(open(LIVE_REG))
assert "s190c" not in R["slices"] and "s190d" not in R["slices"], "s190c / s190d already in the live registry"
ec, dc, mc = entry(C_KLD, C_DIR); ed, dd, md = entry(D_KLD, D_DIR, role="confirm")
# calib ids: exact identity (the calib index). A pair document may clash only with a retired-ok slice, and only as an N document.
clash = []
for d in dc + dd:
    if d["part"] != "c": continue
    for k, s in R["slices"].items():
        for e in s["docs"]:
            if e["part"] == "c" and e["id"] == d["id"]:
                if d["stratum"] == "N" and k in RETIRED_OK: continue
                clash.append((d["id"], k, d["stratum"]))
if clash:
    raise SystemExit(f"calib doc ids in other slices (not allowed by the 5:40 rule): {sorted(clash)[:10]}")
c_ids = {d["id"] for d in dc if d["part"] == "c"}; d_ids = {d["id"] for d in dd if d["part"] == "c"}
assert not (c_ids & d_ids), "s190c and s190d share a calib document"
# x identity: the global xdoc, disjoint within the pair (the x strings are counted only: several pieces share a string across batches)
xdoc_c = {x["xdoc"] for x in mc["docs"] if x["part"] == "x"}; xdoc_d = {x["xdoc"] for x in md["docs"] if x["part"] == "x"}
assert not (xdoc_c & xdoc_d), "s190c and s190d share an x piece (xdoc)"
xs_live = {d["id"] for k, s in R["slices"].items() for d in s["docs"] if d["part"] == "x"}
x_strings = sum(1 for d in dc + dd if d["part"] == "x" and d["id"] in xs_live)
print(f"x doc ids that also occur as a string in other slices (different pieces, see the note in stage190.py): {x_strings}", file=sys.stderr)
R["slices"]["s190c"] = ec; R["slices"]["s190d"] = ed
tmp = OUT_REG + ".part"; json.dump(R, open(tmp, "w"), indent=1); os.replace(tmp, OUT_REG)
D = json.load(open(LIVE_DRV))
pairs = D["confirm"]["pairs"]
assert ["s190c", "s190d"] not in pairs
D["confirm"]["pairs"] = pairs + [["s190c", "s190d"]]
tmp = OUT_DRV + ".part"; json.dump(D, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, OUT_DRV)
summ = lambda dl: {"docs": len(dl), "c": sum(d["part"] == "c" for d in dl), "x": sum(d["part"] == "x" for d in dl),
                   "c_N": sum(1 for d in dl if d["part"] == "c" and d.get("stratum") == "N")}
print(json.dumps({"s190c": summ(dc), "s190d": summ(dd), "pairs": D["confirm"]["pairs"],
                  "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
