# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""stage195.py LIVE_REG KLD_DIR DS8_SLICES OUT_REG -- the staged registry for the 15 slices of RULES-slices195 (#195, w180; the pattern of
stage190ef.py). KLD_DIR holds a local copy of each mix golden's kld.json as golden-<name>.kld.json; DS8_SLICES is the goldens' directory
on ds8 / box2 (the path the hosts read). OUT_REG = a copy of LIVE_REG plus s195s1-5 (status staged) and s195c1-10 (status staged, role
confirm), each with "pair": the slices it is paired with (RULES-slices195 §2). No other key changes. A calib document of a new slice may
also sit in a retired registry slice (RETIRED_OK counts as retired); any clash with another registry slice refuses. Reads only; writes
OUT_REG. The live registry is never touched; the lead merges the keys."""
import json, os, sys, time

LIVE_REG, KD, SL, OUT_REG = sys.argv[1:5]
RETIRED_OK = {"s186b", "s186d", "s186e", "s186f", "s190b"}
SEARCH = [f"s195s{i}" for i in range(1, 6)]; CONFIRM = [f"s195c{k}" for k in range(1, 11)]
PAIRS = [[f"s195s{(k - 1) % 5 + 1}", f"s195c{k}"] for k in range(1, 11)]
NOTE = ("#195 (w180, 11-15, RULES-slices195 §2): {role} slice of the 15-slice set (5 cycling search s195s1-5, 10 one-use confirm s195c1-10; "
        "search s195s((k-1) mod 5 + 1) pairs with confirm s195ck). 2 x {match}'s Latin gen in total, ~10 % of gen non-Latin (10 scripts, all "
        "from the fresh batch xn8k-b195, each piece once). c: Latin calib documents of the merged fit + search pool, cut at 4,096, an equal "
        "share of the allowed pool ({cdesc}); the pool cannot give 2 x c parts, the shortfall is xcorp Latin in the x part. x Latin: {xdesc}. "
        "Staged, not ready.")


def doc_id(d):
    if d["part"] == "c": return f"c:{d.get('orig', d['src'])}"
    return f"x:{d.get('source', '')}:{d.get('orig', d['src'])}"


R = json.load(open(LIVE_REG)); new = {}; ms = {}
for n in SEARCH + CONFIRM:
    assert n not in R["slices"], f"{n} already in the live registry"
    m = json.load(open(f"{KD}/golden-{n}.kld.json")); ms[n] = m; conf = n in CONFIRM
    docs = [dict({"part": d["part"], "src": d["src"], "id": doc_id(d), "tokens": d["tokens"], "rows": d["tokens"] - 1 - d["score_from"]},
                 **({"stratum": d["stratum"]} if "stratum" in d else {})) for d in m["docs"]]
    mix = f"{SL}/golden-{n}"
    e = {"parts": m["goldmix"]["parts"], "mix": mix, "host_dirs": {}, "docs": docs, "status": "staged", "decisions": {},
         "built": m["goldmix"]["built"], "alias": mix,
         "pair": [p[0] for p in PAIRS if p[1] == n] if conf else [p[1] for p in PAIRS if p[0] == n],
         "note": NOTE.format(role="confirm" if conf else "search", match="s186" + "abcde"[(int(n[5:]) - 1) % 5],
                             cdesc="the least used two thirds, the freshest" if conf else "the more used third: retired search / V2-era documents",
                             xdesc="fresh pieces (xc8k-b1, b2-c0..c4; in no earlier golden)" if conf else
                             "pieces of retired slices (s171a-d, s173a-e, s186a-f, s190a-c, search27 / s129), least used first")}
    if conf: e["role"] = "confirm"
    new[n] = e
clash = []; seen_c = {}; seen_x = {}
for n, e in new.items():
    for d, md in zip(e["docs"], ms[n]["docs"]):
        if d["part"] == "c":
            assert d["id"] not in seen_c, (d["id"], n, seen_c[d["id"]]); seen_c[d["id"]] = n
            for k, s in R["slices"].items():
                if s["status"] == "retired" or k in RETIRED_OK: continue
                if any(x["part"] == "c" and x["id"] == d["id"] for x in s["docs"]): clash.append((d["id"], n, k))
        else:
            assert md["xdoc"] not in seen_x, (md["xdoc"], n, seen_x[md["xdoc"]]); seen_x[md["xdoc"]] = n
if clash: raise SystemExit(f"calib documents held by a non-retired registry slice: {sorted(clash)[:10]}")
x_live = {d["id"] for s in R["slices"].values() for d in s["docs"] if d["part"] == "x"}
x_strings = sum(1 for e in new.values() for d in e["docs"] if d["part"] == "x" and d["id"] in x_live)
R["slices"].update(new)
tmp = OUT_REG + ".part"; json.dump(R, open(tmp, "w"), indent=1); os.replace(tmp, OUT_REG)
print(json.dumps({"staged": {n: {"docs": len(e["docs"]), "c": sum(d["part"] == "c" for d in e["docs"]), "x": sum(d["part"] == "x" for d in e["docs"]),
                                 "N": sum(d.get("stratum") == "N" for d in e["docs"]), "role": e.get("role", "search"), "pair": e["pair"]} for n, e in new.items()},
                  "pairs": PAIRS, "calib_docs": len(seen_c), "x_pieces": len(seen_x),
                  "x_id_strings_shared_with_other_slices": x_strings, "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
