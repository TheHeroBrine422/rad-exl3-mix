# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""verify190.py PLAN.json WORK -- checks the built s190a / s190b goldens in WORK against draw190's plan (after build190.py).
Per golden (c, x, mix): the documents / pieces are the plan's (orig indices, as built), every document carries stratum and script,
gen rows = the golden's gen.u8 under each document's gen range, N share, N scripts, max script share, L gen; the mix holds the
parts' documents once each, with the part's rows. Prints JSON (counts and shares only; no text). Exit 1 on a mismatch."""
import collections, json, os, sys

import numpy as np

plan = json.load(open(sys.argv[1])); W = sys.argv[2]
bad = []; out = {}
for part in plan["c"] + plan["x"]:
    name = part["name"]; kind = "c" if part in plan["c"] else "x"
    g = f"{W}/golden-{name}-{kind}"
    m = json.load(open(f"{g}/kld.json")); D = m["docs"]; gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    ok_rows = len(gen) == m["rows"] and sum(d["tokens"] - 1 - d["score_from"] for d in D) == m["rows"]
    orig = [d["orig"] for d in D]
    if sorted(orig) != sorted(part["idx"]): bad.append(f"{g}: documents differ from the plan")
    if not ok_rows: bad.append(f"{g}: rows / gen.u8 size mismatch")
    if any("stratum" not in d or "script" not in d for d in D): bad.append(f"{g}: unlabelled documents")
    gN = collections.Counter(); gL = 0; sc = collections.Counter(); nN = 0
    for d in D:
        n = d["tokens"] - 1 - d["score_from"]
        gg = int(gen[d["row0"]:d["row0"] + n].sum())
        if d["stratum"] == "N": gN[d["script"]] += gg; nN += 1
        else: gL += gg
    tN = sum(gN.values())
    out[g.split("/")[-1]] = {"docs": len(D), "rows": m["rows"], "N_docs": nN, "gen_N": tN, "gen_L": gL, "N_share": round(tN / max(tN + gL, 1), 4),
                             "N_scripts": len(gN), "max_script_share_of_N": round(max(gN.values()) / tN, 4) if tN else 0.0,
                             "N_gen_by_script": dict(gN.most_common())}
mix = {}
for name in [p["name"] for p in plan["c"]]:
    g = f"{W}/golden-{name}"
    m = json.load(open(f"{g}/kld.json")); D = m["docs"]; gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    keys = collections.Counter((d["part"], d.get("orig")) for d in D)
    if any(v > 1 for v in keys.values()): bad.append(f"{g}: a document twice")
    if len(gen) != m["rows"]: bad.append(f"{g}: gen.u8 size")
    ref = {(d["part"], d.get("orig")) for part in plan["c"] + plan["x"] if part["name"] == name
           for d in [{"part": "c" if part in plan["c"] else "x", "orig": i} for i in part["idx"]]}
    if set(keys) != ref: bad.append(f"{g}: mix documents differ from the two parts")
    mix[name] = {"docs": len(D), "rows": m["rows"], "gen_rows": int(gen.sum()), "parts": m.get("goldmix", {}).get("parts", {})}
res = {"parts": out, "mix": mix, "bad": bad}
print(json.dumps(res, indent=1))
sys.exit(1 if bad else 0)
