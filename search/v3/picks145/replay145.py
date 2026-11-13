#!/usr/bin/env python3
"""replay145.py STATE.jsonl REGISTRY.json DRIVER_CONF OUT.json -- RULES-picks' in-search rule replayed on the #171 run
(w145; report only, nothing decided).

Every #171 decision (state.jsonl, in order) is redone against the incumbent the driver compared it with (the size's
incumbent in force at the decision, the decision's recorded size offset), first-order only: the incumbent chain is
the one #171 actually had (a changed accept would have changed every later comparison of its size; not simulated).
  old = the #171 rule (seqrule pmean, raw sums, its boundary applied after the fact as for audit runs), recomputed
        (must reproduce the recorded decision)
  new = (b) capped at 20 % of each part's whole-slice incumbent KL + the (c) per-document sign guard, with the run's
        early-stop boundary (B p2 w5): a whole-slice result that the capped statistic would have stopped is a reject;
        a run #171 stopped early is "undetermined" when the capped statistic would not have stopped within the
        documents it has (the rest was never scored), else a reject."""
import collections, json, sys

sys.path.insert(0, "/workspace/rad-exl3-mix/search/v3")
import seqrule as SR  # noqa: E402

STATE, REG, CONF, OUT = sys.argv[1:5]
S = [json.loads(l) for l in open(STATE)]
reg = json.load(open(REG))["slices"]; conf = json.load(open(CONF))
rule = dict(conf.get("rule") or {"kind": "curve", "p": 2.0, "warmup": 0.05})
if "warmup" in conf: rule["warmup"] = float(conf["warmup"])
res = {e["tag"]: e["r"] for e in S if e["ev"] == "result"}
sub = {e["tag"]: e for e in S if e["ev"] == "submit"}
inc = {}; rows = []
for e in S:
    if e["ev"] == "incumbent": inc[e["size"]] = e["tag"]; continue
    if e["ev"] != "decision" or e["decision"] == "stale": continue
    t = e["tag"]; r = res[t]; ref = inc.get(e["size"]); rr = res.get(ref)
    if rr is None or rr["slice"] != r["slice"]: rows.append({"tag": t, "skip": "no incumbent on the slice"}); continue
    parts = [d["part"] for d in reg[r["slice"]]["docs"]]; off = e.get("off", 0.0) or 0.0
    whole = not r.get("stopped") and all(x is not None for x in r["docs"])

    def go(rl, stop):
        P = SR.Paired(len(parts), dict(rl, off=off))
        for a, b, p in zip(rr["docs"], r["docs"], parts):
            if b is None: break
            P.add(a, b, p)
            if stop and P.check(): break
        return P
    Po = go({"kind": "full"}, False)                                     # the raw (a) per part, for the table
    Pr = go(dict(rule), True)                                            # the #171 rule with its boundary (audit runs: after the fact)
    old = "reject" if (not whole or Pr.stopped) else Pr.final()
    w = SR.cap_weights(rr["docs"], parts, 0.2)
    rn = dict(rule, w=w, guard="docmean")
    Pn = go(rn, True)
    if Pn.stopped: new, why = "reject", "early stop"
    elif not whole: new, why = "undetermined", "stopped by #171, the capped rule would not have stopped yet"
    else:
        new = Pn.final(); why = Pn.why or ("" if new == "accept" else "(b) not negative on every part")
    Pg = go(dict(rule, w=w), True)                                       # (b) without the guard
    nog = "reject" if Pg.stopped else ("undetermined" if not whole else Pg.final())
    sm = Pn.summary()["parts"]
    rows.append({"tag": t, "size": e["size"], "kind": e.get("kind"), "slice": r["slice"], "ref": ref, "recorded": e["decision"], "old": old,
                 "new": new, "new_noguard": nog, "why": why, "whole": whole, "k": Pn.k, "off": off,
                 "parts": {p: {"a": Po.summary()["parts"][p]["logr"], "b": v["logr"], "c": v["docmean"]} for p, v in sm.items()}})
ok = [x for x in rows if "skip" not in x]
mism = [x for x in ok if x["old"] != x["recorded"]]
acc = [x for x in ok if x["recorded"] == "accept"]
summ = {"decisions": len(ok), "skipped": len(rows) - len(ok), "old_reproduced": len(ok) - len(mism),
        "accepts": len(acc),
        "accepts_new": dict(collections.Counter(x["new"] + (f" ({x['why']})" if x["new"] != "accept" else "") for x in acc)),
        "accepts_new_noguard": dict(collections.Counter(x["new_noguard"] for x in acc)),
        "rejects_whole": sum(1 for x in ok if x["recorded"] == "reject" and x["whole"]),
        "rejects_whole_new_accept": [x["tag"] for x in ok if x["recorded"] == "reject" and x["whole"] and x["new"] == "accept"],
        "rejects_stopped": sum(1 for x in ok if x["recorded"] == "reject" and not x["whole"]),
        "rejects_stopped_new": dict(collections.Counter(x["new"] for x in ok if x["recorded"] == "reject" and not x["whole"])),
        "accepts_changed_by_size": dict(collections.Counter(str(x["size"]) for x in acc if x["new"] != "accept"))}
json.dump({"summary": summ, "mismatch": mism, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summ, indent=1))
