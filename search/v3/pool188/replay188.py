# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""replay188.py -- the pool's sanity check on the recorded #183 history (#188, RULES-pool §7). Not a gate.

Per slice X and size s, the whole-slice results on X (stopped runs carry only a prefix, so they are left out; so are
monitor reads and the candidates whose live decision was stale) are replayed in result order with the join rule of
RULES-pool §3 (poolrule.py) and the snapshot's own statistic (cap 0.2 per part, the (c) guard, the recorded size offsets
b (bpw_c - bpw_ref) from each candidate's submit, the snapshot's stratification if any). Start: the size's incumbent
rescored on X (its first whole result of kind "incumbent" on X).

  K = 1 : must reproduce the live accepts among those candidates (the simulator's own check, reported);
  K = 3 : the pool. Reported: joins, the joins that were not the incumbent at their join ("pool members") and how many of
          them later became the incumbent on the same slice; promotions (any member that became incumbent later).
Uncertainty: bootstrap over slices (2,000 resamples, seed 188).

  python3 -I replay188.py --snap SNAP --out DIR     (writes DIR/summary.json and DIR/per_slice.csv)"""
import argparse, collections, csv, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, V3)
import poolrule as PR   # noqa: E402
import seqrule as SR    # noqa: E402

CAP = 0.2


def pair(ref, cand, parts, strata, off, cap=CAP):
    rule = {"kind": "full", "off": off, "guard": "docmean"}
    if cap: rule["w"] = SR.cap_weights(ref["docs"], parts, cap, strata=strata)
    if strata is not None: rule["strata"] = True
    P = SR.Paired(len(parts), rule)
    for i, (a, b) in enumerate(zip(ref["docs"], cand["docs"])):
        P.add(a, b, parts[i], strata[i] if strata is not None else None)
    return P


RULE = {"kind": "curve", "p": 2.0, "warmup": 0.05}      # driver.json "rule" of the #183 run (boundary's b0 = 2, t0 = 0.1)


def audit_stopped(t, res, sub, parts, strata):
    """an audit run is decided by the rule after the fact (driver.on_result): a stop against its submit-time reference
    (the prefix statistic with the reference's cap weights and the submit's size offset) is a reject"""
    ref = sub[t]["ref"]; off = float(sub[t].get("b") or 0.0) * (sub[t]["bpw"] - sub[ref]["bpw"])
    rule = dict(RULE, off=off, guard="docmean", w=SR.cap_weights(res[ref]["docs"], parts, CAP, strata=strata))
    if strata is not None: rule["strata"] = True
    P = SR.Paired(len(parts), rule)
    for i, (a, b) in enumerate(zip(res[ref]["docs"], res[t]["docs"])):
        P.add(a, b, parts[i], strata[i] if strata is not None else None)
        if P.check(): break
    return P.stopped == "reject"


def simulate(K, seq, res, sub, parts, strata, b_of):
    """seq[0]: the size's incumbent rescored on the slice; seq[1:]: the whole-slice candidates in result order (live
    decision non-stale). The driver's join rule (RULES-pool §3) with pool size K. Returns the counters, the final members and
    incumbent, and the end-of-slice reselection (RULES-pool §5 applied to this slice's own scores: in-sample, see the header)."""
    c = collections.Counter(); accepts = set(); joiners = set()
    inc = seq[0]; members = [inc]
    for t in seq[1:]:
        full = len(members) >= K
        W = inc
        if full:
            da = {m: (0.0 if m == inc else pair(res[inc], res[m], parts, strata, b_of(m) * (sub[m]["bpw"] - sub[inc]["bpw"])).da())
                  for m in members}
            W = PR.worst_of(inc, da)
        acc_i = pair(res[inc], res[t], parts, strata, b_of(t) * (sub[t]["bpw"] - sub[inc]["bpw"])).final() == "accept"
        if W == inc: acc_w = acc_i
        else: acc_w = pair(res[W], res[t], parts, strata, b_of(t) * (sub[t]["bpw"] - sub[W]["bpw"])).final() == "accept"
        if K == 1: mode, new_inc = ("replace", True) if acc_i else (None, False)   # K = 1: the worst is the incumbent
        else: mode, new_inc = PR.join_action(full, acc_i, acc_w)
        if mode is None: continue
        c["joins"] += 1
        members = [t if m == W else m for m in members] if mode == "replace" else members + [t]
        if new_inc:
            accepts.add(t); c["inc_changes"] += 1; joiners.discard(t); inc = t
        else:
            joiners.add(t)
    c["pool_members"] = len(joiners)                                                  # joiners that were not the incumbent when they joined
    # end-of-slice reselection on this slice's scores (in-sample)
    Ps = {m: pair(res[inc], res[m], parts, strata, b_of(m) * (sub[m]["bpw"] - sub[inc]["bpw"])) for m in members if m != inc}
    scored = [(m, Ps[m].da(), Ps[m].final() == "accept") for m in Ps]
    if scored:
        best = min(scored, key=lambda x: x[1])
        c["best_guard_fail"] = int(best[1] < 0 and not best[2] and Ps[best[0]].why == "guard")
        c["best_parts_fail"] = int(best[1] < 0 and not best[2] and Ps[best[0]].why != "guard")
    new = PR.reselect_pick(inc, scored) if K > 1 else None
    c["best_da"] = min((x[1] for x in scored), default=None)                       # the best member's statistic vs the final incumbent
    c["promoted"] = 1 if new is not None else 0
    c["promoted_was_pool_member"] = 1 if new is not None and new in joiners else 0
    c["promoted_was_live_rejected"] = 0                                               # filled by the caller (needs the decisions)
    c["best_da"] = c.get("best_da")
    return c, accepts, members, inc, new


def clopper(k, n, alpha=0.05):
    """exact (Clopper-Pearson) interval for k of n, by bisection on the binomial tails"""
    from math import comb
    def cdf(x, p): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
    def bis(f, lo, hi):
        for _ in range(80):
            m = (lo + hi) / 2
            if f(m): lo = m
            else: hi = m
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else bis(lambda p: (1 - cdf(k - 1, p)) < alpha / 2, 0.0, 1.0)
    hi = 1.0 if k == n else bis(lambda p: cdf(k, p) > alpha / 2, 0.0, 1.0)
    return lo, hi


def summarize(rows, boot=2000, seed=188):
    """pool members (non-incumbent joiners), promotions at the slice end (in-sample), members that beat the final
    incumbent on the statistic but were not promoted (guard or parts), with exact intervals and a bootstrap over slices"""
    def tot(sel):
        m = sum(int(r["k3_pool_members"]) for r in sel); p = sum(int(r["k3_promoted"]) for r in sel)
        neg = sum(1 for r in sel if r["k3_best_member_da"] is not None and float(r["k3_best_member_da"]) < 0)
        return m, p, neg
    m, p, neg = tot(rows)
    slices = sorted({r["slice"] for r in rows}); rng = random.Random(seed); bs = []
    for _ in range(boot):
        draw = [rng.choice(slices) for _ in slices]
        sel = [r for x in draw for r in rows if r["slice"] == x]
        mm, pp, _ = tot(sel)
        if mm: bs.append(pp / mm)
    bs.sort()
    return {"slices": len(slices), "slice_size_rows": len(rows), "candidates": sum(int(r["candidates"]) for r in rows),
            "live_accepts": sum(int(r["live_accepts"]) for r in rows), "k1_matches_live_rows": sum(int(r["k1_match"]) for r in rows),
            "k1_accepts": sum(int(r["k1_accepts"]) for r in rows), "k3_joins": sum(int(r["k3_joins"]) for r in rows),
            "k3_inc_changes": sum(int(r["k3_inc_changes"]) for r in rows), "audit_rule_stops": sum(int(r["audit_rule_stops"]) for r in rows),
            "pool_members": m, "promoted_in_sample": p, "best_member_guard_fail": sum(int(r["k3_best_guard_fail"]) for r in rows),
            "best_member_parts_fail": sum(int(r["k3_best_parts_fail"]) for r in rows), "promoted_rate_ci95_exact": clopper(p, m) if m else None,
            "promoted_rate_bootstrap_slices_2000": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None,
            "members_beating_final_inc_on_statistic": neg, "beating_rate_ci95_exact": clopper(neg, m) if m else None,
            "rule": "RULES-pool §7: whole-slice results, K = 3 join rule, end-of-slice reselection on the slice's own scores (in-sample)"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="replay188.py")
    ap.add_argument("--snap", required=True); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    snap = os.path.abspath(a.snap); os.makedirs(a.out, exist_ok=True)
    ev = [json.loads(l) for l in open(f"{snap}/drv/state.jsonl")]
    reg = json.load(open(f"{snap}/registry.json"))["slices"]
    sub, res, dec, rot = {}, {}, {}, []
    res_order = []
    for e in ev:
        if e["ev"] == "submit": sub[e["tag"]] = e
        elif e["ev"] == "result": res[e["tag"]] = e["r"]; res_order.append(e["tag"])
        elif e["ev"] == "decision": dec[e["tag"]] = e
        elif e["ev"] == "rotate": rot.append(e["to"])
    kinds_monitor = {"monitor", "monitor_ref"}    # (kept out of the candidates)
    slices = list(dict.fromkeys(rot))
    rows = []
    def b_of_tag(t): return float(sub[t].get("b") or 0.0)     # the recorded b of each submit (the live rule's offset)
    for X in slices:
        parts = [d["part"] for d in reg[X]["docs"]]
        strata = [d.get("stratum") for d in reg[X]["docs"]]
        strata = strata if any(s is not None for s in strata) else None
        whole = []
        for t in res_order:
            r = res[t]
            if r.get("slice") != X or r.get("stopped") or r.get("fail") or t not in sub: continue
            k = sub[t]["kind"]
            if k in kinds_monitor: continue
            if k != "incumbent" and (t not in dec or dec[t].get("decision") == "stale"): continue
            whole.append(t)
        for size in sorted({sub[t]["size"] for t in whole}):
            L = [t for t in whole if sub[t]["size"] == size]
            inc0 = [t for t in L if sub[t]["kind"] == "incumbent"]
            if not inc0: continue
            seq = [inc0[0]] + [t for t in L if t != inc0[0] and sub[t]["kind"] != "incumbent"]
            rule_stop = {t for t in seq[1:] if sub[t].get("audit") and sub[t].get("ref") in res and audit_stopped(t, res, sub, parts, strata)}
            seq = [seq[0]] + [t for t in seq[1:] if t not in rule_stop]
            live_acc = {t for t in seq[1:] if dec.get(t, {}).get("decision") == "accept"}
            # the simulator's own state: b per tag
            b_of = b_of_tag
            o1, acc1, _, _, _ = simulate(1, seq, res, sub, parts, strata, b_of)
            o3, acc3, mem3, inc3, new3 = simulate(3, seq, res, sub, parts, strata, b_of)
            live_rej = {t for t in seq[1:] if dec.get(t, {}).get("decision") == "reject"}
            row = {"slice": X, "size": size, "candidates": len(seq) - 1, "live_accepts": len(live_acc),
                   "k1_accepts": len(acc1), "k1_match": int(acc1 == live_acc),
                   "k3_joins": o3["joins"], "k3_pool_members": o3["pool_members"], "k3_inc_changes": o3["inc_changes"],
                   "k3_promoted": o3["promoted"], "k3_promoted_pool_member": o3["promoted_was_pool_member"],
                   "k3_promoted_live_rejected": int(new3 is not None and new3 in live_rej),
                   "audit_rule_stops": len(rule_stop), "k3_best_member_da": o3["best_da"],
                   "k3_best_guard_fail": o3.get("best_guard_fail", 0), "k3_best_parts_fail": o3.get("best_parts_fail", 0)}
            rows.append(row)
    json.dump(rows, open(f"{a.out}/rows.json", "w"), indent=0)
    with open(f"{a.out}/per_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = summarize(rows, a.boot)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    print(f"{len(rows)} (slice, size) rows; K=1 matches live on {summ['k1_matches_live_rows']}; pool members {summ['pool_members']}, "
          f"promoted (in-sample) {summ['promoted_in_sample']}; beating the final incumbent on the statistic {summ['members_beating_final_inc_on_statistic']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
