#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_driver_act.py SEARCH_V3_DIR WORKDIR -- stand-in check of the driver half of hostd-driver-a8.patch (w132, #172).
The w126 stand-in (standin.py: synthetic menu, known truth, fake executor) with an A8 executor: an eval whose mix has
"act" scores KL x f_A8(mix) (a size-dependent factor, 0.97 - 0.02 (bpw - 2) / 3, as if A8 helped more at high bpw), so
any comparison across modes is biased. Run: f16 until ~1/3 of the evals; then CONF "act": "a8g128" and a driver
restart mid-slice (as the switch would be done); go on past >= 2 more rotations. Checks:
  wait      after the restart, no eval carries "act" until the next rotation (the mode never changes mid-slice)
  switch    exactly one "act" state event, at a rotation; every eval submitted after it carries act a8g128 (hostd's input)
  rescore   right after the switch every incumbent is submitted on the new slice (whole, no early stop) in A8, and every
            incumbent gets an A8 monitor read
  paired    every decision compares a candidate with an incumbent scored in the same mode (submit "act" equal)
  monitor   monitor improvements are computed within one mode (no A8 read is compared with an f16 read)
  climb     the search still climbs after the switch (true KL of each size's incumbent falls; standin check 1)
  replay    a driver rebuilt from state.jsonl after the switch has the same mode and incumbents"""
import json, math, os, shutil, sys

V3 = os.path.abspath(sys.argv[1]); W = os.path.abspath(sys.argv[2]); sys.path.insert(0, V3)
import driver as DR          # noqa: E402
import menu as MN            # noqa: E402
import slices as SL          # noqa: E402
import standin as SI         # noqa: E402
import numpy as np           # noqa: E402

NRES = int(os.environ.get("NRES", "330"))


class A8Exec(SI.FakeExec):
    """scores like the stand-in; an "act" mix gets KL x f_A8 (per document, so the paired ratios move too)"""
    def score(self, e, M):
        super().score(e, M)
        if not e["mix"].get("act"): return
        tag = e["tag"]; sl = e["slices"][0]["name"]
        a = dict(e["mix"]["levels"])
        f = 0.97 - 0.02 * (M.bpw(a) - 2.0) / 3.0 if a else 0.97
        p = f"{self.root}/docs/{tag}-{sl}.json"; d = json.load(open(p))
        d["m"] = [[x[0] * f, x[1]] if x else None for x in d["m"]]; json.dump(d, open(p, "w"))
        L = open(f"{self.root}/results.jsonl").read().splitlines(); r = json.loads(L[-1]); assert r["tag"] == tag
        r["kld"] *= f; L[-1] = json.dumps(r); open(f"{self.root}/results.jsonl", "w").write("\n".join(L) + "\n")


def main():
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W); seed = 0
    levels = tuple(np.round(np.arange(1.0, 6.01, 0.25), 3))
    md = MN.synth(n_layers=16, levels=levels, seed=seed); json.dump(md, open(f"{W}/menu.json", "w"))
    M = MN.Menu(md); T = SI.Truth(M, seed + 1); rng = np.random.default_rng(seed + 2)
    reg = SL.Registry(f"{W}/registry.json"); S = {}
    for i in range(8):
        name = f"syn{i}"; docs = SI.make_slice(name, rng, T); S[name] = docs
        reg.add(name, f"/synthetic/{name}", parts={"x": "synx", "c": "sync"}, man={"docs": [dict(d, row0=0) for d in docs]})
        reg.d["slices"][name]["built"] = f"2026-11-12T00:00:{i:02d}Z"; reg.save()
    S["ver"] = SI.make_slice("ver", rng, T)
    sizes = [2.0, 3.0, 4.5]
    conf = {"root": f"{W}/run", "menu": f"{W}/menu.json", "registry": f"{W}/registry.json", "base": "BASE", "sizes": sizes, "size_tol": 0.03,
            "size_adjust": True, "evals": f"{W}/evals.jsonl", "results": f"{W}/results.jsonl", "docs": f"{W}/docs", "queue_depth": 4,
            "refit_every": 15, "rotate_after": 20, "audit_frac": 0.1, "seed": seed, "dry": True, "init_head": "H4", "init_emb": "E4",
            "ndraw": 8, "poll_s": 0, "fit_opts": {"beta": "fit"}, "monitor": {"name": "ver", "golden": "/synthetic/ver"}}
    json.dump(conf, open(f"{W}/conf.json", "w"), indent=1)
    D = DR.Driver(f"{W}/conf.json"); X = A8Exec(W, T, S, 2)
    first = {}; switched_at = None; steps = 0; replay_ok = None
    while X.n < NRES and steps < 20 * NRES:
        D.step(); X.step(M); steps += 1
        for fp in D.fitprocs.values():
            if fp: fp[0].wait()
        for s in sizes:
            if s not in first and D.inc.get(s) in D.res: first[s] = T.kl(D.assign(D.inc[s]))
        if switched_at is None and X.n >= NRES // 3:                        # the switch: CONF act + a restart mid-slice
            conf["act"] = "a8g128"; json.dump(conf, open(f"{W}/conf.json", "w"), indent=1)
            D = DR.Driver(f"{W}/conf.json"); switched_at = sum(1 for _ in open(conf["evals"]))
            n_rot_before = sum(1 for l in open(f"{W}/run/state.jsonl") if '"ev": "rotate"' in l)
        if switched_at is not None and replay_ok is None and D.act == "a8g128" and X.n >= 2 * NRES // 3:
            inc0, act0 = dict(D.inc), D.act; D = DR.Driver(f"{W}/conf.json"); replay_ok = D.inc == inc0 and D.act == act0
    D.ingest()
    st = [json.loads(l) for l in open(f"{W}/run/state.jsonl")]
    ev = [json.loads(l) for l in open(conf["evals"])]
    sub = {e["tag"]: e for e in st if e["ev"] == "submit"}
    ok = True

    def chk(name, cond, msg=""):
        nonlocal ok
        ok &= bool(cond); print(f"  {'PASS' if cond else 'FAIL'} {name}: {msg}")
    ia = [i for i, e in enumerate(st) if e["ev"] == "act"]
    iact = ia[0] if ia else None
    rot_i = [i for i, e in enumerate(st) if e["ev"] == "rotate"]
    # wait: evals between the restart and the act event carry no act
    tags_after_act = {e["tag"] for e in st[iact:] if e["ev"] == "submit"} if iact is not None else set()
    pre = [e for e in ev[switched_at:] if e["tag"] not in tags_after_act]
    chk("wait", len(pre) > 0 and all(not e["mix"].get("act") for e in pre), f"{len(pre)} evals submitted after the restart, before the rotation: none with act")
    chk("switch", len(ia) == 1 and iact - 1 in rot_i and all(e["mix"].get("act") == "a8g128" for e in ev if e["tag"] in tags_after_act),
        f"act events {len(ia)} (at state line {iact}, rotate at {[i for i in rot_i if i < (iact or 0)][-1:]}); {len(tags_after_act)} evals after it, all a8g128")
    new_slice = st[iact]["slice"] if iact is not None else None
    after = [e for e in st[iact:] if e["ev"] == "submit"] if iact is not None else []
    inc_new = [e for e in after if e["kind"] == "incumbent" and e["slice"] == new_slice]
    mon_new = [e for e in after if e["kind"] == "monitor" and e.get("act_rescore")]
    chk("rescore", len({e["size"] for e in inc_new}) == len(sizes) and all(not e["early"] and e["act"] == "a8g128" for e in inc_new)
        and len({e["size"] for e in mon_new}) == len(sizes) and all(e["act"] == "a8g128" for e in mon_new),
        f"incumbents rescored on {new_slice}: {sorted(e['size'] for e in inc_new)}, A8 monitor reads {sorted(e['size'] for e in mon_new)}")
    # paired: every decision's incumbent ref has the candidate's mode
    incs = {}; bad = 0; nd = 0
    for e in st:
        if e["ev"] == "incumbent": incs[e["size"]] = e["tag"]
        if e["ev"] == "decision" and e["decision"] in ("accept", "reject"):
            nd += 1; c = sub[e["tag"]]; ref = c.get("ref") or incs.get(e["size"])
            if ref and sub[ref].get("act", "f16") != c.get("act", "f16"): bad += 1
    chk("paired", nd > 0 and bad == 0, f"{nd} decisions, {bad} across modes")
    mons = [e for e in st if e["ev"] == "monitor"]
    mixed = 0
    for i, m in enumerate(mons):
        if m.get("improve") is None: continue
        prev = [p for p in mons[:i] if p["size"] == m["size"]]
        if not any(p.get("act", "f16") == m.get("act", "f16") for p in prev): mixed += 1
    chk("monitor", any(m.get("act") == "a8g128" for m in mons) and mixed == 0,
        f"{len(mons)} monitor reads ({sum(m.get('act') == 'a8g128' for m in mons)} A8); improvements computed across modes: {mixed}")
    gains = {s: math.log(T.kl(D.assign(D.inc[s])) / first[s]) for s in sizes}
    chk("climb", all(g < -0.02 for g in gains.values()), " ".join(f"{s}: {100 * g:+.1f}%" for s, g in gains.items()))
    chk("replay", replay_ok, "driver rebuilt from state after the switch: same mode and incumbents")
    print(D.report()); print("TEST_DRIVER_ACT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
