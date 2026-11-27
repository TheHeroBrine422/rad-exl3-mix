#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""testplan231.py -- w231: plan231.py + qad220 --plan231 on synthetic goldens (CPU, no model; runs on the VM). Sources: calib (2k-20k, cap 8k, pools
with a verify pool and qad220's fixed-eval pool), teacher cuts (8k-40k, only the last 25-50 % scored), long docs (10k-40k, cap 32k), w231 multi-cut sessions (branch units) and 4 deep traces
(60k-80k), V 8 x Q 1024, 2 ranks, budget 1M rows. Checks: data181.plan on the queues = the builder (asserted in plan231), single epoch, judges /
verify / fixed-eval docs absent, source shares within 3 points of the targets, traces start as scheduled on alternating ranks and end before the
budget's step, the carried bound held, and qad220's own build_data with --plan231 gives the same steps; w236: share_windows (a calib-heavy end), trace_hold. Exit 0 = all pass."""
import argparse, importlib.util, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D                                      # noqa: E402

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not ok: fails.append(name)


def golden(d, lens, sfrom, gen_frac=1.0, seed=0):
    """a golden's metadata with sparse token / row files (nrows / scored read only kld.json and the gen mask)"""
    os.makedirs(d, exist_ok=True); rg = np.random.default_rng(seed)
    docs, r0 = [], 0
    for n, s in zip(lens, sfrom): docs.append({"tokens": int(n), "score_from": int(s), "row0": r0}); r0 += int(n) - 1 - int(s)
    json.dump({"docs": docs, "top": 2, "rows": r0}, open(f"{d}/kld.json", "w"))
    for fn, sz in (("tokens.i32", 4 * sum(lens)), ("top.bin", 16 * r0), ("rows.f32", 12 * r0)):
        with open(f"{d}/{fn}", "wb") as f: f.truncate(sz)
    (rg.random(r0) < gen_frac).astype(np.uint8).tofile(f"{d}/gen.u8")
    return r0


def sessions(d, rg, ns=30):
    """w231: a cut golden with REAL tokens: ns sessions (history 10k-40k), 2-4 cuts each = history[:P] + a generated turn (1-3k, scored);
    plus one cut whose prefix differs from its history at one token (re-tokenised) and one cut shorter than 1024 tokens: both must stay separate docs"""
    os.makedirs(d, exist_ok=True); docs, toks, meta = [], [], []
    for s_ in range(ns):
        L = int(rg.integers(10000, 40000)); hist = rg.integers(1, 1000, L)
        Ps = sorted(rg.choice(np.arange(2000, L), int(rg.integers(2, 5)), replace=False))
        if s_ == 0: Ps = [600] + list(Ps)
        for k, P in enumerate(Ps):
            h = hist[:P].copy()
            if s_ == 1 and k == 0: h[P // 2] += 1
            g = rg.integers(1, 1000, int(rg.integers(1000, 3000))); t = np.concatenate([h, g])
            docs.append((len(t), int(P))); toks.append(t); meta.append({"session": f"s{s_}", "cut": k})
    r0, js = 0, []
    for n, P in docs: js.append({"tokens": n, "score_from": P, "row0": r0}); r0 += n - 1 - P
    json.dump({"docs": js, "top": 2, "rows": r0}, open(f"{d}/kld.json", "w")); np.concatenate(toks).astype(np.int32).tofile(f"{d}/tokens.i32")
    for fn, sz in (("top.bin", 16 * r0), ("rows.f32", 12 * r0)):
        with open(f"{d}/{fn}", "wb") as f: f.truncate(sz)
    np.ones(r0, np.uint8).tofile(f"{d}/gen.u8")
    with open(f"{d}/corpus.meta.jsonl", "w") as f: f.writelines(json.dumps(m) + "\n" for m in meta)
    return docs, meta


def main():
    T = tempfile.mkdtemp(); rg = np.random.default_rng(1)
    nc = 400; cl = rg.integers(2000, 20000, nc); golden(f"{T}/calib", cl, np.zeros(nc, int), 0.8, 1)
    pools = ["train214" if k % 5 == 0 else ["qad", "fit", "search", "verify"][k % 4] for k in range(nc)]
    json.dump({"pool": pools}, open(f"{T}/pools.json", "w"))
    judge = [k for k in range(nc) if k % 7 == 3]; json.dump(judge, open(f"{T}/judge.json", "w"))
    nt = 80; tl = rg.integers(8000, 40000, nt); gl = (tl * rg.uniform(0.25, 0.5, nt)).astype(int); golden(f"{T}/tset", tl, tl - gl, 1.0, 2)
    nx = 60; xl = rg.integers(10000, 40000, nx); golden(f"{T}/xgl", xl, np.zeros(nx, int), 1.0, 3)
    nd = 4; dl = rg.integers(60000, 80000, nd); golden(f"{T}/trd", dl, np.zeros(nd, int), 1.0, 4)
    sdocs, smeta = sessions(f"{T}/tses", rg)
    golden(f"{T}/xcs", rg.integers(500, 3000, 20), np.zeros(20, int), 1.0, 5)
    os.makedirs(f"{T}/gm", exist_ok=True); os.link(f"{T}/xcs/gen.u8", f"{T}/gm/xcs.u8")
    fe = {"pools": f"{T}/pools.json", "pool": "train214", "seed": 2181, "rows": 20000, "len": 2048}
    spec = {"V": 8, "Q": 1024, "ranks": 2, "seed": 0, "budget_rows": 1e6, "margin": 0.02, "max_carried_tokens": 200000, "long_tokens": 16384, "max_long": 2,
            "phases_rows": [3e5], "exclude_calib": [f"{T}/judge.json"], "fixed_eval": fe, "deep_first": 5,
            "sources": {"calib": {"golden": "calib", "dir": f"{T}/calib", "pools": f"{T}/pools.json", "pool": ["qad", "fit", "search", "train214"], "cap": 8192, "share": 0.45},
                        "teach": {"golden": "tset", "dir": f"{T}/tset", "share": 0.2},
                        "sess": {"golden": "tses", "dir": f"{T}/tses", "branch": True, "session_key": "session", "share": 0.15},
                        "long": {"golden": "xgl", "dir": f"{T}/xgl", "cap": 32768, "share": 0.2},
                        "traces": {"golden": "trd", "dir": f"{T}/trd", "deep": True}}}
    json.dump(spec, open(f"{T}/spec.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "plan231.py"), f"{T}/spec.json", f"{T}/plan.json", "--steps-tsv", f"{T}/steps.tsv"], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    check("plan231 builds (data181.plan on its queues = its simulation)", r.returncode == 0)
    if r.returncode: return finish()
    P = json.load(open(f"{T}/plan.json")); S = P["summary"]; items = P["items"]
    BR = {int(k): v for k, v in P.get("branch", {}).items()}
    keys = [(x[0], x[1]) for k, x in enumerate(items) if BR.get(k, [""])[0] != "hist"]
    check("single epoch: no document twice (session history pieces aside)", len(keys) == len(set(keys)))
    # w231 branch units
    used = {x[1] for k, x in enumerate(items) if x[0] == "tses" and BR.get(k, [""])[0] != "hist"}
    bad = []
    for q in P["queues"]:
        for j, it in enumerate(q):
            b = BR.get(it)
            if b is None: continue
            role, st, _, hdoc = b; doc = items[it][1]
            if role == "hist" and not (j + 1 < len(q) and BR.get(q[j + 1], [""])[0].startswith("cut") and BR[q[j + 1]][1] == items[it][2] and doc == hdoc): bad.append(("hist", it))
            if role.startswith("cut") and st != sdocs[doc][1] // 1024 * 1024: bad.append(("start", it))
            if role == "cut" and not (j + 1 < len(q) and q[j + 1] in BR): bad.append(("unit broken", it))
    sep = [k for k, x in enumerate(items) if x[0] == "tses" and k not in BR]
    sepdocs = {items[k][1] for k in sep}; tok = lambda i: smeta[i]
    check("branch units: each history piece ends where the next cut starts (same stream, in order), cuts start at their prefix rounded down to 1024",
          not bad and len(BR) > 0, f"{sum(1 for v in BR.values() if v[0] != 'hist')} cuts branched, {len(sep)} separate; problems {bad[:3]}")
    check("the re-tokenised cut and the < 1024-token-prefix cut stay separate full-prefix documents (if used)",
          all(not (smeta[i]["session"] == "s1" and smeta[i]["cut"] == 0) or i in sepdocs for i in used) and all(not (smeta[i]["session"] == "s0" and smeta[i]["cut"] == 0) or i in sepdocs for i in used),
          f"separate docs {sorted(sepdocs)}")
    bs = S["sources"]["sess"]["branch"]
    check("branch units cost less a row than full-prefix documents", S["sources"]["sess"]["cost_per_row"] < bs["cost_per_row_full_prefix"], json.dumps(S["sources"]["sess"]))
    sys.path.insert(0, HERE); import plan231 as PL
    fed = set(PL.fixed_eval_docs(fe, D.Gold(f"{T}/calib", "calib")))
    cal = {x[1] for x in items if x[0] == "calib"}
    check("no judge / verify / fixed-eval calib doc in the plan", not (cal & set(judge)) and not (cal & fed) and all(pools[i] != "verify" for i in cal), f"{len(cal)} calib docs, {len(fed)} fixed-eval")
    sh = S["share_rows"]; mixed = {k: v for k, v in sh.items() if k != "traces"}; tm = sum(mixed.values())
    dev = max(abs(mixed[k] / tm - spec["sources"][k]["share"]) for k in mixed)
    check("source shares (rows to the budget's step, traces aside) within 3 points of the targets", dev < 0.03, f"{sh}")
    ts = S["trace_starts"]; ps = S["planned_starts"]
    check("4 traces started, each at its planned step or later (a free stream / carried room), on both ranks", len(ts) == 4 and all(t >= p for (_, _, _, _, t), p in zip(ts, ps)) and {s % 2 for (_, _, _, s, _) in ts} == {0, 1}, f"planned {ps}, started {[(s, t) for (_, _, _, s, t) in ts]}")
    check("every trace ends before the budget's step", S["trace_ends_by_last"], f"last {S['last']}")
    check("carried bound held (no forced document over the bound)", S["forced_over_bound"] == 0 and S["carried_gib_per_rank"]["max"] <= spec["max_carried_tokens"] * 65536 / 2**30, json.dumps(S["carried_gib_per_rank"]))
    # the trainer's own build_data on the plan
    spec_ = importlib.util.spec_from_file_location("q220", os.path.join(HERE, "..", "qad220", "qad220.py")); Q = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(Q)
    a = Q.parser().parse_args([f"{T}/run", "--model", "-", "--level", "-", "--chunks", f"{T}/xcs", "--genmask", f"{T}/gm", "--calib", f"{T}/calib", "--pools", f"{T}/pools.json",
                               "--pool", "train214", "--fixed-eval", "20000", "--V", "8", "--Q", "1024", "--budgets", "1e6", "--plan231", f"{T}/plan.json"])
    os.makedirs(a.out, exist_ok=True)
    class D1:
        rank, world = 0, 2
        def barrier(self): pass
    B = Q.build_data(a, D1(), print)
    sim = D.plan([(0, x[1], x[2], x[3]) for x in items], None, 8, 1024, queues=P["queues"], starts={k: v[1] for k, v in BR.items() if v[1]})
    same_items = [(B["C"].g[gi].name, i, n, nr) for gi, i, n, nr in B["C"].items] == [tuple(x) for x in items]
    check("qad220 build_data --plan231: the plan's documents and steps", same_items and B["steps"] == sim, f"{len(B['steps'])} steps, last {B['last']}")
    check("qad220's budget step = the builder's", B["last"] == S["last"], f"{B['last']} vs {S['last']}")
    check("qad220 loads the branch roles (history / cut / last cut, starts, source docs)", {k: (v[0], v[1], v[3]) for k, v in B["C"].branch.items()} == {k: (v[0], v[1], v[3]) for k, v in BR.items()} and B["C"].starts == {k: v[1] for k, v in BR.items() if v[1]})
    share_windows(T, spec)
    trace_hold(T, spec, S)
    finish()


def trace_hold(T, spec, S0):
    """w236: "trace_hold": a due trace's rank takes no new long document while the trace waits: the plan builds (data181 = simulation), every trace
    starts and ends by the budget's step (the wait itself is reported, not judged: in this small plan it is a few steps either way; on the finals
    plan it cut the late traces' waits by 50-280 steps, w236 report)"""
    sp = json.loads(json.dumps(spec)); sp["trace_hold"] = True; sp["sources"]["traces"]["starts"] = S0["planned_starts"]
    json.dump(sp, open(f"{T}/spec-h.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "plan231.py"), f"{T}/spec-h.json", f"{T}/plan-h.json"], capture_output=True, text=True)
    check("plan231 builds with trace_hold", r.returncode == 0, r.stderr[-500:])
    if r.returncode: return
    S = json.load(open(f"{T}/plan-h.json"))["summary"]
    wait = lambda z: sum(t - p for (_, _, _, _, t), p in zip(z["trace_starts"], z["planned_starts"]))
    check("trace_hold: every trace starts and ends by the budget's step", len(S["trace_starts"]) == len(S0["trace_starts"]) and S["trace_ends_by_last"],
          f"wait {wait(S)} steps ({wait(S0)} without)")


def share_windows(T, spec):
    """w236: "share_windows": a calib-heavy end (calib 0.8 / long 0.2 from 0.75M assigned rows) after a calib-light start: the rows trained after 0.8M
    (to the budget's step) are mostly calib, the rows before 0.7M follow the sources' own shares, and the plan still builds and checks"""
    sp = json.loads(json.dumps(spec)); sp["share_windows"] = [[7.5e5, {"calib": 0.8, "long": 0.2}]]; sp["phases_rows"] = [7e5, 8e5]
    sp["sources"]["calib"]["share"] = 0.3; sp["sources"]["teach"]["share"] = 0.3
    json.dump(sp, open(f"{T}/spec-w.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "plan231.py"), f"{T}/spec-w.json", f"{T}/plan-w.json"], capture_output=True, text=True)
    check("plan231 builds with share_windows", r.returncode == 0, r.stderr[-500:])
    if r.returncode: return
    S = json.load(open(f"{T}/plan-w.json"))["summary"]; ph = S["rows_by_phase"]
    mix = [k for k in ph if k != "traces"]
    part = lambda p: {k: ph[k][p] / max(1, sum(ph[m][p] for m in mix)) for k in mix}
    a, z = part(0), part(2)
    check("share_windows: the end (after 0.8M rows) is >= 70 % calib, teacher / sessions ~0", z["calib"] >= 0.7 and z.get("teach", 0) + z.get("sess", 0) <= 0.05, f"end {json.dumps({k: round(v, 3) for k, v in z.items()})}")
    want = {k: sp["sources"][k]["share"] / sum(sp["sources"][m]["share"] for m in mix) for k in mix}
    check("share_windows: before 0.7M rows the sources' own shares (within 5 points; small synthetic plan, training order)", all(abs(a[k] - want[k]) < 0.05 for k in mix),
          f"start {json.dumps({k: round(v, 3) for k, v in a.items()})}, own {json.dumps({k: round(v, 3) for k, v in want.items()})}")


def finish():
    print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__": main()
