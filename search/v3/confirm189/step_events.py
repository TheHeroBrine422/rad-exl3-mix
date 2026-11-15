# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""step_events.py -- runs Driver.step() of one driver tree on a copy of a live run's state (#189-02, w168): the identity check
for the step paths (ensure_pool, re-seed, RULES-pool §5, go-live, confirmation, report) that replay_events.py (on_result only)
does not reach; and the dry run of CONF changes (--conf-set) on the live state.

  python3 -I step_events.py --tree TREE --snap SNAP --out OUT.json [--steps 3] [--conf-set KEY=JSON ..] [--role SLICE=ROLE ..]
                            [--reselect-now]

SNAP: driver.json, registry.json, menu.json, drv/state.jsonl, drv/assign/ (read only; copied to a temp dir). No new results
arrive (an empty results file); fits are not reloaded ("fit_done" events dropped from the copy: no generation, no fit
process); every queued eval goes to the copy's evals.jsonl. OUT: the events emitted (time stamps dropped), the queued
lines (tag, kind, slice), the log lines, and (--reselect-now) the re-selection plan built at once even while go-live is pending."""
import argparse, json, os, shutil, sys, tempfile


def main(argv=None):
    ap = argparse.ArgumentParser(prog="step_events.py")
    ap.add_argument("--tree", required=True); ap.add_argument("--snap", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--conf-set", action="append", default=[], metavar="KEY=JSON"); ap.add_argument("--role", action="append", default=[])
    ap.add_argument("--reselect-now", action="store_true", help="dry run: build the re-selection plan now (go-live may be pending)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.abspath(a.tree), "search", "v3"))
    import driver as DR     # noqa: E402  (the tree under test)

    snap = os.path.abspath(a.snap); tmp = tempfile.mkdtemp(prefix="step189-")
    conf = json.load(open(f"{snap}/driver.json")); reg = json.load(open(f"{snap}/registry.json"))
    for rr in a.role:
        n, r = rr.split("=", 1)
        if r in ("search", "-"): reg["slices"][n].pop("role", None)
        else: reg["slices"][n]["role"] = r
    json.dump(reg, open(f"{tmp}/registry.json", "w"))
    os.makedirs(f"{tmp}/drv/assign"); os.makedirs(f"{tmp}/docs")
    for f in os.listdir(f"{snap}/drv/assign"): shutil.copy(f"{snap}/drv/assign/{f}", f"{tmp}/drv/assign/{f}")
    with open(f"{tmp}/drv/state.jsonl", "w") as o:
        for l in open(f"{snap}/drv/state.jsonl"):
            if '"ev": "fit_done"' not in l: o.write(l)
    open(f"{tmp}/results.jsonl", "w").close()
    conf.update(root=f"{tmp}/drv", menu=f"{snap}/menu.json", registry=f"{tmp}/registry.json", evals=f"{tmp}/evals.jsonl",
                results=f"{tmp}/results.jsonl", docs=f"{tmp}/docs", min_rows_fit=10 ** 9, refit_every=10 ** 9, fit_cmd=["false"])
    for kv in a.conf_set:
        k, v = kv.split("=", 1); conf[k] = json.loads(v)
    json.dump(conf, open(f"{tmp}/conf.json", "w"))
    n0 = sum(1 for _ in open(f"{tmp}/drv/state.jsonl"))
    real = sys.stdout; sys.stdout = open(os.devnull, "w")
    try:
        D = DR.Driver(f"{tmp}/conf.json"); logs0 = sum(1 for _ in open(f"{tmp}/drv/driver.log")) if os.path.exists(f"{tmp}/drv/driver.log") else 0
        for _ in range(a.steps): D.step()
        plan = None
        if a.reselect_now and getattr(D, "rs", None) and D.rs_plan is None:
            D.reselect_start(D.reg.active()); plan = D.rs_plan
    finally:
        sys.stdout.close(); sys.stdout = real
    ev = [json.loads(l) for i, l in enumerate(open(f"{tmp}/drv/state.jsonl")) if i >= n0]
    for e in ev: e.pop("t", None)
    q = [json.loads(l) for l in open(f"{tmp}/evals.jsonl")] if os.path.exists(f"{tmp}/evals.jsonl") else []
    logs = [l.split(" CST ", 1)[-1].rstrip("\n") for l in list(open(f"{tmp}/drv/driver.log"))[logs0:]]
    out = {"events": ev, "queued": [{"tag": x["tag"], "slice": x["slices"][0]["name"], "early": "early_stop" in x} for x in q], "log": logs,
           "plan": plan, "status": open(f"{tmp}/drv/status.txt").read().strip() if os.path.exists(f"{tmp}/drv/status.txt") else None,
           "inc": {str(k): v for k, v in D.inc.items()}, "pool": {str(k): v for k, v in D.pool.items()}}
    json.dump(out, open(a.out, "w"), indent=0, default=str)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{a.out}: {len(ev)} events, {len(q)} queued")
    return 0


if __name__ == "__main__":
    sys.exit(main())
