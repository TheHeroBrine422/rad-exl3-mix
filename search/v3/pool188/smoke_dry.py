# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""smoke_dry.py -- an end-to-end dry run of the search driver with the pool (#188): the live run's state, menu and fits (read
only), a copy of its registry with the staged s186 slices, and fabricated per-document results for every eval the driver
submits. Nothing is evaluated and nothing outside --work is written. It exercises ingest, the early-stop reference
(the worst member), the join rule (add / replace), the re-seed on s186a, the rotation rescore, the re-selection and the
candidate kinds (cross included) on the real menu and fits.

  python3 -I smoke_dry.py --tree TREE --live RUNDIR --staged registry-staged-186.json --work DIR [--steps 60] [--pool 3]

RUNDIR: the live run dir (driver.json, registry.json, menu.json, drv/state.jsonl, drv/assign, drv/frontier.json, drv/fits)."""
import argparse, json, math, os, random, shutil, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smoke_dry.py")
    ap.add_argument("--tree", required=True); ap.add_argument("--live", required=True); ap.add_argument("--staged", required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--pool", type=int, default=3)
    ap.add_argument("--seed", type=int, default=188); ap.add_argument("--depth", type=int, default=6, help="CONF queue_depth (the live run: 22)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.abspath(a.tree), "search", "v3")); import driver as DR     # noqa: E402
    live = os.path.abspath(a.live); W = os.path.abspath(a.work)
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(f"{W}/drv/assign"); os.makedirs(f"{W}/docs")
    for f in os.listdir(f"{live}/drv/assign"): shutil.copy(f"{live}/drv/assign/{f}", f"{W}/drv/assign/{f}")
    shutil.copy(f"{live}/drv/state.jsonl", f"{W}/drv/state.jsonl"); shutil.copy(f"{live}/drv/frontier.json", f"{W}/drv/frontier.json")
    open(f"{W}/results.jsonl", "w").close(); open(f"{W}/evals.jsonl", "w").close()
    reg = json.load(open(f"{live}/registry.json")); staged = json.load(open(a.staged))
    for k in [k for k in staged["slices"] if k.startswith("s186")]:
        reg["slices"][k] = dict(staged["slices"][k], status="ready")
    for k in ("s173e", "s173f"): reg["slices"][k]["status"] = "retired"            # the rotation goes to s186a
    json.dump(reg, open(f"{W}/registry.json", "w"))
    conf = json.load(open(f"{live}/driver.json"))
    conf.update(root=f"{W}/drv", registry=f"{W}/registry.json", evals=f"{W}/evals.jsonl", results=f"{W}/results.jsonl",
                docs=f"{W}/docs", menu=f"{live}/menu.json", menu_remote=None, pool=a.pool, cap=None, refit_every=10 ** 9,
                fit_cmd=["true"], fit_retry_s=10 ** 9, queue_depth=a.depth, rotate_after=40, go=True,
                reseed={"slice": "s186a", "sizes": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0], "top": 3, "source": f"{W}/drv/frontier.json"},
                cross_share0=0.15, monitor={"name": "ver3-8k", "golden": "none", "when": "never"})
    json.dump(conf, open(f"{W}/conf.json", "w"), indent=1)
    rng = random.Random(a.seed)
    D = DR.Driver(f"{W}/conf.json")
    D.res_off = 0; D.res_lines = 0                                                # the live results file's offset is not ours
    u = {}                                                                        # a fixed per-(slice, document) weight
    done_lines = 0; sizes_seen = set(); outcomes = {"stopped": 0, "whole": 0}
    for step in range(a.steps):
        D.step()
        lines = open(f"{W}/evals.jsonl").read().splitlines()
        with open(f"{W}/results.jsonl", "a") as rf:
            for l in lines[done_lines:]:
                e = json.loads(l); t = e["tag"]; sl = e["slices"][0]["name"]; N = len(D.reg.d["slices"][sl]["docs"])
                if sl not in u: u[sl] = [1.0 + 0.5 * rng.random() for _ in range(N)]
                kind = t.split("-")[2] if t.count("-") >= 2 else ""
                mu = {"inc": 0.0, "init": 0.0, "reseed": -0.01, "cross": -0.015}.get(kind, 0.0)
                fac = math.exp(rng.gauss(mu, 0.04))
                stop = bool(e.get("early_stop")) and rng.random() < 0.5
                docs = []; stopped = None
                for i in range(N):
                    if stop and i >= 4: docs.append(None); continue
                    docs.append([u[sl][i] * fac * (1 + 0.02 * rng.random()), 1000])
                if stop: stopped = {"at": 4}; outcomes["stopped"] += 1
                else: outcomes["whole"] += 1
                json.dump({"m": docs}, open(f"{W}/docs/{t}-{sl}.json", "w"))
                kld = sum(x[0] for x in docs if x) / max(sum(1 for x in docs if x), 1)
                rf.write(json.dumps({"tag": t, "slice": sl, "kld": kld, "bpw": D.sub[t]["bpw"], "bytes": None, "stopped": stopped}) + "\n")
                sizes_seen.add(D.sub[t]["size"])
        done_lines = len(lines)
        D.ingest()
    pool = {f"{z:g}": len(D.members(z)) for z in D.sizes}
    kinds = {}
    for e in D.sub.values(): kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    ev = [json.loads(l) for l in open(f"{W}/drv/state.jsonl")]
    cnt = {}
    for e in ev: cnt[e["ev"]] = cnt.get(e["ev"], 0) + 1
    res = {"steps": a.steps, "active": D.reg.active(), "submits_by_kind": kinds, "events": cnt, "pool_sizes": pool, "outcomes": outcomes,
           "incumbents": {f"{z:g}": D.inc.get(z) for z in D.sizes}, "rotations": [e for e in ev if e["ev"] == "rotate"],
           "reseed_events": [{k: e[k] for k in e if k in ("size", "inc", "pool", "prev")} for e in ev if e["ev"] == "reseed"],
           "reselect_events": len([e for e in ev if e["ev"] == "reselect"]), "joins": [{k: e[k] for k in ("size", "join", "replaced", "tag") if k in e} for e in ev if e["ev"] == "pool" and e.get("why") == "join"][:40]}
    json.dump(res, open(f"{W}/smoke-summary.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("active", "events", "pool_sizes", "outcomes")}, indent=None))
    print(f"reseed sizes {len(res['reseed_events'])}, reselect events {res['reselect_events']}, joins {len(res['joins'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
