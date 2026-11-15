# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""replay_events.py -- replays a recorded #183 search state through one driver tree (#188, RULES-pool §8).

Inputs, in recorded order: submit (the driver's own submit table), result (-> Driver.on_result, the decision path),
rotate (registry statuses), b, act, bytes_offset, incumbent (set, not emitted: the ensure_* steps are not replayed).
Output: the events the replayed driver emits (decision, incumbent, pool, reseed, ...) and the recorded decisions, as one
JSON file. Two trees with the same output = the same decisions (identity, K = 1). Nothing is written under the snapshot.

  python3 -I replay_events.py --tree TREE --snap SNAP --out OUT.json [--pool K]

TREE: a directory with search/v3 and eval/ (and pipeline/ for the tests). SNAP: a live run's driver dir with driver.json,
registry.json, menu.json and drv/ (state.jsonl, assign/). The snapshot is read only."""
import argparse, hashlib, json, os, shutil, sys, tempfile


def main(argv=None):
    ap = argparse.ArgumentParser(prog="replay_events.py")
    ap.add_argument("--tree", required=True); ap.add_argument("--snap", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=None, help="CONF pool (default: the snapshot's, 1)")
    ap.add_argument("--conf-set", action="append", default=[], metavar="KEY=JSON", help="override a CONF key (JSON value)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.abspath(a.tree), "search", "v3"))
    import driver as DR     # noqa: E402  (the tree under test)

    snap = os.path.abspath(a.snap); tmp = tempfile.mkdtemp(prefix="rep188-")
    conf = json.load(open(f"{snap}/driver.json"))
    reg = json.load(open(f"{snap}/registry.json"))
    for s in reg["slices"].values(): s["status"] = "ready"            # rotate events set the statuses in recorded order
    json.dump(reg, open(f"{tmp}/registry.json", "w"))
    conf.update(root=f"{tmp}/drv", menu=f"{snap}/menu.json", registry=f"{tmp}/registry.json", evals=f"{tmp}/evals.jsonl",
                results=f"{tmp}/results.jsonl", docs=f"{tmp}/docs")
    conf["monitor"] = dict(conf.get("monitor") or {}, when="never")  # replay: no monitor submits (they never decide)
    if a.pool is not None: conf["pool"] = a.pool
    for kv in a.conf_set:
        k, v = kv.split("=", 1); conf[k] = json.loads(v)
    os.makedirs(f"{tmp}/drv/assign")
    for f in os.listdir(f"{snap}/drv/assign"): shutil.copy(f"{snap}/drv/assign/{f}", f"{tmp}/drv/assign/{f}")
    json.dump(conf, open(f"{tmp}/conf.json", "w"))

    D = DR.Driver(f"{tmp}/conf.json")
    captured = []
    D.rec = lambda e: captured.append(dict(e))                         # the events the replayed driver emits
    recorded = []
    real_stdout = sys.stdout; sys.stdout = open(os.devnull, "w")       # the driver's log lines
    try:
        for l in open(f"{snap}/drv/state.jsonl"):
            e = json.loads(l); ev = e["ev"]
            if ev == "submit": D.sub[e["tag"]] = e; D.seen.add(e["hash"])
            elif ev == "result":
                t, r = e["tag"], e["r"]; D.res[t] = r; D.n_results += 1; D.on_result(t, r)
            elif ev == "rotate":
                for s in D.reg.d["slices"].values():
                    if s["status"] == "active": s["status"] = "retired"
                D.reg.d["slices"][e["to"]]["status"] = "active"
            elif ev == "b": D.b[e["size"]] = e["b"]
            elif ev == "incumbent": D.inc[e["size"]] = e["tag"]
            elif ev == "act": D.act = e["mode"]
            elif ev == "bytes_offset": D.bytes_off = e["off"]; D.bytes_src = e.get("src", "measured"); D.M.fixed = D.fixed0 + e["off"]
            elif ev == "decision": recorded.append(e)
    finally:
        sys.stdout.close(); sys.stdout = real_stdout
    keep = ("ev", "tag", "size", "decision", "kind", "k", "logr", "logr_pooled", "off", "se", "z", "parts", "prev", "join", "replaced", "members", "why")
    out = {"captured": [{k: e[k] for k in e if k in keep or k in ("inc", "pool", "slice", "ranking")} for e in captured],
           "recorded_decisions": [{k: e.get(k) for k in keep if k in e} for e in recorded],
           "pool": conf.get("pool", 1), "snap_state_sha256": hashlib.sha256(open(f"{snap}/drv/state.jsonl", "rb").read()).hexdigest()}
    json.dump(out, open(a.out, "w"), indent=0, default=str)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{a.out}: {len(captured)} captured events, {len(recorded)} recorded decisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
