# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""dryrun192.py -- the #192 driver inbox on a copy of a live run's state (w171): Driver.step() N times on the snapshot's copy with no
results arriving, optional inbox files copied into the copy's inbox/ (oldest first, mtimes kept). Output (JSON): the events emitted
(time stamps dropped), the queued eval lines (tag, slice, early), the driver log and the status line (time stamps dropped), the
incumbents and pools, and the files left in inbox/done and inbox/rejected.

  python3 -I dryrun192.py --tree TREE --snap SNAP --out OUT.json [--steps 3] [--inbox FILE ...] [--conf-set KEY=JSON ...]

TREE: a directory with search/v3 (the tree under test). SNAP: a live run's driver dir (driver.json, registry.json, menu.json,
drv/state.jsonl, drv/assign/); read only. As confirm189/step_events.py: fit_done events are dropped from the copy (no fit is
reloaded and generation stays off), so the queued evals are the inbox's and the maintenance ones only."""
import argparse, json, os, shutil, sys, tempfile


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dryrun192.py")
    ap.add_argument("--tree", required=True); ap.add_argument("--snap", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--inbox", nargs="*", default=[])
    ap.add_argument("--conf-set", action="append", default=[], metavar="KEY=JSON", help="override a CONF key on the copy (e.g. inbox_max=20)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.abspath(a.tree), "search", "v3"))
    import driver as DR     # noqa: E402  (the tree under test)

    snap = os.path.abspath(a.snap); tmp = tempfile.mkdtemp(prefix="dry192-")
    conf = json.load(open(f"{snap}/driver.json")); reg = json.load(open(f"{snap}/registry.json"))
    json.dump(reg, open(f"{tmp}/registry.json", "w"))
    os.makedirs(f"{tmp}/drv/assign"); os.makedirs(f"{tmp}/docs")
    for f in os.listdir(f"{snap}/drv/assign"): shutil.copy(f"{snap}/drv/assign/{f}", f"{tmp}/drv/assign/{f}")
    with open(f"{tmp}/drv/state.jsonl", "w") as o:
        for l in open(f"{snap}/drv/state.jsonl"):
            if '"ev": "fit_done"' not in l: o.write(l)
    if a.inbox:
        os.makedirs(f"{tmp}/drv/inbox")
        for f in a.inbox: shutil.copy2(f, f"{tmp}/drv/inbox/{os.path.basename(f)}")
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
    finally:
        sys.stdout.close(); sys.stdout = real
    ev = [json.loads(l) for i, l in enumerate(open(f"{tmp}/drv/state.jsonl")) if i >= n0]
    for e in ev: e.pop("t", None)
    q = [json.loads(l) for l in open(f"{tmp}/evals.jsonl")] if os.path.exists(f"{tmp}/evals.jsonl") else []
    logs = [l.split(" CST ", 1)[-1].rstrip("\n") for l in list(open(f"{tmp}/drv/driver.log"))[logs0:]]
    status = open(f"{tmp}/drv/status.txt").read().strip() if os.path.exists(f"{tmp}/drv/status.txt") else None
    out = {"events": ev, "queued": [{"tag": x["tag"], "slice": x["slices"][0]["name"], "early": "early_stop" in x} for x in q], "log": logs,
           "status": status.split(" | ", 1)[-1] if status else None,
           "inc": {str(k): v for k, v in D.inc.items()}, "pool": {str(k): v for k, v in D.pool.items()},
           "inbox": {d: sorted(os.listdir(f"{tmp}/drv/inbox/{d}")) for d in ("done", "rejected") if os.path.isdir(f"{tmp}/drv/inbox/{d}")}}
    json.dump(out, open(a.out, "w"), indent=0, default=str)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{a.out}: {len(ev)} events, {len(q)} queued, {len(out['inbox'].get('done', []))} inbox files done, "
          f"{len(out['inbox'].get('rejected', []))} rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
