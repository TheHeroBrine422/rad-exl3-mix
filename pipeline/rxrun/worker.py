# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.worker [--device N|cpu] -- a persistent rxq job process (w105).

Protocol: stdin one JSON request a line, {"id": ID, "spec": SPEC.json, "out": DIR} or {"quit": true}; stdout one JSON
reply a line: {"ready": PID} once, then per request {"id", "rc", "wall", "tensors", "oom", "err"}. rc 0 = done,
1 = failed (the worker exits after replying: a process is never reused after a failure), 3 = respawn (the spec
needs another engine setup, e.g. another codebook: nothing was written; the worker exits).

Per request it does what `python -m rxq job SPEC -o DIR --device N` does (rxq/__main__.py): rxq.job.run_quant(spec,
DIR, device) with the profiler on, after VRAM_CAP_GB's allocator cap (once). Differences, none of which reaches an
output byte: the torch import, the HIP context, the kernel builds / loads and the engine setup happen once a
process; the settings file is parsed once (reloaded when its mtime changes); the code stamp is hashed once (the code
does not change under a running worker); the job's own prints go to DIR/rxq.log (as layer92.sh's redirect).
Before each job the dir is cleaned as rad-quant-at-home's rxq_quant run() does (clean92.py): log.jsonl keeps the
lines whose tensor exists, tensor / C9 files that log.jsonl does not record (a job killed between save and log) are
removed, so a rerun redoes exactly those tensors."""
import argparse, contextlib, json, os, sys, time, traceback


def clean(o):
    """= v3/e3/settle/w92/clean92.py"""
    lp = f"{o}/log.jsonl"; ok = set()
    if os.path.exists(lp):
        rows = []
        for l in open(lp):
            try: r = json.loads(l)
            except ValueError: break                     # a torn last line
            if os.path.exists(f"{o}/t-{r['key']}.safetensors"): rows.append(l if l.endswith("\n") else l + "\n"); ok.add(r["key"])
        with open(lp + ".part", "w") as f: f.writelines(rows)
        os.replace(lp + ".part", lp)
    for f in os.listdir(o):
        for p in ("t-", "c9-"):
            if f.startswith(p) and f.endswith(".safetensors") and f[len(p):-len(".safetensors")] not in ok: os.remove(f"{o}/{f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.worker")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    proto = os.fdopen(os.dup(1), "w", buffering=1)       # replies only; every other print (python or C) goes to stderr
    os.dup2(2, 1)
    import torch
    from rxq import config as C, engine, job
    from rxq.__main__ import _cap, _dev
    device = _dev(a.device)
    _cap(device)
    stamps = {}
    orig_stamp = engine.code_stamp

    def code_stamp(Q):
        if id(Q) not in stamps: stamps[id(Q)] = orig_stamp(Q)
        return stamps[id(Q)]
    engine.code_stamp = code_stamp                       # job.run_quant calls it through the module
    cfgs = {}

    def cfg_of(path):
        p = path or os.environ.get("RXQ_CONFIG") or C.DEFAULT
        m = os.stat(p).st_mtime_ns
        if p not in cfgs or cfgs[p][0] != m: cfgs[p] = (m, C.load(p))
        return cfgs[p][1]
    proto.write(json.dumps({"ready": os.getpid()}) + "\n")
    for line in sys.stdin:
        if not line.strip(): continue
        req = json.loads(line)
        if req.get("quit"): break
        out = req["out"]; os.makedirs(out, exist_ok=True); clean(out)
        spec = json.load(open(req["spec"]))
        t0 = time.time(); rep = {"id": req.get("id"), "rc": 0, "tensors": 0, "oom": False, "err": None}
        with open(f"{out}/rxq.log", "a") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            print(f"rxrun.worker pid {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
            try:
                jr = job.run_quant(spec, out, device, cfg=cfg_of(spec.get("config")))
                print(json.dumps(jr), flush=True)
                rep["tensors"] = jr["tensors"]
            except RuntimeError as e:
                if "one setup a process" in str(e): rep.update(rc=3, err=str(e)[:300])
                else:
                    traceback.print_exc(); rep.update(rc=1, err=repr(e)[:300], oom="out of memory" in str(e).lower())
            except BaseException as e:            # noqa: BLE001 (SystemExit from rxq's checks too)
                traceback.print_exc()
                rep.update(rc=1, err=repr(e)[:300], oom=isinstance(e, torch.OutOfMemoryError))
        rep["wall"] = round(time.time() - t0, 3)
        proto.write(json.dumps(rep) + "\n")
        if rep["rc"]: sys.exit(rep["rc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
