#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""d0jobs.py noop SPECDIR OUT NAME...  /  d0jobs.py asym SPECDIR OUT --hin DIR --wpre DIR
#165 arm D0's writer requant in one process (no per-job start-up): rxq.job.run_quant on a ladder level's own job
specs. noop: the named specs unchanged (the no-op gate: rxq.pairs same LEVEL OUT/NAME must hold); asym: every writer
spec (L*-g1 = o_proj / out_proj, L*-g3 = down_proj) with inputs.hin / inputs.wpre set (rxq.config.c9_injob then skips
the in-job C9: it runs per mix on the final writers, rxq.post c9 --dn). A job dir with job.json is finished (skipped);
run_quant itself skips finished tensors, so a killed run resumes."""
import argparse, glob, json, os, sys, time

from rxq import job

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("noop", "asym")); ap.add_argument("specdir"); ap.add_argument("out")
ap.add_argument("names", nargs="*"); ap.add_argument("--hin"); ap.add_argument("--wpre")
a = ap.parse_args()
if a.mode == "noop":
    specs = [os.path.join(a.specdir, f"{n}.json") for n in a.names]
else:
    assert a.hin and a.wpre, "asym needs --hin and --wpre"
    specs = sorted(glob.glob(os.path.join(a.specdir, "L*-g1.json")) + glob.glob(os.path.join(a.specdir, "L*-g3.json")))
    assert len(specs) == 128, f"{len(specs)} writer specs, not 128"
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
for i, p in enumerate(specs):
    name = os.path.basename(p)[:-len(".json")]
    o = os.path.join(a.out, name)
    if os.path.exists(os.path.join(o, "job.json")): continue
    s = json.load(open(p))
    if a.mode == "asym":
        s["inputs"]["hin"], s["inputs"]["wpre"] = os.path.abspath(a.hin), os.path.abspath(a.wpre)
    os.makedirs(o, exist_ok=True)
    json.dump(s, open(os.path.join(o, "spec.json"), "w"), indent=1)
    r = job.run_quant(s, o, 0)
    print(f"{i + 1}/{len(specs)} {name} {s['keys']} {time.time() - t0:.0f} s", flush=True)
print(f"D0JOBS DONE {a.mode} {len(specs)} specs in {time.time() - t0:.0f} s")
