#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""exportcheck.py RUN MODEL LEVEL GOLDEN POOLS SCRATCH [--layers N] [--devs ...] -- the export round trip of a qad.py
run: rxq.swj apply_level(LEVEL, RUN) -> SCRATCH/level, then (1) every trained tensor's stored suh / svh equal the
student's own f16 export of its best state bit for bit, and (2) a fresh, untrained student built on the healed level
scores the run's monitor documents at the run's best monitor KL (differences = the f16 rounding of the stored
scales only; PASS if within 0.5% relative)."""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qmodel  # noqa: E402
from qad import Golden, eval_kl  # noqa: E402
from rxq import swj, deq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("model"); ap.add_argument("level"); ap.add_argument("golden"); ap.add_argument("pools")
ap.add_argument("scratch"); ap.add_argument("--layers", type=int, default=None); ap.add_argument("--devs", default="cuda:0")
a = ap.parse_args()
P = torch.load(os.path.join(a.run, "params.pt"), weights_only=False)
args = P["meta"]["args"]
dst = os.path.join(a.scratch, "level")
swj.apply_level(a.level, a.run, dst)
g = Golden(a.golden)
mon = json.load(open(os.path.join(a.run, "docs.json")))["monitor_docs"]
devs = a.devs.split(",")
s0 = qmodel.Student(a.model, a.level, devs, layers=a.layers, norms=args["arm"] == "C")
s0.load_state(P["state"])
ex = s0.export_scales()
lv = deq.Level(dst)
bad = 0
for k, (su, sv) in ex.items():
    r = lv.raw(k)
    bad += int(not torch.equal(r["suh"], su)) + int(not torch.equal(r["svh"], sv))
print(f"stored vs exported f16 scales: {2 * len(ex) - bad}/{2 * len(ex)} vectors bit-equal")
with torch.no_grad():
    v_trained, _ = eval_kl(s0, g, mon, args["len"])
del s0; torch.cuda.empty_cache()
s1 = qmodel.Student(a.model, dst, devs, layers=a.layers)
if args["arm"] == "C":                        # norms are not in the level: carry them over for the comparison
    print("arm C: norms not in the level; healed-level KL below excludes them")
with torch.no_grad():
    v_level, _ = eval_kl(s1, g, mon, args["len"])
best = P["meta"]["best_val"]
rel = abs(v_level - v_trained) / v_trained
print(f"monitor KL: run best {best:.6g}, student with params {v_trained:.6g}, fresh student on the healed level {v_level:.6g} "
      f"(rel {rel:.2e}); unhealed {P['meta']['base_val']:.6g}")
ok = bad == 0 and (rel <= 0.005 or args["arm"] == "C") and abs(v_trained - best) / best <= 1e-3
print("EXPORTCHECK " + ("PASS" if ok else "FAIL"))
