#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check_split.py -- the codes / scales stamp split (user 11-12): a [post.swj] / [post.healing] change keeps the codes
stamp (jobs, level names) and only makes swj levels' DONE stale; a quant or C9 change moves the codes stamp; levels
without a swj pass (K > swj k_max, or a run without the swj step) stay DONE. CPU only, a second."""
import copy, json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import config as C, ladder  # noqa: E402
from rxrun.run import level_current  # noqa: E402

fails = 0


def report(name, ok, msg):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")


cfg = C.load()
lv = {float(v["K"]): v for v in ladder.levels(cfg)}
a, b = lv[2.5], lv[6.0]
swj = copy.deepcopy(cfg); swj["post"]["swj"]["fit_tokens"] *= 4
heal = copy.deepcopy(cfg); heal["post"]["healing"]["note"] = "x"
c9 = copy.deepcopy(cfg); c9["post"]["c9"]["e_clip"] = 0.3
q = copy.deepcopy(cfg); q["quant"]["_probe"] = 1
st, ss = C.settings_stamp, C.scales_stamp
report("swj-keeps-codes", st(swj) == st(cfg) and ss(swj) != ss(cfg), f"fit_tokens x4: codes {st(swj)}, scales {ss(cfg)} -> {ss(swj)}")
report("healing-keeps-codes", st(heal) == st(cfg) and ss(heal) != ss(cfg), "[post.healing] change")
report("c9-moves-codes", st(c9) != st(cfg) and ss(c9) == ss(cfg), "e_clip 0.3 (C9 runs inside the job)")
report("quant-moves-codes", st(q) != st(cfg) and ss(q) == ss(cfg), "[quant] change")
report("names", [v["name"] for v in ladder.levels(swj)] == [v["name"] for v in ladder.levels(cfg)], "level names follow the codes stamp only")
with tempfile.TemporaryDirectory() as D:
    def done(s): json.dump({"scales": s}, open(f"{D}/DONE", "w"))
    done(ss(cfg))
    report("current", level_current(D, cfg, a), "K2.5 DONE at today's scales stamp")
    report("stale", not level_current(D, swj, a), "K2.5 DONE after a swj change: stale (redo swj + post)")
    done(None)
    report("no-swj-level", level_current(D, swj, b), "K6 (above swj k_max) DONE without scales: current")
    report("pre-split", not level_current(D, cfg, a), "K2.5 DONE without scales (pre-split / swj off): stale")
    report("steps-no-swj", level_current(D, cfg, a, ["jobs", "assemble", "post", "recipe"]), "a run without the swj step: current")
print(f"SPLIT CHECKS {'ALL PASS' if not fails else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
