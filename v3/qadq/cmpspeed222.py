#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmpspeed222.py RUNDIR [RUNDIR..] -- per run: step 1 / 2 train KL and gnorm, the median wall (s / step) of steps 3-5 and the peak GiB of the main steps (w222's --dynamo-limit gate)."""
import json, os, sys
import numpy as np
for r in sys.argv[1:]:
    f = f"{r}/train.jsonl"
    if not os.path.exists(f): print(os.path.basename(r), "(no train.jsonl yet)"); continue
    m = [x for x in map(json.loads, open(f)) if x.get("what") == "main"]
    w = [x["wall"] for x in m if 3 <= x["step"] <= 5 and "wall" in x]
    print(f"{os.path.basename(r):14s} steps {len(m)} | KL1 {m[0]['train_kl']!r} g1 {m[0]['gnorm']} | KL2 {m[1]['train_kl']!r} g2 {m[1]['gnorm']} | wall(3-5) {np.median(w) if w else float('nan'):.1f} s | peak {max(x.get('peak_gib', 0) for x in m):.2f} GiB" if len(m) >= 2 else f"{os.path.basename(r)}: {len(m)} steps")
