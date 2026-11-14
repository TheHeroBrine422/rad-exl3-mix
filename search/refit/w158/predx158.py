#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""predx158.py MENU MODELS NAME FIT OBS_TRAIN ASSIGN.json.. -- (w158) a fit's A8 KL prediction for assignments (offset from the
training rows), e.g. the A3 picks"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fin158 as F8, m158 as M8, menu as MN  # noqa: E402
M = MN.Menu.load(sys.argv[1]); spec = json.load(open(sys.argv[2]))[sys.argv[3]]; obs = F8.load_obs([sys.argv[5]])
Mo = F8.build(M, obs, spec); Mo.layout(); x = np.array(json.load(open(sys.argv[4]))["x"])
rows = [{"assign": json.load(open(p)), "mode": "a8"} for p in sys.argv[6:]]
lp = F8.predict_for(Mo, x, obs, rows)
for p, l in zip(sys.argv[6:], lp): print(f"{os.path.basename(p)} {math.exp(l):.5f}")
