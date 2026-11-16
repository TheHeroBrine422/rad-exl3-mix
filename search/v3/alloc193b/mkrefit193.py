# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkrefit193.py SNAP MENU_PATH [NDRAW] -- w181 (RULES-193 amendment 12): my fit.py refit inputs. SNAP/fit-b{1,2,3}.in.json (a copy
of the driver's latest fit inputs) with every row whose label's slice is not on common193.ALLOWED dropped ->
SNAP/refit-b{1,2,3}.in.json (same opts; ndraw draws for rankreg's generator). Prints kept / dropped by slice."""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common193 as C  # noqa: E402

snap, menu = sys.argv[1], sys.argv[2]; nd = int(sys.argv[3]) if len(sys.argv) > 3 else 16
for g in ("b1", "b2", "b3"):
    d = json.load(open(f"{snap}/fit-{g}.in.json")); C.dropped.clear()
    obs = [o for o in d["obs"] if C.ok_slice(o["label"])]
    out = {"menu": menu, "obs": obs, "opts": d.get("opts", {}), "ndraw": nd, "seed": 193}
    json.dump(out, open(f"{snap}/refit-{g}.in.json", "w"))
    print(g, len(obs), "rows kept", dict(collections.Counter(o["label"].split("|")[0] for o in obs)), "dropped", dict(C.dropped))
