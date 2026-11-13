#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""loo174.py FIT.in.json ARM HOLDOUT OUT.json -- one leave-one-out fit for the embedding term's form (w135, #174): fit
the search's rows + the calibration rows without the calibration eval HOLDOUT (its rows of every part; "none" = all
in), then predict that eval's per-part log ratio vs its anchor (the incumbent row of the same label) and compare with
the measured one. ARM = JSON fit opts merged over the input's (e.g. '{"emb_at": "out"}', or @FILE). Needs fit.py with
fit-embprior.patch. CPU, single-threaded (colo)."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import fit as FT             # noqa: E402
import menu as MN            # noqa: E402


def main(argv):
    d = json.load(open(argv[0])); ho = argv[2]
    arm = json.load(open(argv[1][1:])) if argv[1].startswith("@") else json.loads(argv[1])
    M = MN.Menu.load(d["menu"]); opts = dict(d.get("opts") or {}, **arm)
    cal = [o for o in d["obs"] if o["label"].startswith("emb174-") and o.get("se", 0) > 0]
    test = [o for o in cal if o["tag"].split("@")[0] == ho]
    obs = [o for o in d["obs"] if o not in test]
    Mo, x, _, rep = FT.fit_obs(M, obs, opts)
    out = {"arm": arm, "holdout": ho, "report": rep, "emb": Mo.dump(x)["emb"], "rows": {}}
    for o in (test if ho != "none" else cal):
        anc = next(a for a in d["obs"] if a["label"] == o["label"] and a.get("se", 0) == 0
                   and a["assign"].get(MN.EMB) == "base" and all(a["assign"][w] == o["assign"][w] for w in o["assign"] if w != MN.EMB))
        lp = math.log(Mo.predict(x, o["assign"]) / Mo.predict(x, anc["assign"]))
        if Mo.xscale and FT.part_of(o["label"]) == "x": lp *= Mo.lam(x)          # x-part ratios are compressed by lamx
        lm = math.log(o["kld"] / anc["kld"])
        out["rows"][o["tag"]] = {"meas": lm, "pred": lp, "se": o["se"]}
    json.dump(out, open(argv[3], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
