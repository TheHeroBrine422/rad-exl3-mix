# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rerank_static.py -- train the static reranker the driver's CONF "rerank" hook loads (RULES-187 arm d, default off).

  rerank_static.py SNAP CANDS.jsonl RUNDIR BASE OUT.json [OUT_CHECK.json]

Trains on every whole-slice row (result index > 20, not early-stopped) with the base arm's predictions as the feature,
with rerank.py's hyper-parameters, and saves xgboost's JSON model. Checks that rerank_hook.Booster reproduces
xgboost's predictions on those rows (max |diff| printed). This is the model used for the replay's last refit point
(N = 320), so it is trained on all replay rows: it is NOT an out-of-sample model."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "refit")); sys.path.insert(0, HERE)
import replay as R           # noqa: E402
import metrics as MT         # noqa: E402
import rerank as RR          # noqa: E402
import rerank_hook as RH     # noqa: E402


def main(argv):
    snap, cpath, rundir, base, out = argv[:5]
    import xgboost as xgb
    S = R.Snap(snap); M = S.M
    cands = MT.load_cands(cpath); bp = MT.load_preds(rundir, base, cands)
    rows = [c for c in cands if not c["stopped"] and math.isfinite(bp[c["tag"]])]
    X = np.array([RH.features(M, S.assign(c["tag"]), S.assign(c["ref"]), c["kind"], c["size"], c["band"], bp[c["tag"]]) for c in rows])
    y = np.array([c["logr"] for c in rows])
    assert X.shape[1] == len(RH.FEATURE_NAMES)
    mdl = xgb.XGBRegressor(**RR.XGB); mdl.fit(X, y)
    mdl.get_booster().save_model(out)
    b = RH.Booster(out)
    want = mdl.predict(X); got = np.array([b.predict(list(x)) for x in X])
    print(json.dumps({"rows": len(rows), "features": X.shape[1], "trees": len(b.trees), "max_abs_diff_vs_xgboost": float(np.max(np.abs(want - got)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
