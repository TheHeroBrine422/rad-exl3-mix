# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_hook.py -- checks for the RULES-187 driver hook (driver-187.patch; run on colo with the patched tree).

  test_hook.py SNAP CANDS.jsonl RUNDIR MODEL.json ORIG_MOVES.py
    1. rerank_hook.features == rerank.features on every replay candidate (the model was trained with rerank.py's).
    2. rerank_hook.Booster == xgboost's predictions on those candidates (max |diff| < 1e-4).
    3. moves.generate with rerank=None == the original moves.py (same rng seed, every kind): the default path is unchanged.
    4. moves.generate with the reranker returns a valid new candidate in the window for every kind, on a real fit (N = 200).
"""
import importlib.util, json, math, os, random, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "refit")); sys.path.insert(0, HERE)
import replay as R           # noqa: E402
import metrics as MT         # noqa: E402
import rerank as RR          # noqa: E402
import rerank_hook as RH     # noqa: E402
import fit as FT             # noqa: E402
import moves as MV           # noqa: E402  (the patched tree's moves.py when run from the patched tree)


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
    if not ok: check.failed += 1
check.failed = 0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def main(argv):
    snap, cpath, rundir, model, orig = argv[:5]
    S = R.Snap(snap); M = S.M; nl = RH.n_layers(M)
    cands = MT.load_cands(cpath); bp = MT.load_preds(rundir, "b", cands)
    # 1. features
    fmax = 0
    for c in cands:
        if not math.isfinite(bp[c["tag"]]): continue
        f1 = RH.features(M, S.assign(c["tag"]), S.assign(c["ref"]), c["kind"], c["size"], c["band"], bp[c["tag"]], nl)
        Sx = type("S", (), {"assign": lambda self, t: S.assign(t)})()
        f2 = RR.features(M, Sx, dict(c, pred=bp[c["tag"]]), bp[c["tag"]], nl)
        d = np.abs(np.array(f1, float) - np.array(f2, float)); fmax = max(fmax, float(np.nanmax(d)) if np.any(np.isfinite(d)) else 0.0)
    check("features equal rerank.py on every candidate", fmax < 1e-12, f"max |diff| {fmax:.1e}")
    # 2. booster
    import xgboost as xgb
    B = RH.Booster(model)
    X = np.array([RH.features(M, S.assign(c["tag"]), S.assign(c["ref"]), c["kind"], c["size"], c["band"], bp[c["tag"]], nl)
                  for c in cands if math.isfinite(bp[c["tag"]])])
    xb = xgb.Booster(); xb.load_model(model)
    want = xb.predict(xgb.DMatrix(X)); got = np.array([B.predict(list(x)) for x in X])
    check("Booster == xgboost predictions", float(np.max(np.abs(want - got))) < 1e-4, f"max |diff| {np.max(np.abs(want - got)):.1e} over {len(X)}")
    # 3./4. moves on a real fit at N = 200 (arm b's rows)
    obs = S.obs_at(200) + S.prior
    Mo = FT.build(M, obs, S.conf.get("fit_opts", {})); x = Mo.fit(); pr = MV.Pred(Mo, x)
    old = load(orig, "moves_orig")
    a = S.assign(cands[-1]["ref"]); size = 2.5
    lo, hi = (size - 0.02) * M.P / 8, size * M.P / 8
    same = True; valid = True; nn = 0
    rr = RH.scorer(M, B, size, "b1", a, pr, nl)
    for kind in MV.KINDS:
        for seed in range(3):
            c_old = old.generate(kind, pr, [], a, lo, hi, set(), random.Random(seed))
            c_new = MV.generate(kind, pr, [], a, lo, hi, set(), random.Random(seed))
            same = same and (c_old == c_new)
            c_rr = MV.generate(kind, pr, [], a, lo, hi, set(), random.Random(seed), rerank=rr, rerank_k=8)
            if c_rr is not None:
                nn += 1
                valid = valid and (lo <= M.bytes(c_rr) <= hi) and c_rr != a
    check("default path (rerank=None) == original moves.generate", same, "every kind x 3 seeds")
    check("reranked path returns valid new candidates in the window", valid and nn > 0, f"{nn} candidates returned")
    print("FAILED %d" % check.failed if check.failed else "all checks passed")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
