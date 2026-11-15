# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rerank.py -- arm (d) of RULES-187 section 4: an xgboost reranker on top of the base arm.

  rerank.py SNAP CANDS.jsonl RUNDIR BASE_ARM     writes RUNDIR/d/<N>-all.json for every refit point N (replay.py's format)

Target: measured log r (decision.logr, whole-slice rows only). Features: the base arm's prediction, move kind, bpw moved,
changed-tensor counts per group and per depth tercile, mean K before / after, K change, depth of the changed tensors,
embedding / head changed, size, band. Training at refit point N: whole-slice rows with result index <= N, each with its
out-of-sample base prediction (made at its own refit point). Fewer than 30 training rows: the base prediction is returned."""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "refit")); sys.path.insert(0, HERE)
import menu as MN            # noqa: E402
import replay as R           # noqa: E402
import metrics as MT         # noqa: E402

KINDS = ["solve", "partial", "pair", "block", "thompson", "explore"]
GROUPS = ["attn_in", "attn_out", "ssm_in", "ssm_out", "ffn_gu", "ffn_down"]
MIN_TRAIN = 30
XGB = dict(n_estimators=60, max_depth=2, learning_rate=0.1, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
           random_state=187, n_jobs=1, objective="reg:squarederror", tree_method="hist")


def features(M, S, c, base_pred, nl):
    cc, aa = S.assign(c["tag"]), S.assign(c["ref"])
    ch = []
    for w in cc:
        if w == MN.EMB or cc[w] == aa.get(w): continue
        ch.append((w, M.opt(w, aa[w]).get("K", float("nan")), M.opt(w, cc[w]).get("K", float("nan"))))
    trunk = [x for x in ch if x[0] != MN.HEAD]
    layers = [M.T[M.WI[w]]["layer"] for w, _, _ in trunk]
    groups = {g: 0 for g in GROUPS}
    for w, _, _ in trunk:
        g = M.T[M.WI[w]]["group"]
        if g in groups: groups[g] += 1
    terc = [0, 0, 0]
    for l in layers: terc[min(2, 3 * l // max(nl, 1))] += 1
    Ka = [k for _, k, _ in trunk]; Kc = [k for _, _, k in trunk]
    kind = [1.0 if c["kind"] == k else 0.0 for k in KINDS]
    bpw_moved = 1000.0 * (M.bytes(cc) - M.bytes(aa)) * 8 / M.P          # milli-bits per parameter
    f = [base_pred] + kind + [bpw_moved, len(trunk), sum(1 for a, b in zip(Ka, Kc) if b > a), sum(1 for a, b in zip(Ka, Kc) if b < a),
          float(np.mean(Ka)) if Ka else 0.0, float(np.mean(Kc)) if Kc else 0.0, float(np.sum(np.array(Kc) - np.array(Ka))) if Ka else 0.0,
          float(np.mean(layers)) / max(nl, 1) if layers else 0.0] + [groups[g] for g in GROUPS] + terc + \
        [float(cc.get(MN.EMB) != aa.get(MN.EMB)), float(cc.get(MN.HEAD) != aa.get(MN.HEAD)), c["size"], float(MT.BANDS.index(c["band"]) if c["band"] in MT.BANDS else -1)]
    return f


FEATURE_NAMES = (["base_pred"] + [f"kind_{k}" for k in KINDS] +
                 ["bpw_moved", "n_changed", "n_K_up", "n_K_down", "mean_K_before", "mean_K_after", "sum_K_delta", "mean_depth"] +
                 [f"n_{g}" for g in GROUPS] + ["n_layers_t1", "n_layers_t2", "n_layers_t3", "emb_changed", "head_changed", "size", "band"])


def main(argv):
    snap, cpath, rundir, base = argv[0], argv[1], argv[2], argv[3]
    S = R.Snap(snap); M = S.M
    nl = 1 + max(t["layer"] for t in M.T if t["w"].startswith("blk."))
    cands = MT.load_cands(cpath)
    base_p = MT.load_preds(rundir, base, cands)
    F = {c["tag"]: np.array(features(M, S, c, base_p[c["tag"]] if math.isfinite(base_p[c["tag"]]) else 0.0, nl)) for c in cands}
    outdir = os.path.join(rundir, "d"); os.makedirs(outdir, exist_ok=True)
    import xgboost as xgb
    maxN = max(MT.chunk_of(c["result_idx"]) for c in cands)
    log = []
    for N in range(MT.CHUNK, maxN + 1, MT.CHUNK):
        chunk = [c for c in cands if MT.chunk_of(c["result_idx"]) == N]
        if not chunk: continue
        train = [c for c in cands if c["result_idx"] <= N and not c["stopped"] and math.isfinite(base_p[c["tag"]])]
        if len(train) >= MIN_TRAIN:
            X = np.vstack([F[c["tag"]] for c in train]); y = np.array([c["logr"] for c in train])
            mdl = xgb.XGBRegressor(**XGB); mdl.fit(X, y)
            pred = {c["tag"]: float(v) for c, v in zip(chunk, mdl.predict(np.vstack([F[c["tag"]] for c in chunk])))}
            mode = "xgb"
        else:
            pred = {c["tag"]: base_p[c["tag"]] for c in chunk}; mode = "base"
        rep = {"N": N, "arm": "d", "group": "all", "base": base, "mode": mode, "n_train": len(train), "n_cand": len(chunk),
               "n_nonfinite": sum(1 for v in pred.values() if not math.isfinite(v)), "preds": pred}
        p = os.path.join(outdir, f"{N:04d}-all.json"); json.dump(rep, open(p + ".part", "w")); os.replace(p + ".part", p)
        log.append({"N": N, "mode": mode, "n_train": len(train), "n_cand": len(chunk)})
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
