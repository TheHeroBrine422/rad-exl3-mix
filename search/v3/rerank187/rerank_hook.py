# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rerank_hook.py -- the RULES-187 reranker as the driver calls it (arm d; CONF "rerank", default off).

The driver loads a static reranker (an xgboost model saved as JSON by rerank_static.py) and scores each proposal of
moves.generate(rerank=...) by its predicted measured log r. The trees are evaluated here in pure python, so the driver
(on the VM, which has no xgboost) needs no new package. features() is rerank.py's feature function on the driver's own
objects (the menu and the two assignments), so the two agree (test_hook.py checks it on every replay candidate).

  score(M, booster, ca, aa, kind, size, band, base_pred, nl) -> predicted log r of candidate ca vs its reference aa
  scorer(M, booster, size, band, aa, pred, kind_of_call) -> the callable moves.generate(rerank=...) takes: (c, kind) -> score
"""
import json, math

import numpy as np

import menu as MN            # the driver's directory is on sys.path

KINDS = ["solve", "partial", "pair", "block", "thompson", "explore"]
GROUPS = ["attn_in", "attn_out", "ssm_in", "ssm_out", "ffn_gu", "ffn_down"]
BANDS = ["b1", "b2", "b3", "b4"]
FEATURE_NAMES = (["base_pred"] + [f"kind_{k}" for k in KINDS] +
                 ["bpw_moved", "n_changed", "n_K_up", "n_K_down", "mean_K_before", "mean_K_after", "sum_K_delta", "mean_depth"] +
                 [f"n_{g}" for g in GROUPS] + ["n_layers_t1", "n_layers_t2", "n_layers_t3", "emb_changed", "head_changed", "size", "band"])


def n_layers(M):
    return 1 + max(t["layer"] for t in M.T if t["w"].startswith("blk."))


def features(M, ca, aa, kind, size, band, base_pred, nl=None):
    """the feature vector of one candidate (ca) against its reference (aa): assignments (tensor -> option name)"""
    nl = nl or n_layers(M)
    ch = []
    for w in ca:
        if w == MN.EMB or ca[w] == aa.get(w): continue
        ch.append((w, M.opt(w, aa[w]).get("K", float("nan")), M.opt(w, ca[w]).get("K", float("nan"))))
    trunk = [x for x in ch if x[0] != MN.HEAD]
    layers = [M.T[M.WI[w]]["layer"] for w, _, _ in trunk]
    groups = {g: 0 for g in GROUPS}
    for w, _, _ in trunk:
        g = M.T[M.WI[w]]["group"]
        if g in groups: groups[g] += 1
    terc = [0, 0, 0]
    for l in layers: terc[min(2, 3 * l // max(nl, 1))] += 1
    Ka = [k for _, k, _ in trunk]; Kc = [k for _, _, k in trunk]
    kind_f = [1.0 if kind == k else 0.0 for k in KINDS]
    bpw_moved = 1000.0 * (M.bytes(ca) - M.bytes(aa)) * 8 / M.P          # milli-bits per parameter
    return [base_pred] + kind_f + [bpw_moved, len(trunk), sum(1 for a, b in zip(Ka, Kc) if b > a), sum(1 for a, b in zip(Ka, Kc) if b < a),
            float(np.mean(Ka)) if Ka else 0.0, float(np.mean(Kc)) if Kc else 0.0, float(np.sum(np.array(Kc) - np.array(Ka))) if Ka else 0.0,
            float(np.mean(layers)) / max(nl, 1) if layers else 0.0] + [groups[g] for g in GROUPS] + terc + \
        [float(ca.get(MN.EMB) != aa.get(MN.EMB)), float(ca.get(MN.HEAD) != aa.get(MN.HEAD)), float(size), float(BANDS.index(band) if band in BANDS else -1)]


class Booster:
    """xgboost's JSON model (reg:squarederror, regression trees), evaluated in pure python"""

    def __init__(self, src):
        d = json.load(open(src)) if isinstance(src, str) else src
        lm = d["learner"]
        self.base = float(json.loads(lm["learner_model_param"]["base_score"])[0]) if str(lm["learner_model_param"]["base_score"]).startswith("[") \
            else float(lm["learner_model_param"]["base_score"])
        self.trees = [(t["split_indices"], t["split_conditions"], t["left_children"], t["right_children"], t["default_left"])
                      for t in lm["gradient_booster"]["model"]["trees"]]
        self.nfeat = int(lm["learner_model_param"]["num_feature"])

    def predict(self, x):
        """xgboost compares float32 features with float32 split points: so does this (a float64 compare flips a few rows)"""
        out = self.base
        for si, sc, lc, rc, dl in self.trees:
            n = 0
            while lc[n] != -1:
                v = np.float32(x[si[n]])
                if np.isnan(v): n = lc[n] if dl[n] else rc[n]
                else: n = lc[n] if v < np.float32(sc[n]) else rc[n]
            out += sc[n]
        return out


def score(M, booster, ca, aa, kind, size, band, base_pred, nl=None):
    return booster.predict(features(M, ca, aa, kind, size, band, base_pred, nl))


def scorer(M, booster, size, band, aa, pr, nl=None):
    """the callable moves.generate(rerank=...) takes: (c, kind) -> predicted log r vs aa (lower = better).
    pr: the size's fitted Pred (its kl gives the base prediction)."""
    nl = nl or n_layers(M)
    def f(c, kind):
        base = math.log(pr.kl(c) / pr.kl(aa))
        return score(M, booster, c, aa, kind, size, band, base, nl)
    return f
