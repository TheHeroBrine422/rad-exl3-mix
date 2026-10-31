#!/usr/bin/env python3
"""rr.py DUMP.npz [OUT.json] -- residual re-ranker study (2026-10-31): can a learned model correct the structured
kfit2 model's KLD predictions?

Target   y = log(measured KLD) - log(out-of-fold kfit2 prediction)  (oos: the structured model never saw the row)
Features (per measurement, from the dump's error vectors; rel = log(err / base err), bf16 -> err floor):
  - mean rel per (tensor group x 8-layer band)              48
  - rel of every edge (group, layer) in layers 0-3 / 60-63  up to 32
  - head rel; embedding level one-hot; option-family shares (ours_lo / mid / hi / head / plain / sc)
  - slice one-hot, real bpw, log(kfit2 prediction)
Models: none (y_hat = 0), ridge, Gaussian process (RBF + white noise, standardised inputs), XGBoost (shallow).
CV: GroupKFold(5) over sources (round / stream fit / swap batch), so a fit's near-duplicate mean / Thompson / local-
search mixes never sit on both sides. Scored on whole mixes (knap-*) only -- the re-ranker's job:
  log-RMSE of the corrected prediction, and pairwise ranking among mixes of the same slice and target whose measured
  KLDs differ by > 3% (the decision a re-ranker makes)."""
import json, re, sys, warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import xgboost as xgb
warnings.filterwarnings("ignore")

d = np.load(sys.argv[1], allow_pickle=False)
EV, FV, LV, SV, KM, BPW, oos = d["EV"], d["FV"], d["LV"], d["SV"], d["KM"], d["BPW"], d["oos"]
TAGS, W, layer, grp, eb = list(d["TAGS"]), list(d["W"]), d["layer"], d["grp"], d["eb"]
NG = int(grp.max()) + 1; NL = int(layer[grp >= 0].max()) + 1
keep = np.isfinite(oos) & (oos > 0)
y = np.log(KM) - np.log(np.where(keep, oos, 1))

floor = np.where(eb > 0, eb, 1) * 1e-4
rel = np.log((np.maximum(EV, 0) + floor) / np.where(eb > 0, eb, 1))          # [N, T]
feats, names = [], []
band = layer // 8
for g in range(NG):
    for b in range(NL // 8):
        m = (grp == g) & (band == b)
        if m.any(): feats.append(rel[:, m].mean(1)); names.append(f"g{g}b{b}")
edge_layers = list(range(4)) + list(range(NL - 4, NL))
for g in range(NG):
    for l in edge_layers:
        m = (grp == g) & (layer == l)
        if m.any(): feats.append(rel[:, m].mean(1)); names.append(f"g{g}L{l}")
hm = grp < 0
feats.append(rel[:, hm].mean(1)); names.append("head")
for j in range(int(LV.max()) + 1): feats.append((LV == j).astype(float)); names.append(f"emb{j}")
for f in range(int(FV.max()) + 1): feats.append((FV == f).mean(1)); names.append(f"fam{f}")
for s in range(int(SV.max()) + 1): feats.append((SV == s).astype(float)); names.append(f"slice{s}")
feats += [BPW, np.log(np.where(keep, oos, 1))]; names += ["bpw", "logpred"]
X = np.stack(feats, 1)


def source(t):
    m = re.match(r"(?:a|f2|f3)-knap-(r\d+|s\d+|k2)", t)
    if m: return m.group(1)
    m = re.match(r"(cal|emb|head|xp|dz|bf16test|base)", t.split("-", 1)[-1] if t[:2] in ("a-", "f2", "f3") else t)
    return m.group(1) if m else t.split("-")[0]


src = np.array([source(t) for t in TAGS])
ismix = np.array([bool(re.match(r"(a|f2|f3)-knap-", t)) for t in TAGS])
idx = np.where(keep)[0]
tgt = np.array([float(re.search(r"-([\d.]+)$", t).group(1)) if ismix[i] else -1 for i, t in enumerate(TAGS)])


def models():
    return {
        "none": None,
        "ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 25)),
        "gp": lambda: GaussianProcessRegressor(ConstantKernel(0.05) * RBF(length_scale=np.full(X.shape[1], 5.0),
                                                 length_scale_bounds=(0.3, 1e3)) + WhiteKernel(0.01), normalize_y=True,
                                                 n_restarts_optimizer=0),
        "xgb": lambda: xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                        colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0, n_jobs=8),
    }


out = {}
gkf = GroupKFold(n_splits=5)
for name, mk in models().items():
    yh = np.zeros(len(TAGS))
    if mk is not None:
        for tr, te in gkf.split(idx, groups=src[idx]):
            tr, te = idx[tr], idx[te]
            sc = StandardScaler().fit(X[tr])
            mdl = mk(); mdl.fit(sc.transform(X[tr]), y[tr]); yh[te] = mdl.predict(sc.transform(X[te]))
    mi = idx[ismix[idx]]
    err = y[mi] - yh[mi]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pred = np.log(oos[mi]) + yh[mi]; meas = np.log(KM[mi])
    ok = tot = 0
    for a in range(len(mi)):
        for b in range(a + 1, len(mi)):
            i, j = mi[a], mi[b]
            if SV[i] != SV[j] or tgt[i] != tgt[j] or abs(meas[a] - meas[b]) <= 0.03: continue
            tot += 1; ok += (pred[a] < pred[b]) == (meas[a] < meas[b])
    out[name] = {"mix_logrmse": rmse, "pair_ok": ok, "pairs": tot}
    print(f"{name:6} mixes: log-RMSE {rmse:.4f}   same-target pair ranking {ok}/{tot} ({ok / max(1, tot):.1%})", flush=True)
print(f"n={len(idx)} measurements ({ismix[idx].sum()} mixes), {X.shape[1]} features, {len(set(src[idx]))} source groups")
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], "w"), indent=1)
