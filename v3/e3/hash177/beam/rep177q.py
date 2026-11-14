#!/usr/bin/env python3
"""rep177q.py -- RULES-177h amendment 3 readout: the libquant IQR-V4-form head donors (iqh.py). Per set (fit-gen deciding,
search-gen): calib-weighted head-only KL ratios (report119 Set.ratio, document bootstrap) (a) Qmad / Qi8 at K2 and K4,
(b) Qi8-K4 and Qmad-K4 / w122's R16s003-K4 (ours, same code / other encoder), plus Qi8-K2 / R16s003-K2; equal-quality
K on w122's i8 curve (as rep177h.py). Label: "resolved" only if the CI excludes 1 on both sets in the same direction and
both points lie outside ±4.5 % (amendment 3), else "within noise". Env as rep177h.py (SC, HSC, DOCS, DOCS94, SHARE, R119).
(w139, #177)"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.environ["R119"])
import report119 as R

HSC = os.environ["HSC"]
GRID = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5]


def curve(S):                                        # rep177h.py's (it runs on import)
    ks, kl = [], []
    for k in GRID:
        f = f"{HSC}/{S.name}/R16s003-K{k:g}.npz"
        if os.path.exists(f): ks.append(k); kl.append(S.mean(np.load(f)["kl"].astype(np.float64)))
    return np.array(ks), np.log(np.array(kl))


def k_equal(ks, lkl, x):
    lx = np.log(x)
    for i in range(len(ks) - 1):
        a, b = lkl[i], lkl[i + 1]
        if (a - lx) * (b - lx) <= 0 and a != b: return ks[i] + (ks[i + 1] - ks[i]) * (a - lx) / (a - b)
    return float("nan")

PAIRS = [("Qi8-K2", "Qmad-K2"), ("Qi8-K4", "Qmad-K4"), ("R16s003-K4", "Qi8-K4"), ("R16s003-K4", "Qmad-K4"),
         ("R16s003-K2", "Qi8-K2"), ("R16s003-K2", "Qmad-K2")]


def load(S, o):
    return S.kl(o) if S.has(o) else np.load(f"{HSC}/{S.name}/{o}.npz")["kl"].astype(np.float64)


res = {}
for st in ("fit", "search"):
    S = R.Set(st); ks, lkl = curve(S)
    for a, b in PAIRS:
        ka, kb = load(S, a), load(S, b); r = S.ratio(ka, kb)
        res.setdefault(f"{b} / {a}", {})[st] = {"ratio": r, "kl_a": S.mean(ka), "kl_b": S.mean(kb),
                                               "k_equal_b": k_equal(ks, lkl, S.mean(kb))}
for k, d in res.items():
    f, s = d["fit"]["ratio"], d["search"]["ratio"]
    up = f[1] > 1 and s[1] > 1 and f[0] > 1.045 and s[0] > 1.045; dn = f[2] < 1 and s[2] < 1 and f[0] < 0.955 and s[0] < 0.955
    d["label"] = "resolved: costs" if up else "resolved: better" if dn else "within noise"
    R.say(f"{k}: fit {R.fmt(f)} / search {R.fmt(s)} -> {d['label']}; K_eq(b) fit {d['fit']['k_equal_b']:.3f} search {d['search']['k_equal_b']:.3f}")
json.dump(res, open(os.path.join(R.OUT, "rep177q.json"), "w"), indent=1, default=float)
