#!/usr/bin/env python3
"""kfit.py DIR [TARGETS] -- fit KLD(mix) models jointly to every fit-pool measurement, compare them by
cross-validation, and solve the mix frontier with the best one.

DIR holds hproxy*.jsonl (weight x container: bytes, err = tr(dW H dW^T)/rows), radls.jsonl (container entry bytes),
evq-results.jsonl (eval results) and mixes/ (the .mix files of knapsack-mix measurements).

Measurements: every result on the fit pool (calib half A): base-*, cal-* (group/band swaps), head-*, emb-*,
bf16test-*, a-knap-* (mixes), a-ref-* (whole containers). v-* (calib B, verify pool) is never fitted; it is only
reported against with KFIT_REPORT_B=1 (relative to its own base, as a check).

A measurement is an assignment: weight -> container (bf16 = zero error), embedding -> bit level.

Model family (all fitted to log KLD, smoothness prior on the sensitivities across layers):
  s_w        = exp(theta[group, layer]): piecewise linear in the layer index between knots every 8 layers
  cost_w     = s_w * eb_w * (e_w / eb_w)^q            q = 1: linear in the proxy error; q > 1: each tensor's cost
                                                     grows faster than its error (eb_w = the base's error)
  X          = c0 + emb[level] + sum_w cost_w * (1 + c * U_w)   U_w = sum of cost over weights in earlier layers
                                                     (c > 0: errors compound through depth)
  KLD        = X                  (lin)
             = (exp(a X) - 1) / a (mult: multiplicative / compounding over the total)
             = X^p                (pow)
Every combination of {q, c} x {lin, mult, pow} is fitted; leave-k-out CV (KFIT_FOLDS, default 10) picks one
(KFIT_MODEL overrides). The frontier: the separable part (cost_w with U fixed) by Lagrangian multiple-choice
knapsack, U re-linearised around the mix a few times; g() is monotone so it only changes the predictions.

Env: KFIT_BASE (sc4.00-e4), KFIT_MODEL, KFIT_FOLDS, KFIT_TAG (mix name prefix, default "kf-"), KFIT_NO_BF16,
     KFIT_BOOT (bootstrap refits for prediction spread, default 0), KFIT_SIGMA (log-KLD noise, default 0.02),
     KFIT_SMOOTH (prior sd of the second difference of log s per knot, default 0.5)."""
import json, os, re, sys, glob, collections, itertools
import numpy as np
from scipy.optimize import least_squares
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")

D = sys.argv[1]
targets = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
PARAMS = 26895998464
F = os.environ.get("KFIT_F", RXM_MODELS)
BASE = os.environ.get("KFIT_BASE", "sc4.00-e4")
SIGMA = float(os.environ.get("KFIT_SIGMA", 0.02))
SMOOTH = float(os.environ.get("KFIT_SMOOTH", 0.5))
H, E = "output.weight", "token_embd.weight"
rad = lambda t: f"{F}/q38-27b-{t}.rad" if not t.startswith("ours-") else f"{RXM_OURS}/{t}.rad"
def cname(path):
    m = re.search(r"q38-27b-(.+)\.rad$", path) or re.search(r"/(ours-K\d+)\.rad$", path)
    if m: return m.group(1)
    m = re.search(r"/hd-(K\d+)\.rad$", path)
    if m: return "ours-" + m.group(1)
    raise ValueError(path)
def elevel(c):
    if c == "bf16": return "bf16"
    m = re.search(r"-e(\d)$", c)
    return f"e{m.group(1)}" if m else None

# ------------------------------------------------------------------ options
opt = collections.defaultdict(dict)             # weight -> container -> (bytes, err)
for p in glob.glob(f"{D}/hproxy*.jsonl"):
    for l in open(p):
        r = json.loads(l); opt[r["w"]][r["c"]] = (r["bytes"], r["err"])
ls = {}
for l in open(f"{D}/radls.jsonl"):
    r = json.loads(l); ls[cname(r["c"])] = r
W = sorted(opt)                                  # the EXL3 weights incl. the head
WI = {w: i for i, w in enumerate(W)}
eb = np.array([opt[w][BASE][1] for w in W])
fixed = ls[BASE]["total"] - sum(ls[BASE]["w"][w] for w in W) - ls[BASE]["w"][E]

GROUPS = {"attn_in": ["attn_qg", "attn_k", "attn_v"], "attn_out": ["attn_output"], "ssm_in": ["ssm_qkv", "ssm_z"],
          "ssm_out": ["ssm_out"], "ffn_gu": ["ffn_gate_up"], "ffn_down": ["ffn_down"]}
TYPE2G = {t: g for g, ts in GROUPS.items() for t in ts}
SEL = {"attn_in": lambda l: l % 4 == 3, "attn_out": lambda l: l % 4 == 3, "ssm_in": lambda l: l % 4 != 3,
       "ssm_out": lambda l: l % 4 != 3, "ffn_gu": lambda l: True, "ffn_down": lambda l: True}
GL = list(GROUPS)
NL = 1 + max(int(re.match(r"blk\.(\d+)", w).group(1)) for w in W if w.startswith("blk."))
KN = np.unique(np.r_[np.arange(0, NL, 8), NL - 1])     # knots
layer = np.array([int(re.match(r"blk\.(\d+)", w).group(1)) if w.startswith("blk.") else NL for w in W])
grp = np.array([GL.index(TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight", w).group(1)]) if w != H else -1 for w in W])
# interpolation matrix: log s_w = Aw @ theta_knots[group]  (head: its own parameter)
NK = len(KN)
A = np.zeros((len(W), len(GL) * NK + 1))
for i, w in enumerate(W):
    if w == H: A[i, -1] = 1; continue
    j = np.searchsorted(KN, layer[i], side="right") - 1; j = min(j, NK - 2)
    t = (layer[i] - KN[j]) / (KN[j + 1] - KN[j])
    A[i, grp[i] * NK + j] += 1 - t; A[i, grp[i] * NK + j + 1] += t
# depth: U_w = sum of costs of weights in strictly earlier layers (the head comes after everything)
EARLIER = (layer[None, :] < layer[:, None]).astype(float)   # [w, v] = 1 if v is before w

# ------------------------------------------------------------------ measurements
res = {}
for l in open(f"{D}/evq-results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r
def assign_base():
    return {w: BASE for w in W}, elevel(BASE)
def read_mix(path):
    a, e = assign_base()
    for l in open(path):
        w, p = l.split()
        c = cname(p)
        if w == E: e = elevel(c)
        else: a[w] = c
    return a, e
def measurement(tag):
    if tag == f"base-{BASE}": return assign_base()
    if (m := re.match(rf"cal-{re.escape(BASE)}-(.+)-({'|'.join(GROUPS)})-L(\d+)(?:w(\d+))?$", tag)):
        donor, g, b0, bw = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 8)
        a, e = assign_base()
        for l in range(b0, b0 + bw):
            if SEL[g](l):
                for t in GROUPS[g]: a[f"blk.{l}.{t}.weight"] = donor
        return a, e
    if (m := re.match(rf"head-{re.escape(BASE)}-(.+)$", tag)):
        a, e = assign_base(); a[H] = m.group(1); return a, e
    if (m := re.match(rf"emb-{re.escape(BASE)}-(.+)$", tag)):
        a, e = assign_base(); return a, elevel(m.group(1))
    if (m := re.match(r"bf16test-(.+)-L(\d+)down$", tag)):
        a, e = assign_base(); a[f"blk.{m.group(2)}.ffn_down.weight"] = "bf16"; return a, e
    if (m := re.match(r"a-ref-(.+)$", tag)):
        c = m.group(1); return {w: c for w in W}, elevel(c)
    if (m := re.match(r"a-(knap-.+)$", tag)):
        p = f"{D}/mixes/{m.group(1)}.mix"
        return read_mix(p) if os.path.exists(p) else None
    return None
FAMS = ["plain", "ours"]                         # per-family error scale exp(phi), relative to SC (and bf16: 0 error)
def family(c): return "plain" if c.startswith("plain") else "ours" if c.startswith("ours") else "sc"
def fam_vec(a): return np.array([FAMS.index(family(a[w])) + 1 if family(a[w]) in FAMS else 0 for w in W])
def err_vec(a):
    v = np.empty(len(W))
    for i, w in enumerate(W):
        c = a[w]
        if c == "bf16": v[i] = 0.0
        elif c in opt[w]: v[i] = opt[w][c][1]
        else: return None                        # no proxy error for this option (e.g. ours heads before hproxy)
    return v
meas, skipped = [], []
for tag, r in res.items():
    if tag.startswith("v-"): continue
    m = measurement(tag)
    if m is None: skipped.append(tag); continue
    ev = err_vec(m[0])
    if ev is None: skipped.append(tag); continue
    meas.append((tag, ev, m[1], r["kld"], fam_vec(m[0])))
LEVELS = sorted({m[2] for m in meas if m[2]})
NM = len(meas)
EV = np.array([m[1] for m in meas]); KM = np.array([m[3] for m in meas])
LV = np.array([LEVELS.index(m[2]) for m in meas])
FV = np.array([m[4] for m in meas])
FAMS_FIT = [f for j, f in enumerate(FAMS) if (FV == j + 1).any()]
print(f"{NM} fit measurements ({len(skipped)} skipped: {', '.join(sorted(skipped)[:8])}{' ...' if len(skipped) > 8 else ''}); "
      f"embedding levels {LEVELS}", file=sys.stderr)

# ------------------------------------------------------------------ model
# parameters: theta (len(GL)*NK + 1 log sensitivities), log c0, emb offsets (one per level, base level fixed 0),
#             shape: log(q) [conv], log(c) [depth], log(a) [mult] or log(p) [pow]
B_LV = LEVELS.index(elevel(BASE))
def unpack(x, form):
    conv, depth, g, fam = form
    i = 0
    th = x[i:i + A.shape[1]]; i += A.shape[1]
    c0 = np.exp(x[i]); i += 1
    emb = np.zeros(len(LEVELS)); k = [j for j in range(len(LEVELS)) if j != B_LV]
    emb[k] = x[i:i + len(k)]; i += len(k)
    q = np.exp(x[i]) if conv else 1.0; i += conv
    c = np.exp(x[i]) if depth else 0.0; i += depth
    s = np.exp(x[i]) if g != "lin" else None; i += g != "lin"
    phi = np.zeros(len(FAMS) + 1)
    if fam:
        for f in FAMS_FIT: phi[FAMS.index(f) + 1] = x[i]; i += 1
    return th, c0, emb, q, c, s, phi
def nparam(form): return A.shape[1] + 1 + len(LEVELS) - 1 + form[0] + form[1] + (form[2] != "lin") + form[3] * len(FAMS_FIT)
def costs(th, q, ev):
    s = np.exp(A @ th)
    r = np.where(eb > 0, ev / np.where(eb > 0, eb, 1), 0)
    return s * eb * r ** q                       # [.., W]
def predict_X(x, form, ev, lv, fv):
    th, c0, emb, q, c, _, phi = unpack(x, form)
    cw = costs(th, q, ev * np.exp(phi[fv]))
    if c: cw = cw * (1 + c * (cw @ EARLIER.T))
    return c0 + emb[lv] + cw.sum(-1)
def gfun(X, form, s):
    X = np.maximum(X, 1e-9)
    if form[2] == "lin": return X
    if form[2] == "mult": return np.expm1(s * X) / s
    return X ** s
def predict(x, form, ev, lv, fv):
    return gfun(predict_X(x, form, ev, lv, fv), form, unpack(x, form)[5])
def prior_res(x):
    th = x[:A.shape[1]]; out = []
    for g in range(len(GL)):
        t = th[g * NK:(g + 1) * NK]
        out.append(np.diff(t, 2) / SMOOTH)
        out.append((t - t.mean()) / 4.0)          # weak pull toward the group's mean
    out.append(np.atleast_1d((th[:-1].mean() - th[-1]) / 6.0))   # the head: very weak tie to the rest
    return np.concatenate(out)
def x0(form):
    # start: every s equal, chosen so the base's sum matches its KLD (the swaps' scale comes from the fit)
    k0 = KM[[i for i, m in enumerate(meas) if m[0] == f"base-{BASE}"][0]] if any(m[0] == f"base-{BASE}" for m in meas) else KM.min()
    th = np.full(A.shape[1], np.log(0.7 * k0 / eb.sum()))
    x = [th, [np.log(0.1 * k0)], np.zeros(len(LEVELS) - 1)]
    if form[0]: x.append([np.log(1.2)])
    if form[1]: x.append([np.log(1.0)])
    if form[2] == "mult": x.append([np.log(10.0)])
    if form[2] == "pow": x.append([np.log(1.1)])
    if form[3]: x.append(np.zeros(len(FAMS_FIT)))
    return np.concatenate(x)
def fit(form, idx, xinit=None):
    def r(x):
        p = predict(x, form, EV[idx], LV[idx], FV[idx])
        return np.concatenate([(np.log(p) - np.log(KM[idx])) / SIGMA, prior_res(x)])
    sol = least_squares(r, x0(form) if xinit is None else xinit, method="trf", x_scale="jac", max_nfev=4000)
    return sol.x
FORMS = [(cv, dp, g, fm) for fm in (0, 1) for cv in (0, 1) for dp in (0, 1) for g in ("lin", "mult", "pow")]
fname = lambda f: f"{f[2]}{'+conv' if f[0] else ''}{'+depth' if f[1] else ''}{'+fam' if f[3] else ''}"
ONLY = set(filter(None, os.environ.get("KFIT_FORMS", "").split(",")))
if ONLY: FORMS = [f for f in FORMS if fname(f) in ONLY]

rng = np.random.default_rng(0)
NF = int(os.environ.get("KFIT_FOLDS", 10))
fold = rng.permutation(NM) % NF
base_i = [i for i, m in enumerate(meas) if m[0] == f"base-{BASE}"]
cv = {}
want = os.environ.get("KFIT_MODEL")
for form in FORMS:
    if want and fname(form) != want and not os.environ.get("KFIT_CV_ALL"): continue
    full = fit(form, np.arange(NM))
    pin = predict(full, form, EV, LV, FV)
    oos = np.empty(NM)
    for f in range(NF):
        tr = np.where((fold != f) | np.isin(np.arange(NM), base_i))[0]; te = np.where(fold == f)[0]
        te = te[~np.isin(te, base_i)]
        xf = fit(form, tr, full)
        oos[te] = predict(xf, form, EV[te], LV[te], FV[te])
    m = ~np.isin(np.arange(NM), base_i)
    lr = np.log(oos[m] / KM[m]); li = np.log(pin / KM)
    cv[fname(form)] = (form, full, float(np.sqrt(np.mean(lr ** 2))), float(np.sqrt(np.mean(li ** 2))), oos)
    _, _, _, q, c, s, phi = unpack(full, form)
    print(f"{fname(form):22} params {nparam(form):3}  in-sample log-RMSE {cv[fname(form)][3]:.3f}  "
          f"CV log-RMSE {cv[fname(form)][2]:.3f}  q {q:.2f}  c {c:.3g}  shape {s if s is None else round(float(s), 3)}  err scale " + " ".join(f"{f} {np.exp(phi[FAMS.index(f) + 1]):.2f}" for f in FAMS_FIT if form[3]),
          file=sys.stderr)
best = want or min(cv, key=lambda k: cv[k][2])
form, X, _, _, oos = cv[best]
print(f"model: {best}", file=sys.stderr)

# per-measurement view of the chosen model (mixes and whole containers: the points that matter)
pin = predict(X, form, EV, LV, FV)
print(f"{'measurement':40} {'KLD':>9} {'fit':>9} {'CV pred':>9}", file=sys.stderr)
for i in np.argsort(KM):
    t = meas[i][0]
    if t.startswith(("a-", "base-", "emb-", "bf16")) or os.environ.get("KFIT_VERBOSE"):
        print(f"{t:40} {KM[i]:9.5f} {pin[i]:9.5f} {oos[i] if i not in base_i else float('nan'):9.5f}", file=sys.stderr)
th, c0, emb, q, cdep, gs, phi = unpack(X, form)
print("log10 s at knots " + " ".join(str(int(k)) for k in KN), file=sys.stderr)
for g in range(len(GL)):
    print(f"  {GL[g]:9} " + " ".join(f"{v:6.2f}" for v in th[g * NK:(g + 1) * NK] / np.log(10)), file=sys.stderr)
print(f"  head {th[-1] / np.log(10):6.2f}  c0 {c0:.5f}  emb " + " ".join(f"{l}:{e:+.5f}" for l, e in zip(LEVELS, emb)), file=sys.stderr)

# ------------------------------------------------------------------ frontier
if targets:
    bf = ls.get("bf16", {}).get("w", {}) if not os.environ.get("KFIT_NO_BF16") else {}
    ch = []                                          # per weight: [(container, bytes, err)]
    for w in W:
        o = [(c, b, e) for c, (b, e) in opt[w].items()]
        if w in bf: o.append(("bf16", bf[w], 0.0))
        ch.append(o)
    embopt = {}                                      # level -> (container to take it from, bytes)
    for c, r in ls.items():
        lv = elevel(c)
        if lv in LEVELS and lv not in embopt: embopt[lv] = (c, r["w"][E])
    s_w = np.exp(A @ th)
    def solve(lam, mult):
        pick, B = [], fixed
        for i, o in enumerate(ch):
            r = np.array([e * np.exp(phi[FAMS.index(family(c)) + 1] if family(c) in FAMS else 0) for c, _, e in o]) / eb[i] if eb[i] > 0 else np.zeros(len(o))
            cost = s_w[i] * eb[i] * r ** q * mult[i]
            j = int(np.argmin(cost + lam * np.array([b for _, b, _ in o])))
            pick.append(o[j][0]); B += o[j][1]
        lv = min(embopt, key=lambda l: emb[LEVELS.index(l)] + lam * embopt[l][1])
        return pick, lv, B + embopt[lv][1]
    TAG = os.environ.get("KFIT_TAG", "kf-")
    os.makedirs(f"{D}/mixes", exist_ok=True)
    out = open(f"{D}/frontier-{TAG.rstrip('-')}.jsonl", "w")
    lams = np.logspace(-20, -4, 400)
    for t in targets:
        cap = t * PARAMS / 8
        mult = np.ones(len(W))
        for it in range(4 if cdep else 1):
            best_p = None
            for lam in lams[::-1]:                   # from cheap to expensive: the last one that fits
                pick, lv, B = solve(lam, mult)
                if B <= cap: best_p = (pick, lv, B)
                else: break
            if best_p is None: break
            pick, lv, B = best_p
            ev = np.array([0.0 if c == "bf16" else opt[w][c][1] for w, c in zip(W, pick)])
            fv = fam_vec(dict(zip(W, pick)))
            if cdep: mult = 1 + cdep * (costs(th, q, ev * np.exp(phi[fv])) @ EARLIER.T)
        if best_p is None: print(f"{t}: nothing fits", file=sys.stderr); continue
        K = float(predict(X, form, ev[None], np.array([LEVELS.index(lv)]), fv[None])[0])
        name = f"knap-{TAG}{BASE}-{t:.2f}"
        with open(f"{D}/mixes/{name}.mix", "w") as f:
            for w, c in zip(W, pick):
                if c != BASE: f.write(f"{w} {rad(c)}\n")
            if lv != elevel(BASE): f.write(f"{E} {rad(embopt[lv][0])}\n")
        use = collections.Counter(pick)
        rec = {"name": name, "target": t, "bpw": B * 8 / PARAMS, "pred_kld": K, "model": best,
               "head": pick[WI[H]], "emb": lv, "donors": dict(use.most_common())}
        out.write(json.dumps(rec) + "\n")
        print(f"{name:28} bpw {B*8/PARAMS:.3f}  pred KLD {K:.5f}  head {pick[WI[H]]:13} emb {lv:4} "
              + " ".join(f"{c}:{n}" for c, n in use.most_common(6)))
