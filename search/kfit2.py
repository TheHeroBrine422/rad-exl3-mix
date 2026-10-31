#!/usr/bin/env python3
"""kfit2.py DIR [TARGETS] -- directional-depth KLD model, fitted to every fit-pool measurement, and an exact DP over
byte buckets for the mix frontier. Successor of kfit.py (see docs/research-mix-allocation.md, R1/R2).

Model (per measurement = assignment weight -> option, embedding -> level):
  cost_w   = s_w * eb_w * (phi_fam * e_w / eb_w)^q          e_w = hproxy error of the chosen option (bf16: 0),
                                                            eb_w the base's; phi_fam: per quantizer family x {trunk, head}
  log s_w  = piecewise linear in layer per group (knots every 8) ; the head has its own s
  X_l      = sum of cost_w over layer l (within a layer additive); the head is layer NL
  P_-1     = c0 + emb[level]                                (unmodelled error + embedding, enters before layer 0)
  P_l      = P_{l-1} * (1 + kappa_l * X_l) + X_l            errors are amplified by the error *downstream* of them
  KLD      = P_NL ;  kappa_l = exp(piecewise linear on knots 0/32/64)  (KFIT_KAPPA=0: additive, 1: scalar, 3: knots)
Fitted by least squares on log KLD with a smoothness prior. Reported: leave-k-out log-RMSE and the decision metrics
(pairwise ranking of measured mixes/containers within 0.15 bpw whose KLDs differ > 3%, CV predictions).

Solver: per layer an MCKP DP over its tensors -> m_l(b) (min X_l at b buckets); then over layers
V_l(B) = min_b V_{l-1}(B-b) * (1 + kappa_l m_l(b)) + m_l(b)  -- exact for this model (monotone in both arguments).
Byte buckets of KFIT_BUCKET bytes (default 1 MiB, rounded to nearest per tensor); the exact bytes are checked and the
budget lowered bucket by bucket until the real size fits.
Optional trust region KFIT_MAXSTEP: options within that many bits of the base option only (0 = off).

Env: KFIT_BASE (sc4.00-e4), KFIT_FOLDS (10), KFIT_TAG ("k2-"), KFIT_KAPPA (3), KFIT_Q (1 = fit q), KFIT_NO_BF16,
     KFIT_SIGMA (0.02), KFIT_SMOOTH (0.5), KFIT_F (container dir), KFIT_EXCLUDE (comma containers, not as options)."""
import math, json, os, re, sys, glob, collections
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
SIGMA = float(os.environ.get("KFIT_SIGMA", 0.02)); SMOOTH = float(os.environ.get("KFIT_SMOOTH", 0.5))
NKAP = int(os.environ.get("KFIT_KAPPA", 3)); FITQ = int(os.environ.get("KFIT_Q", 1))
EXCL = set(filter(None, os.environ.get("KFIT_EXCLUDE", "").split(",")))
OURS_ONLY = os.environ.get("KFIT_OURS_ONLY") == "1"
# with OURS_ONLY: still allow turboderp options with >= TD_MIN bits on that tensor (2026-10-31: ours-only lost at >= 6 bpw)
TD_MIN = float(os.environ.get("KFIT_TD_MIN", "0") or 0)
def ours_ok(c, bits): return not OURS_ONLY or c.startswith("ours-") or (TD_MIN > 0 and bits is not None and bits >= TD_MIN)
H, E = "output.weight", "token_embd.weight"
def rad(t, w=None):
    if not t.startswith("ours-"): return f"{F}/q38-27b-{t}.rad"
    hd = f"{RXM_OURS}/hd-{t[5:]}.rad"
    return hd if w == "output.weight" and os.path.exists(hd) else f"{RXM_OURS}/{t}.rad"
def cname(path):
    m = re.search(r"q38-27b-(.+)\.rad$", path) or re.search(r"/(ours-K\d+)\.rad$", path)
    if m: return m.group(1)
    m = re.search(r"/hd-(K\d+)\.rad$", path)
    if m: return "ours-" + m.group(1)
    raise ValueError(path)
def elevel(c):
    if c == "bf16": return "bf16"
    if c.startswith("ours-"): return "e4"
    m = re.search(r"-e(\d)$", c)
    return f"e{m.group(1)}" if m else None

# ------------------------------------------------------------------ options
opt = collections.defaultdict(dict)
for p in glob.glob(f"{D}/hproxy*.jsonl"):
    for l in open(p):
        try: r = json.loads(l)
        except ValueError:   # e.g. -nan from a bad proxy row (ours-K5 blk.40 GDN, 2026-10-30): that option is left out
            print(f"skipping unparsable proxy row in {p}: {l[:90].strip()}", file=sys.stderr); continue
        if not (math.isfinite(r["err"]) and math.isfinite(r["bytes"])): continue
        opt[r["w"]][r["c"]] = (r["bytes"], r["err"], r.get("bits"))
ls = {}
for l in open(f"{D}/radls.jsonl"):
    r = json.loads(l); ls[cname(r["c"])] = r
W = sorted(opt); WI = {w: i for i, w in enumerate(W)}
eb = np.array([opt[w][BASE][1] for w in W])
fixed = ls[BASE]["total"] - sum(ls[BASE]["w"][w] for w in W) - ls[BASE]["w"][E]
GROUPS = {"attn_in": ["attn_qg", "attn_k", "attn_v"], "attn_out": ["attn_output"], "ssm_in": ["ssm_qkv", "ssm_z"],
          "ssm_out": ["ssm_out"], "ffn_gu": ["ffn_gate_up"], "ffn_down": ["ffn_down"]}
TYPE2G = {t: g for g, ts in GROUPS.items() for t in ts}
SEL = {"attn_in": lambda l: l % 4 == 3, "attn_out": lambda l: l % 4 == 3, "ssm_in": lambda l: l % 4 != 3,
       "ssm_out": lambda l: l % 4 != 3, "ffn_gu": lambda l: True, "ffn_down": lambda l: True}
GL = list(GROUPS)
NL = 1 + max(int(re.match(r"blk\.(\d+)", w).group(1)) for w in W if w.startswith("blk."))
layer = np.array([int(re.match(r"blk\.(\d+)", w).group(1)) if w.startswith("blk.") else NL for w in W])
grp = np.array([GL.index(TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight", w).group(1)]) if w != H else -1 for w in W])
KN = np.unique(np.r_[np.arange(0, NL, 8), NL - 1]); NK = len(KN)
A = np.zeros((len(W), len(GL) * NK + 1))
for i, w in enumerate(W):
    if w == H: A[i, -1] = 1; continue
    j = min(np.searchsorted(KN, layer[i], side="right") - 1, NK - 2); t = (layer[i] - KN[j]) / (KN[j + 1] - KN[j])
    A[i, grp[i] * NK + j] += 1 - t; A[i, grp[i] * NK + j + 1] += t
LM = np.zeros((len(W), NL + 1)); LM[np.arange(len(W)), layer] = 1          # weight -> layer (head = NL)
KK = np.array([0, NL // 2, NL]) if NKAP == 3 else np.array([0, NL])
KA = np.zeros((NL + 1, len(KK)))                                             # layer -> kappa knots
for l in range(NL + 1):
    j = min(np.searchsorted(KK, l, side="right") - 1, len(KK) - 2); t = (l - KK[j]) / (KK[j + 1] - KK[j])
    KA[l, j] += 1 - t; KA[l, j + 1] += t

def family(c, w):
    if c.startswith("plain"): return "plain_head" if w == H else "plain"
    if c.startswith("ours"):
        if w == H: return "ours_head"
        k = int(c[6:]); return "ours_lo" if k <= 3 else "ours_mid" if k <= 6 else "ours_hi"
    return "sc"
FAMS = ["plain", "plain_head", "ours_lo", "ours_mid", "ours_hi", "ours_head"]

# ------------------------------------------------------------------ measurements
res = {}
for l in open(f"{D}/evq-results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r
def assign_base(): return {w: BASE for w in W}, elevel(BASE)
def read_mix(path):
    a, e = assign_base()
    for l in open(path):
        w, p = l.split(); c = cname(p)
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
    if (m := re.match(r"a-((?:knap|xp|dz)-.+)$", tag)):
        p = f"{D}/mixes/{m.group(1)}.mix"
        return read_mix(p) if os.path.exists(p) else None
    return None
def vecs(a):
    e = np.empty(len(W)); f = np.zeros(len(W), int)
    for i, w in enumerate(W):
        c = a[w]
        if c == "bf16": e[i] = 0.0; continue
        if c not in opt[w]: return None
        e[i] = opt[w][c][1]; fa = family(c, w); f[i] = FAMS.index(fa) + 1 if fa in FAMS else 0
    return e, f
SLICES = ["a", "f2", "f3"]                              # calib-A (also base-/cal-/head-/emb-/bf16test-), fit2, fit3
def slice_of(tag):
    p = tag.split("-", 1)[0]
    return SLICES.index(p) if p in SLICES[1:] else 0
MAXKLD = float(os.environ.get("KFIT_MAXKLD", 10))        # 2026-10-30: was 1.5, which dropped the K1 whole-model point and let the fit rate K1 nearly free
meas, skipped = [], []
for tag, r in res.items():
    if tag.startswith(("v-", "w-")): continue          # verify-pool results never enter the fit
    if r["kld"] > MAXKLD: skipped.append(tag); continue
    sl = slice_of(tag)
    m = measurement("a-" + tag.split("-", 1)[1] if sl else tag)
    if m is None: skipped.append(tag); continue
    v = vecs(m[0])
    if v is None: skipped.append(tag); continue
    meas.append((tag, v[0], m[1], r["kld"], v[1], r["bytes"] * 8 / PARAMS, sl))
LEVELS = sorted({m[2] for m in meas if m[2]}); B_LV = LEVELS.index(elevel(BASE))
NM = len(meas)
EV = np.array([m[1] for m in meas]); KM = np.array([m[3] for m in meas]); LV = np.array([LEVELS.index(m[2]) for m in meas])
FV = np.array([m[4] for m in meas]); BPW = np.array([m[5] for m in meas]); TAGS = [m[0] for m in meas]
SV = np.array([m[6] for m in meas]); NS = int(SV.max()) + 1
# points below 1.6 bpw (research floor) count 3x less: they must not drag the shared parameters
WT = np.where(BPW < 1.6, 1 / 3, 1.0)
# clean single-factor embedding swaps (emb-BASE-*) weigh KFIT_EMB_WT (default 8) times more: they are the only direct
# data on the embedding levels, and with ~700 mixed measurements the fit let e3 drift to <= e4 (2026-10-30: r11 / s1-s4
# picked the 3-bit embedding at 5-7 bpw, +~0.0006 KLD there)
WT = WT * np.where([t.startswith("emb-") for t in TAGS], float(os.environ.get("KFIT_EMB_WT", 8)), 1.0)
FAMS_FIT = [f for j, f in enumerate(FAMS) if (FV == j + 1).any()]
print(f"{NM} fit measurements ({len(skipped)} skipped{': ' + ', '.join(sorted(skipped)[:6]) if skipped else ''}); "
      f"families fitted {FAMS_FIT}; embedding levels {LEVELS}", file=sys.stderr)

# ------------------------------------------------------------------ model
NTH = A.shape[1]
def unpack(x):
    i = 0
    th = x[i:i + NTH]; i += NTH
    c0 = np.exp(x[i]); i += 1
    emb = np.zeros(len(LEVELS)); k = [j for j in range(len(LEVELS)) if j != B_LV]; emb[k] = x[i:i + len(k)]; i += len(k)
    q = np.exp(x[i]) if FITQ else 1.0; i += FITQ
    kap = np.exp(x[i:i + len(KK)]) if NKAP else np.zeros(len(KK)); i += len(KK) if NKAP else 0
    phi = np.zeros(len(FAMS) + 1)
    for f in FAMS_FIT: phi[FAMS.index(f) + 1] = x[i]; i += 1
    global SOFF
    SOFF = np.r_[0.0, x[i:i + NS - 1]]
    return th, c0, emb, q, kap, phi
def wcost(th, q, phi, ev, fv):
    s = np.exp(A @ th)
    r = np.where(eb > 0, ev * np.exp(phi[fv]) / np.where(eb > 0, eb, 1), 0)
    return s * eb * r ** q
def run(P, Xl, kl):                                 # P: [N], Xl: [N, NL+1], kl: [NL+1]
    for l in range(Xl.shape[1]): P = P * (1 + kl[l] * Xl[:, l]) + Xl[:, l]
    return P
SOFF = np.zeros(1)
def predict(x, ev, lv, fv, sv=None):
    th, c0, emb, q, kap, phi = unpack(x)
    Xl = wcost(th, q, phi, ev, fv) @ LM
    p = np.maximum(run(c0 + emb[lv], Xl, KA @ kap), 1e-9)
    return p if sv is None else p * np.exp(SOFF[sv])
def prior(x):
    th = x[:NTH]; out = []
    for g in range(len(GL)):
        t = th[g * NK:(g + 1) * NK]; out += [np.diff(t, 2) / SMOOTH, (t - t.mean()) / 4.0]
    out.append(np.atleast_1d((th[:-1].mean() - th[-1]) / 6.0))
    if NKAP == 3: k = x[NTH + 1 + len(LEVELS) - 1 + FITQ:][:3]; out.append(np.diff(k, 2) / 1.0)
    if NS > 1: out.append(x[len(x) - (NS - 1):] / 0.3)
    return np.concatenate(out)
def x0():
    k0 = KM[TAGS.index(f"base-{BASE}")]
    x = [np.full(NTH, np.log(0.7 * k0 / eb.sum())), [np.log(0.1 * k0)], np.zeros(len(LEVELS) - 1)]
    if FITQ: x.append([np.log(1.1)])
    if NKAP: x.append(np.full(len(KK), np.log(3.0)))
    x.append(np.zeros(len(FAMS_FIT)))
    x.append(np.zeros(NS - 1))
    return np.concatenate(x)
def fit(idx, xi=None):
    r = lambda x: np.concatenate([(np.log(predict(x, EV[idx], LV[idx], FV[idx], SV[idx])) - np.log(KM[idx])) * np.sqrt(WT[idx]) / SIGMA,
                                  prior(x)])
    return least_squares(r, x0() if xi is None else xi, method="trf", x_scale="jac", max_nfev=3000).x

X = fit(np.arange(NM))
NF = int(os.environ.get("KFIT_FOLDS", 10))
fold = np.random.default_rng(0).permutation(NM) % NF
bi = TAGS.index(f"base-{BASE}")
oos = np.full(NM, np.nan)
for f in range(NF):
    tr = np.where((fold != f) | (np.arange(NM) == bi))[0]; te = np.where((fold == f) & (np.arange(NM) != bi))[0]
    oos[te] = predict(fit(tr, X), EV[te], LV[te], FV[te], SV[te])
pin = predict(X, EV, LV, FV, SV)
m = ~np.isnan(oos)
cvr = float(np.sqrt(np.mean(np.log(oos[m] / KM[m]) ** 2))); inr = float(np.sqrt(np.mean(np.log(pin / KM) ** 2)))
# decision metric: pairwise order of whole mixes / containers of similar size
whole = [i for i, t in enumerate(TAGS) if re.match(r"(a|f2|f3)-(knap|ref|dz)|base-", t)]
pairs = [(i, j) for a, i in enumerate(whole) for j in whole[a + 1:]
         if SV[i] == SV[j] and abs(BPW[i] - BPW[j]) <= 0.15 and abs(np.log(KM[i] / KM[j])) > 0.03]
cvp = np.where(np.isnan(oos), pin, oos)
ok = sum((cvp[i] < cvp[j]) == (KM[i] < KM[j]) for i, j in pairs)
th, c0, emb, q, kap, phi = unpack(X)
print(f"model kappa={'0' if not NKAP else '3 knots' if NKAP == 3 else '2 knots'} q={'fit' if FITQ else 1}: in-sample log-RMSE {inr:.3f}  CV log-RMSE {cvr:.3f}  "
      f"pair ranking (CV) {ok}/{len(pairs)}  q {q:.3f}  kappa@{list(KK)} " + " ".join(f"{k:.3g}" for k in KA[KK] @ kap if True)
      + "  fam " + " ".join(f"{f} {np.exp(phi[FAMS.index(f) + 1]):.2f}" for f in FAMS_FIT)
      + ("  slice offsets " + " ".join(f"{SLICES[j]} {np.exp(SOFF[j]):.3f}" for j in range(1, NS)) if NS > 1 else ""), file=sys.stderr)
if os.environ.get("KFIT_VERBOSE", "1") != "0":
    print(f"{'measurement':40} {'bpw':>6} {'KLD':>9} {'fit':>9} {'CV pred':>9}", file=sys.stderr)
    for i in np.argsort(KM):
        if TAGS[i].startswith(("a-", "f2-", "f3-", "base-", "emb-", "head-")):
            print(f"{TAGS[i]:40} {BPW[i]:6.3f} {KM[i]:9.5f} {pin[i]:9.5f} {oos[i]:9.5f}", file=sys.stderr)
    print("log10 s at knots " + " ".join(str(int(k)) for k in KN), file=sys.stderr)
    for g in range(len(GL)):
        print(f"  {GL[g]:9} " + " ".join(f"{v:6.2f}" for v in th[g * NK:(g + 1) * NK] / np.log(10)), file=sys.stderr)
    print(f"  head {th[-1] / np.log(10):6.2f}  c0 {c0:.5f}  emb " + " ".join(f"{l}:{e:+.5f}" for l, e in zip(LEVELS, emb)), file=sys.stderr)

# ------------------------------------------------------------------ exact DP over byte buckets
BK = int(os.environ.get("KFIT_BUCKET", 1 << 20)); MAXSTEP = int(os.environ.get("KFIT_MAXSTEP", 0))
BF = ls.get("bf16", {}).get("w", {}) if not os.environ.get("KFIT_NO_BF16") else {}
FIT_FAMS = set(FAMS_FIT)
def allowed(c, w):
    # an option of a quantizer family with no fitted factor would get factor 1 blindly: off unless asked for
    fa = family(c, w)
    return fa not in FAMS or fa in FIT_FAMS or os.environ.get("KFIT_ALLOW_UNFIT")
EO = {}
for c, r in ls.items():
    lv = elevel(c)
    if lv in LEVELS and c not in EXCL and (lv not in EO or r["w"][E] < EO[lv][1]): EO[lv] = (c, r["w"][E])
def frontier(Xp, tgts):
    """[(target, pick, emb level, emb container, real bytes, predicted KLD)] under parameters Xp."""
    th, c0, emb, q, kap, phi = unpack(Xp)
    s_w = np.exp(A @ th); kl = KA @ kap
    def options(i):
        w = W[i]; o = []; bb = opt[w][BASE][2]
        for c, (b, e, bits) in opt[w].items():
            if c in EXCL or not allowed(c, w): continue
            if not ours_ok(c, bits): continue   # A/B: our quants only (weights + head; embedding free), + KFIT_TD_MIN
            if MAXSTEP and bits is not None and bb is not None and abs(bits - bb) > MAXSTEP: continue
            fa = family(c, w); ph = phi[FAMS.index(fa) + 1] if fa in FAMS else 0.0
            o.append((c, b, s_w[i] * eb[i] * (e * np.exp(ph) / eb[i]) ** q if eb[i] > 0 else 0.0))
        # the bf16 head breaks the model (r9 9.00/9.50 mixes: KLD 17, 2026-10-30) -- off until that is understood
        if w in BF and not MAXSTEP and (w != H or os.environ.get("KFIT_HEAD_BF16") == "1"): o.append(("bf16", BF[w], 0.0))
        o.sort(key=lambda x: (x[1], x[2])); p = []
        for x in o:                                    # per-tensor Pareto set (bytes up, cost down)
            if not p or x[2] < p[-1][2] - 1e-15: p.append(x)
        return p
    stages = []                                        # per layer: bucket -> (min X_l, [(weight idx, option, bytes)])
    for l in range(NL + 1):
        cur = {0: (0.0, [])}
        for i in np.where(layer == l)[0]:
            nxt = {}
            for b0, (x, pk) in cur.items():
                for c, b, cst in options(i):
                    nb = b0 + int(round(b / BK))
                    if nb not in nxt or x + cst < nxt[nb][0]: nxt[nb] = (x + cst, pk + [(i, c, b)])
            cur = {}; best = np.inf
            for k in sorted(nxt):
                if nxt[k][0] < best - 1e-18: cur[k] = nxt[k]; best = nxt[k][0]
        stages.append(cur)
    def solve(cap):
        V = np.full(cap + 1, np.inf); arg = []; e_choice = np.full(cap + 1, -1)
        for j, (lv, (c, b)) in enumerate(EO.items()):
            nb = int(round(b / BK))
            if nb <= cap:
                v = c0 + emb[LEVELS.index(lv)]; sl = slice(nb, None); better = v < V[sl]
                V[sl] = np.where(better, v, V[sl]); e_choice[sl] = np.where(better, j, e_choice[sl])
        for l in range(NL + 1):
            NV = np.full(cap + 1, np.inf); ch = np.full(cap + 1, -1, dtype=np.int32)
            for k, (x, _) in stages[l].items():
                if k > cap: continue
                cand = np.full(cap + 1, np.inf); cand[k:] = V[:cap + 1 - k] * (1 + kl[l] * x) + x
                better = cand < NV; NV = np.where(better, cand, NV); ch = np.where(better, k, ch)
            V = NV; arg.append(ch)
        Bend = int(np.argmin(V))
        if not np.isfinite(V[Bend]): return None
        pick = {}; B = Bend; real = fixed
        for l in range(NL, -1, -1):
            k = int(arg[l][B])
            for i, c, b in stages[l][k][1]: pick[W[i]] = c; real += b
            B -= k
        lv, (ec, eb_) = list(EO.items())[int(e_choice[B])]
        return pick, lv, ec, real + eb_
    out = []
    for t in tgts:
        limit = t * PARAMS / 8 - 1; cap = int((limit - fixed) / BK); sol = None     # strictly under (tier edges)
        while cap > 0:
            sol = solve(cap)
            if sol is None or sol[3] <= limit: break
            cap -= max(1, int((sol[3] - limit) / BK))
        if sol is None: print(f"{t}: nothing fits", file=sys.stderr); continue
        pick, lv, ec, real = sol
        out.append((t, pick, lv, ec, real, float(pred_assign(Xp, pick, lv))))
    return out
def pred_assign(Xp, pick, lv):
    ev = np.array([0.0 if pick[w] == "bf16" else opt[w][pick[w]][1] for w in W])
    return predict(Xp, ev[None], np.array([LEVELS.index(lv)]), vecs(pick)[1][None])[0]
def write_mix(name, pick, lv, ec):
    os.makedirs(f"{D}/mixes", exist_ok=True)
    with open(f"{D}/mixes/{name}.mix", "w") as f:
        for w in W:
            if pick[w] != BASE: f.write(f"{w} {rad(pick[w], w)}\n")
        if lv != elevel(BASE): f.write(f"{E} {rad(ec)}\n")

# ------------------------------------------------------------------ per-document bootstrap (posterior draws)
# KFIT_BOOT=n: resample the slice's documents n times, recompute every measurement's KLD from its stored per-row
# scores (KFIT_SCORES dir, KFIT_GOLDEN slice), refit from X. The draws give the predictions' spread and Thompson mixes.
DRAWS = []
NB = int(os.environ.get("KFIT_BOOT", 0))
if NB:
    S = os.environ.get("KFIT_SCORES", f"{RXM_WORK}/evq/scores")
    GOLD = {0: f"{RXM_WORK}/golden-calA", 1: f"{RXM_WORK}/golden-fit2", 2: f"{RXM_WORK}/golden-fit3"}   # per slice, documents resampled within it
    per = {}
    for sl in range(NS):
        gm = json.load(open(f"{GOLD[sl]}/kld.json")); gen = np.fromfile(f"{GOLD[sl]}/gen.u8", np.uint8).astype(bool)
        spans = [(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in gm["docs"]]
        nd = np.array([gen[a:b].sum() for a, b in spans], float)
        idx = np.where(SV == sl)[0]; KD = np.empty((len(idx), len(spans)))
        for j, mi in enumerate(idx):
            s = np.fromfile(f"{S}/{TAGS[mi]}.f32", np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
            KD[j] = [s[a:b][gen[a:b]].sum() for a, b in spans]
        assert np.allclose(KD.sum(1) / nd.sum(), KM[idx], rtol=2e-3), f"slice {sl}: per-row scores do not reproduce the measured KLDs"
        per[sl] = (idx, KD, nd)
    rng = np.random.default_rng(int(os.environ.get("KFIT_SEED", 1)))   # per-fit seed: the stream's Thompson draws differ fit to fit
    KM0 = KM.copy()
    for d in range(NB):
        KM = KM0.copy()
        for sl, (idx, KD, nd) in per.items():
            ix = rng.integers(0, len(nd), len(nd)); KM[idx] = KD[:, ix].sum(1) / nd[ix].sum()
        DRAWS.append(fit(np.arange(NM), X))
    KM = KM0
    print(f"{NB} bootstrap refits done", file=sys.stderr)

# ------------------------------------------------------------------ outputs
TAG = os.environ.get("KFIT_TAG", "k2-")
if targets:
    fr = frontier(X, targets)
    with open(f"{D}/frontier-{TAG.rstrip('-')}.jsonl", "w") as out:
        for t, pick, lv, ec, real, K in fr:
            name = f"knap-{TAG}{BASE}-{t:.2f}"; write_mix(name, pick, lv, ec)
            sd = float(np.std([np.log(pred_assign(xd, pick, lv)) for xd in DRAWS])) if DRAWS else None
            use = collections.Counter(pick.values())
            out.write(json.dumps({"name": name, "target": t, "bpw": real * 8 / PARAMS, "pred_kld": K, "log_sd": sd,
                                  "head": pick[H], "emb": lv, "donors": dict(use.most_common())}) + "\n")
            print(f"{name:28} bpw {real*8/PARAMS:.3f}  pred KLD {K:.5f}{f' +-{sd:.1%}' if sd is not None else ''}  "
                  f"head {pick[H]:13} emb {lv:4} " + " ".join(f"{c}:{n}" for c, n in use.most_common(6)))
    # LOCAL SEARCH (KFIT_LS = mutations per target; 2026-10-31): start from the best MEASURED mix at the target (fit
    # slices, KLD / the fitted slice offset) and let the mean model propose a few same-budget tensor swaps: the
    # single-tensor changes it rates best per byte (upgrades) paid for by the cheapest per-byte downgrades, kept only if
    # the model predicts the whole mutation better than the start. Anchored on measured ground, so the model's bias
    # matters less than in a fresh solve. Mixes: knap-TAGlsK-BASE-T.
    NLS = int(os.environ.get("KFIT_LS", 0))
    if NLS:
        rng_ls = np.random.default_rng(int(os.environ.get("KFIT_SEED", 1)) + 7)
        unpack(X); soff = SOFF                       # the fitted slice offsets (log), [0] = calib-A
        def ok_opt(w, c):
            if c == "bf16": return w in BF and (w != H or os.environ.get("KFIT_HEAD_BF16") == "1")
            if c in EXCL or not allowed(c, w): return False
            return ours_ok(c, opt[w][c][2])
        def wbytes(w, c): return BF[w] if c == "bf16" else opt[w][c][0]
        def rows_pred(picks, lv):
            ev = np.array([[0.0 if p[w] == "bf16" else opt[w][p[w]][1] for w in W] for p in picks])
            fv = np.array([vecs(p)[1] for p in picks])
            return predict(X, ev, np.full(len(picks), LEVELS.index(lv)), fv)
        nls = 0
        for t in targets:
            limit = t * PARAMS / 8 - 1
            best = None
            for j, tg in enumerate(TAGS):
                mm = re.fullmatch(r"(?:a|f2|f3)-knap-.+-sc4\.00-e4-([\d.]+)", tg)
                if not mm or abs(float(mm.group(1)) - t) > 1e-6: continue
                nk = KM[j] / np.exp(soff[SV[j]])
                if best is not None and nk >= best[0]: continue
                sl = slice_of(tg); mm2 = measurement("a-" + tg.split("-", 1)[1] if sl else tg)
                if mm2 is None or mm2[1] not in EO or any(not ok_opt(w, mm2[0][w]) for w in W): continue   # allowed options only
                best = (nk, tg, mm2)
            if best is None: continue
            tg = best[1]; a0, lv0 = best[2]
            b0 = fixed + EO[lv0][1] + sum(wbytes(w, a0[w]) for w in W)
            p0 = float(rows_pred([a0], lv0)[0])
            ch = []                                       # single-tensor changes: (gain, dbytes, w, c)
            for w in W:
                alts = [c for c in list(opt[w]) + ["bf16"] if c != a0[w] and ok_opt(w, c)]
                if not alts: continue
                picks = []
                for c in alts:
                    p = dict(a0); p[w] = c; picks.append(p)
                g = p0 - rows_pred(picks, lv0)
                for c, gg in zip(alts, g): ch.append((float(gg), wbytes(w, c) - wbytes(w, a0[w]), w, c))
            ups = sorted([x for x in ch if x[0] > 0 and x[1] > 0], key=lambda x: -x[0] / x[1])
            downs = sorted([x for x in ch if x[1] < 0], key=lambda x: (-x[0]) / -x[1])
            free = [x for x in ch if x[0] > 0 and x[1] <= 0]
            made = set()
            for k in range(NLS):
                p = dict(a0); used = set(); bb = b0
                cands = ups[k:k + 12]; rng_ls.shuffle(cands)
                for gg, db, w, c in ([max(free, key=lambda x: x[0])] if free and k == 0 else []) + cands[:3]:
                    if w in used: continue
                    p[w] = c; used.add(w); bb += db
                for gg, db, w, c in downs:
                    if bb <= limit: break
                    if w in used: continue
                    p[w] = c; used.add(w); bb += db
                if bb > limit or not used: continue
                key = tuple(sorted((w, p[w]) for w in used))
                if key in made: continue
                made.add(key)
                pk = float(rows_pred([p], lv0)[0])
                if pk >= p0: continue
                name = f"knap-{TAG}ls{k}-{BASE}-{t:.2f}"; write_mix(name, p, lv0, EO[lv0][0]); nls += 1
                print(f"{name:28} bpw {bb*8/PARAMS:.3f}  from {tg} (meas {KM[TAGS.index(tg)]:.5f}, pred {p0:.5f} -> {pk:.5f}, "
                      f"{len(used)} tensors)")
        print(f"local search: {nls} mutations", file=sys.stderr)
    # Thompson mixes: the DP solution under each of the first KFIT_THOMPSON draws (distinct from the mean's only)
    NT = int(os.environ.get("KFIT_THOMPSON", 0))
    mean_by_t = {f[0]: f for f in fr}
    for d, xd in enumerate(DRAWS[:NT]):
        for t, pick, lv, ec, real, _ in frontier(xd, targets):
            if t not in mean_by_t: continue
            p0, lv0 = mean_by_t[t][1], mean_by_t[t][2]
            n = sum(pick[w] != p0[w] for w in W) + (lv != lv0)
            if n == 0: continue
            name = f"knap-{TAG}ts{d}-{BASE}-{t:.2f}"; write_mix(name, pick, lv, ec)
            print(f"{name:28} bpw {real*8/PARAMS:.3f}  pred (mean model) {pred_assign(X, pick, lv):.5f}  "
                  f"differs in {n} tensors")
    # EXPLORATION (KFIT_TILT = tilted solves per fit; 2026-10-31): the exact DP under deliberately MIS-specified
    # sensitivities, so the measured mixes reach regions neither the mean model nor its posterior draws (Thompson,
    # which stays inside the model's uncertainty) would pick, yet every mix is exactly at its target size and optimal
    # for some plausible sensitivity profile. Even k: STRUCTURED (one tensor group's s x0.6 / x1.6, the head's, or a
    # depth slope exp(+-0.5) end to end; rotating with the fit number) -> knap-sN-tlK; odd k: RANDOM direction (a
    # per-group level N(0, KFIT_TILT_LVL=0.35) + smoothed per-knot noise N(0, KFIT_TILT_KN=0.3), in log s)
    # -> knap-sN-rdK. Only at KFIT_TILT_TGTS (the focus targets). Winners teach the fit like any other measurement.
    NTL = int(os.environ.get("KFIT_TILT", 0))
    if NTL:
        TT = [float(x) for x in os.environ.get("KFIT_TILT_TGTS", "2.0,2.5,3.0,3.5,4.0,4.5").split(",")]
        TT = [t for t in TT if t in mean_by_t]
        nfit = int(os.environ.get("KFIT_SEED", 1000)) - 1000
        rngt = np.random.default_rng(int(os.environ.get("KFIT_SEED", 1)) + 7919)
        ST = [(g, m) for g in range(len(GL)) for m in (0.6, 1.6)] + [("head", 0.6), ("head", 1.6), ("depth", -0.5), ("depth", 0.5)]
        LVL, KNS = float(os.environ.get("KFIT_TILT_LVL", 0.35)), float(os.environ.get("KFIT_TILT_KN", 0.3))
        if DRAWS: print(f"tilt: posterior sd of log s (mean over knots) {np.std([xd[:NTH] for xd in DRAWS], 0).mean():.3f}", file=sys.stderr)
        nst = (NTL + 1) // 2
        for k in range(NTL):
            xt = X.copy()
            if k % 2 == 0:
                g, m = ST[(nfit * nst + k // 2) % len(ST)]
                if g == "depth":
                    for gi in range(len(GL)): xt[gi * NK:(gi + 1) * NK] += m * (2 * KN / (NL - 1) - 1)
                    desc = f"depth slope {m:+.1f}"
                elif g == "head": xt[NTH - 1] += np.log(m); desc = f"head x{m}"
                else: xt[g * NK:(g + 1) * NK] += np.log(m); desc = f"{GL[g]} x{m}"
                kind = "tl"
            else:
                d = rngt.normal(0, LVL, (len(GL), 1)) + rngt.normal(0, KNS, (len(GL), NK))
                d[:, 1:-1] = 0.25 * d[:, :-2] + 0.5 * d[:, 1:-1] + 0.25 * d[:, 2:]
                xt[:NTH - 1] += d.ravel(); kind = "rd"
                desc = "random " + " ".join(f"{GL[gi]}:{d[gi].mean():+.2f}" for gi in range(len(GL)))
            for t, pick, lv, ec, real, _ in frontier(xt, TT):
                p0, lv0 = mean_by_t[t][1], mean_by_t[t][2]
                n = sum(pick[w] != p0[w] for w in W) + (lv != lv0)
                if n == 0: continue
                name = f"knap-{TAG}{kind}{k}-{BASE}-{t:.2f}"; write_mix(name, pick, lv, ec)
                print(f"{name:28} bpw {real*8/PARAMS:.3f}  pred (mean model) {pred_assign(X, pick, lv):.5f}  "
                      f"differs in {n} tensors  [{desc}]")
