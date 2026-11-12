#!/usr/bin/env python3
"""kfit3.py --v2 DIR [--v3 ROWS.json] [--diag DIAGSUM.jsonl] [options] -- kfit2's model refit for V3 (w120, 2026-11-12;
ideas round 2 ranks 14 / 15, TODO "Knapsack refit after pass 1"). Scaffolding: tested on existing data, decides nothing.

Model = kfit2's (search/kfit2.py: cost_w = s_w eb_w (phi_fam e_w / eb_w)^q, X_l = sum over layer l, depth compounding
P_l = P_{l-1} (1 + kappa_l X_l) + X_l, KLD = P_NL, optional saturation C (1 - e^(-P/C))) with four additions:
 (a) old swaps as a weak prior: the V2 measurements (kfit2's DIR) enter with weight --v2-wt (default 0.25) and their own
     slice offsets (a / f2 / f3); every V3 slice has its own offset too, so only ratios inside a slice inform the fit.
 (b) per-tensor Fisher prior (#139 diag files, diagsum.py): log s_w = A theta + beta * p_w, p_w = log hbar_w centred
     per group (hbar = sum h e1 / sum e1, the tensor's error-weighted output sensitivity; V2's hproxy err and the diag
     E1 are the same metric: V2 ref == diag R to 1e-6). --beta fit (free) | 1 (the prior as is: the swaps fit only the
     group x depth corrections) | 0 (no prior: kfit2). theta keeps kfit2's smoothness prior; --theta-sd shrinks its
     knot deviations toward the group mean (strong = corrections only).
 (c) this week's 27B mixes (rows27.json: whole-trunk levels with per-tensor proxies, flagged by stamp and slice):
     V3 options are their own family per recipe ("fam" of a row: v3plain / v3hout; phi_v3 + per-group deviations
     with prior sd --v3-group-sd) = the family transfer factors; swj on a row's layers multiplies their gate_up cost
     by f_swj(K) (--swj K:f,..; knap5's anchor curve by default, or fitted with --fit-swj).
 Dry run (w120 11-12, colo): rows with KLD > --v3-maxkld (2) are predicted, not fitted; rows marked "skip" in the rows
 file (w99 K6 in V2 8.00: head / embedding outside the V2 option set) are left out.
 (d) validation: --holdout REGEX fits without the matching rows and predicts them (their slice offset comes from the
     slice's other rows, so a held-out slice needs one kept row: an anchor); --compare runs beta 0 / fit / 1 on the
     same split and prints log-RMSE and the within-slice pairwise order of the held-out rows.
Outputs: --out MODEL.json (kfit2dump-like: per-tensor s, eb, phi, kappa, q, c0, emb; V3 family factors), the fit
report on stderr. CPU only, single-threaded; a fit takes ~1-3 min on colo."""
import argparse, collections, glob, json, math, os, re, sys

import numpy as np
from scipy.optimize import least_squares

H, E = "output.weight", "token_embd.weight"
GROUPS = {"attn_in": ["attn_qg", "attn_k", "attn_v"], "attn_out": ["attn_output"], "ssm_in": ["ssm_qkv", "ssm_z"],
          "ssm_out": ["ssm_out"], "ffn_gu": ["ffn_gate_up"], "ffn_down": ["ffn_down"]}
TYPE2G = {t: g for g, ts in GROUPS.items() for t in ts}
GL = list(GROUPS)
SEL = {"attn_in": lambda l: l % 4 == 3, "attn_out": lambda l: l % 4 == 3, "ssm_in": lambda l: l % 4 != 3,
       "ssm_out": lambda l: l % 4 != 3, "ffn_gu": lambda l: True, "ffn_down": lambda l: True}
V2FAMS = ["plain", "plain_head", "ours_lo", "ours_mid", "ours_hi", "ours_head"]
HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z",
         "linear_attn.out_proj": "ssm_out", "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up",
         "mlp.down_proj": "ffn_down"}
SWJ_DEFAULT = "0.5:0.8936,1:0.7627,1.5:0.8283,2.5:0.8604,4:0.9733"      # knap5 swj-anchors.json solve (w120, 11-12)


def kname(key):
    k = key.split(".")
    if len(k) < 6 or k[2] != "layers": return None
    n = HF2KN.get(".".join(k[4:6]))
    return f"blk.{k[3]}.{n}.weight" if n else None


def v2fam(c, w):
    if c.startswith("plain"): return "plain_head" if w == H else "plain"
    if c.startswith("ours"):
        if w == H: return "ours_head"
        k = int(c[6:]); return "ours_lo" if k <= 3 else "ours_mid" if k <= 6 else "ours_hi"
    return "sc"


def elevel(c):
    if c == "bf16": return "bf16"
    if c.startswith("ours-"): return "e4"
    m = re.search(r"-e(\d)$", c)
    return f"e{m.group(1)}" if m else None


def cname(path):
    m = re.search(r"q38-27b-(.+)\.rad$", path) or re.search(r"/(ours-K\d+)\.rad$", path)
    if m: return m.group(1)
    m = re.search(r"/hd-(K\d+)\.rad$", path)
    if m: return "ours-" + m.group(1)
    raise ValueError(path)


# ------------------------------------------------------------------------------------------------- data
class Data:
    """tensors W, per measurement: error vector e (absolute, H_in metric), family names, emb level, slice, weight,
    swj K per layer (V3), tag"""

    def __init__(self, v2dir, base="sc4.00-e4", maxkld=10.0):
        self.opt = collections.defaultdict(dict)
        for p in glob.glob(f"{v2dir}/hproxy*.jsonl"):
            for l in open(p):
                try: r = json.loads(l)
                except ValueError: continue
                if math.isfinite(r["err"]) and math.isfinite(r["bytes"]): self.opt[r["w"]][r["c"]] = (r["bytes"], r["err"], r.get("bits"), r.get("ref"))
        self.W = sorted(self.opt); self.WI = {w: i for i, w in enumerate(self.W)}; self.base = base
        self.eb = np.array([self.opt[w][base][1] for w in self.W])
        self.ref = {w: next((v[3] for v in self.opt[w].values() if v[3]), None) for w in self.W}
        self.NL = 1 + max(int(re.match(r"blk\.(\d+)", w).group(1)) for w in self.W if w.startswith("blk."))
        self.layer = np.array([int(re.match(r"blk\.(\d+)", w).group(1)) if w.startswith("blk.") else self.NL for w in self.W])
        self.grp = np.array([GL.index(TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight", w).group(1)]) if w != H else -1 for w in self.W])
        self.rows = []                                            # dicts: tag, e, fam (list), emb, slice, wt, swjK (per tensor or None), src
        res = {}
        for l in open(f"{v2dir}/evq-results.jsonl"):
            r = json.loads(l)
            if "fail" not in r: res[r["tag"]] = r
        self.v2dir = v2dir; n_skip = 0; self.Rhf = {}
        for tag, r in res.items():
            if tag.startswith(("v-", "w-", "vt-", "v2-")) or r["kld"] > maxkld: n_skip += 1; continue   # verify pools never fit
            sl = tag.split("-", 1)[0] if tag.split("-", 1)[0] in ("f2", "f3") else "a"
            m = self._v2meas("a-" + tag.split("-", 1)[1] if sl != "a" else tag)
            if m is None: n_skip += 1; continue
            a, e = m
            if any(a[w] != "bf16" and a[w] not in self.opt[w] for w in self.W): n_skip += 1; continue
            ev = np.array([0.0 if a[w] == "bf16" else self.opt[w][a[w]][1] for w in self.W])
            self.rows.append({"tag": tag, "e": ev, "fam": [v2fam(a[w], w) if a[w] != "bf16" else "bf16" for w in self.W], "emb": e,
                              "slice": f"v2{sl}", "kld": r["kld"], "v2": True, "swj": None,
                              "bpw": r["bytes"] * 8 / 26895998464})
        self.n_v2_skip = n_skip

    def _assign_base(self): return {w: self.base for w in self.W}, elevel(self.base)

    def _v2meas(self, tag):
        """kfit2.measurement (the same tags)"""
        B = self.base
        if tag == f"base-{B}": return self._assign_base()
        if (m := re.match(rf"cal-{re.escape(B)}-(.+)-({'|'.join(GROUPS)})-L(\d+)(?:w(\d+))?$", tag)):
            donor, g, b0, bw = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 8)
            a, e = self._assign_base()
            for l in range(b0, b0 + bw):
                if SEL[g](l):
                    for t in GROUPS[g]: a[f"blk.{l}.{t}.weight"] = donor
            return a, e
        if (m := re.match(rf"head-{re.escape(B)}-(.+)$", tag)):
            a, e = self._assign_base(); a[H] = m.group(1); return a, e
        if (m := re.match(rf"emb-{re.escape(B)}-(.+)$", tag)):
            a, e = self._assign_base(); return a, elevel(m.group(1))
        if (m := re.match(r"bf16test-(.+)-L(\d+)down$", tag)):
            a, e = self._assign_base(); a[f"blk.{m.group(2)}.ffn_down.weight"] = "bf16"; return a, e
        if (m := re.match(r"a-ref-(.+)$", tag)):
            c = m.group(1); return {w: c for w in self.W}, elevel(c)
        if (m := re.match(r"a-((?:knap|xp|dz)-.+)$", tag)):
            p = f"{self.v2dir}/mixes/{m.group(1)}.mix"
            if not os.path.exists(p): return None
            a, e = self._assign_base()
            for l in open(p):
                w, pth = l.split(); c = cname(pth)
                if w == E: e = elevel(c)
                else: a[w] = c
            return a, e
        return None

    def add_v3(self, rows_json):
        d = json.load(open(rows_json)); here = os.path.dirname(os.path.abspath(rows_json))
        cache = {}

        def prox(f):
            p = f if os.path.isabs(f) else os.path.normpath(os.path.join(here, "..", "..", "data", "refit", f))
            if p not in cache:
                acc = collections.defaultdict(dict)
                for l in open(p):
                    r = json.loads(l); n = kname(r["key"])
                    if n and r.get("proxy_err") is not None: acc[n][r["key"]] = (r["proxy_err"], float(r["K"]))
                cache[p] = acc
            return cache[p]
        hfref = {}                                                   # V3 proxy_err is relative: abs = sum over the HF parts
        n = 0
        for r in d["rows"]:
            if r.get("skip") or r.get("kld") is None: continue
            tr = r["trunk"]
            ev = np.zeros(len(self.W)); fam = []; swk = np.full(len(self.W), np.nan)
            ok = True
            for i, w in enumerate(self.W):
                if w == H:
                    c = r["head"]
                    if c not in self.opt[w]: ok = False; break
                    ev[i] = self.opt[w][c][1]; fam.append(v2fam(c, w)); continue
                l = int(w.split(".")[1])
                f = tr.get("all") or (tr["even"] if l % 2 == 0 else tr["odd"])
                pr = prox(f).get(w)
                if not pr: ok = False; break
                # relative proxy x the part's own energy R (= V2's hproxy ref of the fused tensor when summed)
                parts = list(pr.items())
                if all(k in self.Rhf for k, _ in parts): ev[i] = sum(p * self.Rhf[k] for k, (p, _) in parts)
                else: ev[i] = sum(p for _, (p, _) in parts) / len(parts) * self.ref[w]   # no diag R: mean relative x fused ref
                fam.append(r.get("fam", "v3hout" if "hout" in r.get("stamp", "") or r["stamp"] in ("w92-fullstack", "159591b2") else "v3plain"))
                K = parts[0][1][1]
                sw = r["swj"]
                if (sw == "all" or (isinstance(sw, list) and K in [float(x) for x in sw])) and w.split(".")[-2] == "ffn_gate_up": swk[i] = K
            if not ok: print(f"kfit3: v3 row {r['tag']} skipped (missing tensors)", file=sys.stderr); continue
            self.rows.append({"tag": r["tag"], "e": ev, "fam": fam, "emb": r["emb"], "slice": r["slice"], "kld": r["kld"],
                              "v2": False, "swj": swk, "stamp": r.get("stamp"), "bpw": None})
            n += 1
        return n


# ------------------------------------------------------------------------------------------------- model
class Model:
    def __init__(self, D, beta="fit", prior_p=None, v2wt=0.25, sigma=0.02, smooth=0.5, theta_sd=4.0, v3_group_sd=0.3,
                 swj=SWJ_DEFAULT, fit_swj=False, sat=False, nkap=3):
        self.D = D; W = D.W; NL = D.NL
        KN = np.unique(np.r_[np.arange(0, NL, 8), NL - 1]); self.NK = NK = len(KN)
        A = np.zeros((len(W), len(GL) * NK + 1))
        for i, w in enumerate(W):
            if w == H: A[i, -1] = 1; continue
            j = min(np.searchsorted(KN, D.layer[i], side="right") - 1, NK - 2); t = (D.layer[i] - KN[j]) / (KN[j + 1] - KN[j])
            A[i, D.grp[i] * NK + j] += 1 - t; A[i, D.grp[i] * NK + j + 1] += t
        self.A = A; self.NTH = A.shape[1]
        self.LM = np.zeros((len(W), NL + 1)); self.LM[np.arange(len(W)), D.layer] = 1
        KK = np.array([0, NL // 2, NL]) if nkap == 3 else np.array([0, NL]); self.KK = KK
        KA = np.zeros((NL + 1, len(KK)))
        for l in range(NL + 1):
            j = min(np.searchsorted(KK, l, side="right") - 1, len(KK) - 2); t = (l - KK[j]) / (KK[j + 1] - KK[j])
            KA[l, j] += 1 - t; KA[l, j + 1] += t
        self.KA = KA
        self.beta_mode = beta; self.p = prior_p if prior_p is not None else np.zeros(len(W))
        self.v2wt, self.sigma, self.smooth, self.theta_sd, self.v3sd, self.sat = v2wt, sigma, smooth, theta_sd, v3_group_sd, sat
        self.swpts = sorted((float(a), float(b)) for a, b in (p.split(":") for p in swj.split(",")))
        self.fit_swj = fit_swj
        R = D.rows
        self.levels = sorted({r["emb"] for r in R if r["emb"]}); self.blv = self.levels.index(elevel(D.base))
        self.slices = sorted({r["slice"] for r in R}, key=lambda s: (s != "v2a", s)); self.NS = len(self.slices)
        fams = set()
        for r in R: fams |= set(r["fam"])
        self.v2fams = [f for f in V2FAMS if f in fams]; self.v3fams = sorted(f for f in fams if f.startswith("v3"))
        self.swk = sorted({float(k) for r in R if r["swj"] is not None for k in r["swj"][~np.isnan(r["swj"])]})

    # parameters: theta | c0 | emb (levels - 1) | q | kappa | phi v2 fams | phi v3 fams | v3 group dev (v3fams x 6) | beta | swj | sat | slice offsets
    def layout(self):
        L = [("theta", self.NTH), ("c0", 1), ("emb", len(self.levels) - 1), ("q", 1), ("kap", len(self.KK)),
             ("phi2", len(self.v2fams)), ("phi3", len(self.v3fams)), ("dev3", len(self.v3fams) * len(GL)),
             ("beta", 1 if self.beta_mode == "fit" else 0), ("swj", len(self.swk) if self.fit_swj else 0),
             ("sat", 1 if self.sat else 0), ("soff", self.NS - 1)]
        out = {}; i = 0
        for n, k in L: out[n] = slice(i, i + k); i += k
        self.N = i
        return out

    def x0(self):
        L = self.layout(); x = np.zeros(self.N)
        bi = next(i for i, r in enumerate(self.D.rows) if r["tag"] == f"base-{self.D.base}")
        k0 = self.D.rows[bi]["kld"]
        x[L["theta"]] = np.log(0.7 * k0 / self.D.eb.sum()); x[L["c0"]] = np.log(0.1 * k0); x[L["q"]] = np.log(1.1)
        x[L["kap"]] = np.log(3.0)
        if self.beta_mode == "fit": x[L["beta"]] = 1.0
        if self.fit_swj: x[L["swj"]] = [np.log(self.swjf(k)) for k in self.swk]
        if self.sat: x[L["sat"]] = np.log(5.0)
        return x

    def swjf(self, K):
        p = self.swpts
        if K <= p[0][0]: return p[0][1]
        if K >= p[-1][0]: return p[-1][1]
        (k0, v0), (k1, v1) = next((a, b) for a, b in zip(p, p[1:]) if a[0] <= K <= b[0])
        return v0 + (K - k0) / (k1 - k0) * (v1 - v0)

    def unpack(self, x):
        L = self.layout()
        beta = x[L["beta"]][0] if self.beta_mode == "fit" else float(self.beta_mode)
        emb = np.zeros(len(self.levels)); k = [j for j in range(len(self.levels)) if j != self.blv]; emb[k] = x[L["emb"]]
        return dict(theta=x[L["theta"]], c0=np.exp(x[L["c0"]][0]), emb=emb, q=np.exp(x[L["q"]][0]), kap=np.exp(x[L["kap"]]),
                    phi2=dict(zip(self.v2fams, x[L["phi2"]])), phi3=dict(zip(self.v3fams, x[L["phi3"]])),
                    dev3=x[L["dev3"]].reshape(len(self.v3fams), len(GL)) if self.v3fams else np.zeros((0, len(GL))),
                    beta=beta, swj=dict(zip(self.swk, map(float, np.exp(x[L["swj"]])))) if self.fit_swj else {k: self.swjf(k) for k in self.swk},
                    sat=np.exp(x[L["sat"]][0]) if self.sat else None, soff=np.r_[0.0, x[L["soff"]]])

    def _mats(self):
        """per-row matrices (built once): errors EV, family index FI (0 = none / sc / bf16, then v2 fams, then v3 fams),
        swj K (nan = no swj), emb level index, slice index"""
        if hasattr(self, "EV"): return
        D = self.D; R = D.rows; fl = ["-"] + self.v2fams + self.v3fams; fi = {f: i for i, f in enumerate(fl)}
        self.FL = fl
        self.EV = np.array([r["e"] for r in R]); self.FI = np.array([[fi.get(f, 0) for f in r["fam"]] for r in R])
        self.SW = np.array([r["swj"] if r["swj"] is not None else np.full(len(D.W), np.nan) for r in R])
        self.LVI = np.array([self.levels.index(r["emb"]) for r in R]); self.SLI = np.array([self.slices.index(r["slice"]) for r in R])
        self.G = np.where(D.grp >= 0, D.grp, 0)

    def predict_rows(self, x, idx):
        self._mats(); u = self.unpack(x); D = self.D; idx = np.asarray(idx, int)
        s = np.exp(self.A @ u["theta"] + u["beta"] * self.p)
        nv2 = len(self.v2fams)
        PH = np.zeros((len(self.FL), len(GL)))                       # family x group log factor
        for j, f in enumerate(self.v2fams): PH[1 + j, :] = u["phi2"][f]
        for j, f in enumerate(self.v3fams): PH[1 + nv2 + j, :] = u["phi3"][f] + u["dev3"][j]
        FI = self.FI[idx]; ph = PH[FI, self.G[None, :]]
        rel = np.where(D.eb > 0, self.EV[idx] * np.exp(ph) / np.where(D.eb > 0, D.eb, 1), 0)
        c = s * D.eb * rel ** u["q"]
        SW = self.SW[idx]
        if self.swk:
            f = np.ones_like(SW)
            for k, v in u["swj"].items(): f[SW == k] = v
            c = c * f
        X = c @ self.LM; kl = self.KA @ u["kap"]
        P = u["c0"] + u["emb"][self.LVI[idx]]
        for l in range(X.shape[1]): P = P * (1 + kl[l] * X[:, l]) + X[:, l]
        P = np.maximum(P, 1e-9)
        if u["sat"]: P = u["sat"] * -np.expm1(-P / u["sat"])
        return P * np.exp(u["soff"][self.SLI[idx]])

    def residuals(self, x, idx):
        D = self.D; L = self.layout(); u = self.unpack(x)
        km = np.array([D.rows[i]["kld"] for i in idx]); wt = np.array([self.v2wt if D.rows[i]["v2"] else 1.0 for i in idx])
        wt = wt * np.array([1 / 3 if (D.rows[i]["bpw"] or 9) < 1.6 else 1.0 for i in idx])
        wt = wt * np.array([8.0 if D.rows[i]["tag"].startswith("emb-") else 1.0 for i in idx])
        res = [(np.log(self.predict_rows(x, idx)) - np.log(km)) * np.sqrt(wt) / self.sigma]
        th = u["theta"]
        for g in range(len(GL)):
            t = th[g * self.NK:(g + 1) * self.NK]; res += [np.diff(t, 2) / self.smooth, (t - t.mean()) / self.theta_sd]
        res.append(np.atleast_1d((th[:-1].mean() - th[-1]) / 6.0))
        if len(self.KK) == 3: res.append(np.diff(np.log(u["kap"]), 2))
        if self.v3fams: res.append(u["dev3"].ravel() / self.v3sd)
        if self.NS > 1: res.append(x[L["soff"]] / 3.0)                  # slice offsets: weak (V3 slices differ a lot)
        if self.fit_swj: res.append((x[L["swj"]] - np.log([self.swjf(k) for k in self.swk])) / 0.3)
        return np.concatenate(res)

    def fit(self, idx, xi=None):
        return least_squares(lambda x: self.residuals(x, idx), self.x0() if xi is None else xi, method="trf", x_scale="jac",
                             max_nfev=int(os.environ.get("KFIT3_NFEV", 400))).x


def prior_vector(D, diagsum):
    """p_w = log hbar_w centred per group (head 0); tensors without a diag value get their group's mean (0)"""
    hb = collections.defaultdict(list)
    for p in diagsum:
        for l in open(p):
            r = json.loads(l); n = kname(r["key"])
            if n and r.get("hbar"): hb[n].append(math.log(r["hbar"]))
    v = np.zeros(len(D.W))
    for i, w in enumerate(D.W):
        if w in hb: v[i] = np.mean(hb[w])
    for g in range(len(GL)):
        m = (D.grp == g) & (v != 0)
        if m.any(): v[D.grp == g] = np.where(v[D.grp == g] != 0, v[D.grp == g] - v[m].mean(), 0.0)
    return v, sum(1 for w in D.W if w in hb)


def evaluate(M, x, idx):
    D = M.D; pr = M.predict_rows(x, idx); km = np.array([D.rows[i]["kld"] for i in idx])
    lr = np.log(pr / km)
    pairs = [(a, b) for ai, a in enumerate(range(len(idx))) for b in range(ai + 1, len(idx))
             if D.rows[idx[a]]["slice"] == D.rows[idx[b]]["slice"] and abs(math.log(km[a] / km[b])) > 0.01]
    ok = sum((pr[a] < pr[b]) == (km[a] < km[b]) for a, b in pairs)
    # slice-relative error: log(pred/meas) minus its slice mean (what a within-slice decision sees)
    sl = collections.defaultdict(list)
    for j, i in enumerate(idx): sl[D.rows[i]["slice"]].append(j)
    rel = np.concatenate([lr[v] - lr[v].mean() for v in sl.values() if len(v) > 1]) if any(len(v) > 1 for v in sl.values()) else np.array([])
    return {"n": len(idx), "logrmse": float(np.sqrt(np.mean(lr ** 2))), "within_slice_rmse": float(np.sqrt(np.mean(rel ** 2))) if len(rel) else None,
            "pairs": f"{ok}/{len(pairs)}", "rows": {D.rows[i]["tag"]: [round(float(km[j]), 6), round(float(pr[j]), 6)] for j, i in enumerate(idx)}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kfit3.py")
    ap.add_argument("--v2", required=True); ap.add_argument("--v3", action="append", default=[]); ap.add_argument("--diag", action="append", default=[])
    ap.add_argument("--beta", default="fit"); ap.add_argument("--v2-wt", type=float, default=0.25)
    ap.add_argument("--theta-sd", type=float, default=4.0); ap.add_argument("--v3-group-sd", type=float, default=0.3)
    ap.add_argument("--swj", default=SWJ_DEFAULT); ap.add_argument("--fit-swj", action="store_true"); ap.add_argument("--sat", action="store_true")
    ap.add_argument("--holdout", default=None, help="regex of row tags fitted out and predicted"); ap.add_argument("--compare", action="store_true")
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="also hold out this fraction of the rows matching --holdout-pool "
                    "(deterministic by tag hash and --seed): a random-split validation on top of --holdout")
    ap.add_argument("--holdout-pool", default="^cal-"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v3-maxkld", type=float, default=2.0, help="V3 rows above this KLD are not fitted, only predicted (w120 dry run: kfit2's compounding mis-predicts the sub-1 rows at KL 2-6 by log-RMSE ~1.8 when fitted on the rest)")
    ap.add_argument("--out"); ap.add_argument("--report")
    a = ap.parse_args(argv)
    D = Data(a.v2)
    for f in a.diag:                                                 # the HF parts' own R (gate / up of a fused tensor)
        for l in open(f):
            r = json.loads(l)
            if r.get("R"): D.Rhf[r["key"]] = r["R"]
    nv2 = len(D.rows); nv3 = sum(D.add_v3(p) for p in a.v3)
    p, npri = prior_vector(D, a.diag) if a.diag else (np.zeros(len(D.W)), 0)
    print(f"kfit3: {nv2} V2 measurements ({D.n_v2_skip} skipped), {nv3} V3 rows, prior on {npri} / {len(D.W)} tensors "
          f"(sd of p {np.std(p[p != 0]) if npri else 0:.2f})", file=sys.stderr)
    tags = [r["tag"] for r in D.rows]
    import hashlib
    hfrac = lambda t: int(hashlib.sha256(f"{a.seed}:{t}".encode()).hexdigest()[:8], 16) / 2 ** 32 < a.holdout_frac
    ho = [i for i, t in enumerate(tags) if (a.holdout and re.search(a.holdout, t)) or (re.search(a.holdout_pool, t) and hfrac(t))]
    big = [i for i, r in enumerate(D.rows) if not r["v2"] and r["kld"] > a.v3_maxkld]
    if big: print(f"kfit3: V3 rows above --v3-maxkld {a.v3_maxkld} not fitted (reported as held out): {[tags[i] for i in big]}", file=sys.stderr)
    big = [i for i in big if i not in ho]
    ho = sorted(set(ho) | set(big)); sel = [i for i in ho if i not in big]
    tr = [i for i in range(len(tags)) if i not in ho]
    arms = [("beta0", "0"), ("beta-fit", "fit"), ("beta1", "1")] if a.compare else [(f"beta-{a.beta}", a.beta)]
    rep = {"n_v2": nv2, "n_v3": nv3, "holdout": a.holdout, "arms": {}}
    for name, bm in arms:
        M = Model(D, beta=bm, prior_p=p, v2wt=a.v2_wt, theta_sd=a.theta_sd, v3_group_sd=a.v3_group_sd, swj=a.swj,
                  fit_swj=a.fit_swj, sat=a.sat)
        x = M.fit(tr); u = M.unpack(x)
        ins = evaluate(M, x, tr); out = evaluate(M, x, sel) if sel else None; outbig = evaluate(M, x, big) if big else None
        v3in = evaluate(M, x, [i for i in tr if not D.rows[i]["v2"]]) if nv3 else None
        line = (f"[{name}] beta {u['beta']:.3f} q {u['q']:.3f} kappa {np.round(M.KA[M.KK] @ u['kap'], 3).tolist()} "
                f"phi3 {({k: round(math.exp(v), 3) for k, v in u['phi3'].items()})} swj {({k: round(v, 3) for k, v in u['swj'].items()})}"
                + (f" sat C {u['sat']:.3g}" if u["sat"] else "")
                + f" | train log-RMSE {ins['logrmse']:.3f} pairs {ins['pairs']}"
                + (f" | V3 rows in-sample within-slice {v3in['within_slice_rmse']}" if v3in else "")
                + (f" | HELD OUT n {out['n']} log-RMSE {out['logrmse']:.3f} within-slice {out['within_slice_rmse']} pairs {out['pairs']}" if out else "")
                + (f" | above v3-maxkld n {outbig['n']} log-RMSE {outbig['logrmse']:.3f}" if outbig else ""))
        print(line, file=sys.stderr)
        for o_ in (out, outbig):
            if not o_ or len(o_["rows"]) > 20: continue
            for t, (m_, p_) in o_["rows"].items(): print(f"    {t:40s} meas {m_:.5f} pred {p_:.5f} ({math.log(p_ / m_):+.3f})", file=sys.stderr)
        rep["arms"][name] = {"beta": u["beta"], "q": u["q"], "phi3": {k: math.exp(v) for k, v in u["phi3"].items()},
                             "swj": u["swj"], "sat": u["sat"], "train": {k: v for k, v in ins.items() if k != "rows"},
                             "heldout": out, "above_maxkld": outbig, "v3_in": v3in}
        if a.out and name == arms[-1][0]:
            s = np.exp(M.A @ u["theta"] + u["beta"] * M.p)
            json.dump({"model": "kfit3", "beta": u["beta"], "q": u["q"], "c0": u["c0"], "kappa": (M.KA @ u["kap"]).tolist(),
                       "phi2": u["phi2"], "phi3": u["phi3"], "dev3": u["dev3"].tolist(), "groups": GL, "swj": u["swj"],
                       "sat": u["sat"], "slices": dict(zip(M.slices, u["soff"].tolist())),
                       "emb": dict(zip(M.levels, u["emb"].tolist())),
                       "tensors": [{"w": w, "layer": int(D.layer[i]), "s": float(s[i]), "eb": float(D.eb[i]), "p": float(M.p[i])}
                                   for i, w in enumerate(D.W)]}, open(a.out, "w"))
    if a.report: json.dump(rep, open(a.report, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
