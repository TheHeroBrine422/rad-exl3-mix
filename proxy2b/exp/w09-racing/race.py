#!/usr/bin/env python3
"""race.py GOLDEN SCORES [opts] -- CPU replay of two search-eval ideas on per-row score files (ranked.md G3, G6).

SCORES: a directory of per-row score files (<tag>.f32, [rows, 4] f32, column 0 = KL: the stream's evq/scores or
evalq.py --out), matched to GOLDEN (kld.json, gen.u8) by row count. Every number is per document: the gen-masked KL
sum of each document (the slice mean is sum / count over all documents, so comparisons are of sums).

Pairs (candidate B vs incumbent A; "accept" = B's slice mean below A's, the stream's best-so-far rule):
  --results RES.jsonl   the V2 stream: every stream candidate on this golden (tags [a|f2|f3]-knap-...-sc4.00-e4-T, in
                        results order) against the best mix measured before it at the same target on the same slice
  (default)             every ordered pair of score files with the same leading 'k<digit>' (0.8B proxy arms; self-test)

G3, racing (group-sequential early rejection): documents in a fixed random order, interleaved by category so every
prefix has the slice's category mix; stage ends at --stages token fractions. At a stage the full-slice sum of B is
predicted from the documents seen: ratio estimator B_rest = R_S * A_rest (A is known on every document: the incumbent
was scored in full), optionally plus the elasticity covariate (below) and a prior from the model's predictions
(--pred); variant "cat" fits a ratio per category, shrunk to the pooled one by LAM = 3 pseudo-documents. Predictive variance = residual variance (proportional to A_d, --vw) over the unseen documents + the
uncertainty of the fitted ratio. P(win) = P(B_full < A_full | seen); reject when P(win) < alpha. Cost of one eval =
T0 + tokens / RATE, T0 paid once (a watcher stops the running eval: the recorder pwrites each row as it is scored).
Reported per alpha: eval time saved, wrong rejections (rejected but the full slice would have accepted) as a share of
the accepts, and the KL gain those wrong rejections forfeited.

G6, variance reduction of a full-slice paired comparison, se of log(sum B / sum A) by the delta method:
  strat-ratio  stratified, a separate ratio per category (category effects are multiplicative: a swap hurts code
               documents in proportion to their KL); needs each category's KL total of A over the population
  plain        documents iid (= pcmp.py's document bootstrap, checked in the self-test)
  strat        stratified by category, proportional allocation (what a category-matched slice is)
  neyman       stratified, documents per category in proportion to N_h S_h / sqrt(cost_h) at equal tokens; S_h from
               the first half of the pairs (time order), applied to the second half
  cv-elast     control variate x_d = A_d (g_d - mean g): g_d = a document's elasticity (how far its per-document ratio
               moves, relative to the slice's, in earlier comparisons); needs the covariate's total over the population:
               known for racing (population = the slice), NOT for slice-vs-pool unless A is scored on the whole pool
Output: a table on stdout and one JSON line (--out FILE, appended)."""
import argparse, glob, json, math, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("golden"); ap.add_argument("scores")
ap.add_argument("--results"); ap.add_argument("--pred", help="glob of frontier-*.txt (V2 kfit2 predictions)")
ap.add_argument("--corpus", help="calib.jsonl: categories via docs[].orig (else from the source name)")
ap.add_argument("--stages", default="0.1,0.2,0.35,0.5,0.7")
ap.add_argument("--alphas", default="0.001,0.003,0.01,0.02,0.05")
ap.add_argument("--orders", type=int, default=20); ap.add_argument("--vw", default="A", choices=("A", "n", "A2"))
ap.add_argument("--t0", default="80,15"); ap.add_argument("--rate", type=float, default=2500.0, help="tokens / s")
ap.add_argument("--hist", type=int, default=50, help="earlier pairs averaged into a document's elasticity")
ap.add_argument("--maxpairs", type=int, default=3000); ap.add_argument("--out"); ap.add_argument("--label", default="")
a = ap.parse_args()
rng = np.random.default_rng(0)

# ---- golden: documents, scored-row counts, tokens (eval cost), categories
m = json.load(open(f"{a.golden}/kld.json")); R = m["rows"]; docs = m["docs"]; D = len(docs)
gname = os.path.basename(a.golden.rstrip("/"))
gen = np.fromfile(f"{a.golden}/gen.u8", np.uint8).astype(bool)
doc = np.zeros(R, np.int64)
for i, d in enumerate(docs): doc[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]] = i
ngen = np.bincount(doc[gen], None, D); tok = np.array([d["tokens"] for d in docs], float)
SRC2CAT = {"hermes-fc": "tools", "glaive-fc-v2": "tools", "when2call": "tools", "wildchat": "chat", "smoltalk2": "chat",
           "stackv2-edu": "code", "the-stack-smol-xl": "code", "wikimedia": "encyclopedic", "fineweb-edu": "web",
           "fineweb2-hq": "multilingual", "wikipedia": "multilingual", "stackexchange": "qa", "finemath-4plus": "math",
           "libretexts": "math", "openthoughts3": "reasoning", "doab": "books", "pg19": "books", "gutenberg": "books",
           "pubmed": "science", "arxiv": "science", "peS2o": "science", "caselaw": "legal", "swe-smith": "agentic",
           "nemotron-swe": "agentic", "swe-rebench-openhands": "agentic", "agent-trace": "long-agentic",
           "pg19-long": "long", "arxiv-long": "long"}
if a.corpus:
    cc = [json.loads(l)["category"] for l in open(a.corpus)]; cat = [cc[d["orig"]] for d in docs]
else:
    cat = [SRC2CAT.get(d["source"].split("/")[0], "other") for d in docs]
cats = sorted(set(cat)); ci = np.array([cats.index(c) for c in cat])
# strata with < 2 documents go to one pooled stratum
small = [k for k in range(len(cats)) if (ci == k).sum() < 2]
si = np.where(np.isin(ci, small), len(cats), ci); strata = sorted(set(si))

# ---- per-document sums of every score file on this golden
def load(p):
    if os.path.getsize(p) != R * 16: return None
    s = np.fromfile(p, np.float32).reshape(R, 4)[:, 0].astype(np.float64)
    if not np.isfinite(s[gen]).all() or (s[gen] > 0).mean() < 0.99: return None
    return np.bincount(doc[gen], s[gen], D)
S = {}
for p in sorted(glob.glob(f"{a.scores}/*.f32")):
    v = load(p)
    if v is not None: S[os.path.basename(p)[:-4]] = v
pairs = []                                                       # (incumbent A, candidate B), time order
if a.results:
    KR = re.compile(r"^(a|f2|f3)-knap-(?:r\d+|s\d+)(?:oo)?(?:-(?:ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)$")
    best = {}
    for l in open(a.results):
        r = json.loads(l); t = r.get("tag", "")
        if t not in S or r.get("golden", gname) != gname: continue          # early lines carry no golden: positions decide
        if "positions" in r and r["positions"] != int(gen.sum()): continue
        k = KR.match(t)
        if not k: continue
        tg = k.group(2)
        if tg in best: pairs.append((best[tg], t))
        if tg not in best or S[t].sum() < S[best[tg]].sum(): best[tg] = t
else:
    grp = {}
    for t in S: grp.setdefault(re.match(r"k\d", t).group(0) if re.match(r"k\d", t) else "x", []).append(t)
    pairs = [(x, y) for g in grp.values() for x in g for y in g if x != y]
    if len(pairs) > a.maxpairs: pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), a.maxpairs, replace=False))]
pairs = [(x, y) for x, y in pairs if not np.array_equal(S[x], S[y])]      # identical arms: nothing to race
P = len(pairs)
print(f"{gname}: {D} docs, {int(ngen.sum())} scored rows, {int(tok.sum())} tokens, {len(S)} score files, {P} pairs; "
      f"categories {dict(zip(cats, np.bincount(ci, None, len(cats)).astype(int).tolist()))}", flush=True)
if P < 10: sys.exit("too few pairs")
A_ = np.stack([S[x] for x, _ in pairs]); B_ = np.stack([S[y] for _, y in pairs])
Lfull = np.log(B_.sum(1) / A_.sum(1)); acc = Lfull < 0

# ---- elasticity: a document's relative ratio deviation in earlier comparisons (time order; proxy: other pairs)
def elast_of(idx):
    with np.errstate(divide="ignore", invalid="ignore"):
        Rq = B_[idx].sum(1) / A_[idx].sum(1)
        u = (B_[idx] / A_[idx] - Rq[:, None]) / (Rq - 1)[:, None]
    u = u[np.abs(Rq - 1) > 0.02]
    if len(u) == 0: return np.zeros(D)
    g = np.nan_to_num(np.clip(u, -5, 5)).mean(0)
    return g
G = np.zeros((P, D)); hasg = np.zeros(P, bool)
for p in range(P):
    if a.results: idx = np.arange(max(0, p - a.hist), p)
    else:
        idx = np.array([q for q in rng.choice(P, min(P, 4 * a.hist), replace=False)
                        if not set(pairs[q]) & set(pairs[p])][:a.hist], int)
    if len(idx) >= 5: G[p] = elast_of(idx); hasg[p] = True

# ---- prior from the model's predictions (V2 frontier reports; slice offsets cancel in a same-slice ratio)
prior = np.full(P, np.nan)
if a.pred:
    pr = {}
    for f in glob.glob(a.pred):
        for l in open(f):
            k = re.match(r"^(knap-\S+)\s+bpw [\d.]+\s+(?:pred KLD|pred \(mean model\)) ([\d.]+)", l) or \
                re.match(r"^(knap-\S+)\s+bpw [\d.]+\s+from .*pred [\d.]+ -> ([\d.]+)", l)
            if k: pr[k.group(1)] = float(k.group(2))
    for p, (x, y) in enumerate(pairs):
        px, py = pr.get(x.split("-", 1)[1]), pr.get(y.split("-", 1)[1])
        if px and py: prior[p] = math.log(py / px)
    ok = np.isfinite(prior)
    print(f"predictions for {ok.sum()} of {P} pairs", flush=True)

def vweight(Ad):
    return {"A": Ad, "n": ngen.astype(float), "A2": Ad ** 2}[a.vw]

# ======================= G6: full-slice standard errors
def se_parts(Ad, Bd, g):
    Rr = Bd.sum() / Ad.sum(); e = Bd - Rr * Ad
    v0 = D / (D - 1) * (e ** 2).sum()
    vs = sum(((si == h).sum() / max((si == h).sum() - 1, 1)) * ((e[si == h] - e[si == h].mean()) ** 2).sum() for h in strata)
    er = np.concatenate([Bd[si == h] - Bd[si == h].sum() / max(Ad[si == h].sum(), 1e-300) * Ad[si == h] for h in strata])
    nh_ = np.concatenate([np.full((si == h).sum(), (si == h).sum() / max((si == h).sum() - 1, 1)) for h in strata])
    vr = (nh_ * er ** 2).sum()
    x = Ad * (g - (Ad * g).sum() / Ad.sum())
    th = (e * x).sum() / max((x * x).sum(), 1e-300); ec = e - th * x
    vc = D / (D - 2) * (ec ** 2).sum()
    Sh = np.array([e[si == h].std(ddof=1) if (si == h).sum() > 1 else 0.0 for h in strata])
    return Bd.sum(), v0, vs, vc, Sh, vr
rows = [se_parts(A_[p], B_[p], G[p]) for p in range(P)]
Bs = np.array([r[0] for r in rows]); v0 = np.array([r[1] for r in rows]); vs = np.array([r[2] for r in rows])
vc = np.array([r[3] for r in rows]); Sh = np.array([r[4] for r in rows]); vr = np.array([r[5] for r in rows])
se0 = np.sqrt(v0) / Bs
Nh = np.array([(si == h).sum() for h in strata], float); ch = np.array([tok[si == h].mean() for h in strata])
half = P // 2
Str = np.sqrt(np.mean((Sh[:half] / np.sqrt(v0[:half] / D)[:, None]) ** 2, 0)) + 1e-12   # train: relative S_h
def neyman_ratio(p):
    Ste = Sh[p]
    vn = (Nh * Str * np.sqrt(ch)).sum() * (Nh * Ste ** 2 * np.sqrt(ch) / Str).sum()
    vp = (Nh * ch).sum() * (Nh * Ste ** 2).sum()
    return vn / vp if vp > 0 else np.nan
ney = np.array([neyman_ratio(p) for p in range(half, P)])
g6 = {"se_plain_median": float(np.median(se0)),
      "se_ratio_strat": float(np.median(np.sqrt(vs / v0))),
      "se_ratio_strat_ratio": float(np.median(np.sqrt(vr / v0))),
      "se_ratio_neyman": float(np.nanmedian(np.sqrt(ney * vs[half:] / v0[half:]))),
      "se_ratio_cv_elast": float(np.median(np.sqrt(vc[hasg] / v0[hasg]))) if hasg.any() else None,
      "neyman_alloc_rel_docs": {cats[h] if h < len(cats) else "pooled": round(float(Str[j] / np.sqrt(ch[j]) * np.sqrt(ch).mean() / Str.mean()), 3)
                                for j, h in enumerate(strata)}}
print("\nG6  full-slice se of log(B/A), median over pairs:")
print(f"  plain {g6['se_plain_median']:.4f}   ratio to plain: strat {g6['se_ratio_strat']:.3f}  strat-ratio {g6['se_ratio_strat_ratio']:.3f}  "
      f"neyman(equal tokens) {g6['se_ratio_neyman']:.3f}  cv-elasticity {g6['se_ratio_cv_elast']}")

# ======================= G3: racing replay
st = [float(x) for x in a.stages.split(",")]; alphas = [float(x) for x in a.alphas.split(",")]
T0s = [float(x) for x in a.t0.split(",")]
from math import erf
Phi = lambda z: 0.5 * (1 + np.vectorize(erf)(z / math.sqrt(2)))
def order():
    """category-interleaved random order: key = (rank in category + U) / size"""
    key = np.empty(D)
    for k in set(ci.tolist()):
        ix = np.flatnonzero(ci == k); rng.shuffle(ix); key[ix] = (np.arange(len(ix)) + rng.random(len(ix))) / len(ix)
    return np.argsort(key)
VARIANTS = ["ratio", "cat", "elast"] + (["prior"] if a.pred else [])
def pwin(Ad, Bd, seen, g, variant, pri):
    """P(sum B < sum A over all documents | the seen ones)"""
    rest = ~seen; As, Bsn = Ad[seen].sum(), Bd[seen].sum(); Ar = Ad[rest].sum()
    if As <= 0: return 1.0
    Rh = Bsn / As; e = Bd[seen] - Rh * Ad[seen]; brest = Rh * Ar
    if variant == "elast":
        gc = g - (Ad[seen] * g[seen]).sum() / As; x = Ad[seen] * gc[seen]
        th = (e * x).sum() / max((x * x).sum(), 1e-300); e = e - th * x; brest += th * (Ad[rest] * gc[rest]).sum()
    if variant == "cat":     # a ratio per category, shrunk to the pooled one with LAM pseudo-documents of mean A
        Ab = Ad.mean(); brest = 0.0; Rd = np.empty(D)
        for h in strata:
            k = si == h; As_h = Ad[seen & k].sum()
            Rd[k] = (Bd[seen & k].sum() + LAM * Rh * Ab) / (As_h + LAM * Ab)
            brest += Rd[k][0] * Ad[rest & k].sum()
        e = Bd[seen] - Rd[seen] * Ad[seen]
    w = vweight(Ad); s2 = (e ** 2).sum() / max(w[seen].sum(), 1e-300)
    if variant == "cat":     # with --vw A: var(e_d) = s2 A_d -> var(R_h) ~ s2 / (A_S,h + LAM A)
        nh = sum(bool((seen & (si == h)).any()) for h in strata); s2 *= seen.sum() / max(seen.sum() - nh, 1)   # fitted ratios
        V = s2 * w[rest].sum()
        for h in strata:
            k = si == h; Ash = Ad[seen & k].sum(); ws = LAM * Ad.mean() / (Ash + LAM * Ad.mean())   # shrink weight
            V += s2 / (Ash + LAM * Ad.mean()) * Ad[rest & k].sum() ** 2 + (ws * Ad[rest & k].sum()) ** 2 * s2 / As
        return float(Phi(-(Bsn + brest - Ad.sum()) / math.sqrt(max(V, 1e-300))))
    vR = (e ** 2).sum() / As ** 2 * seen.sum() / max(seen.sum() - 1, 1)
    if variant == "prior" and np.isfinite(pri):
        # precision-weighted ratio for the unseen documents: data R_S (var vR) and the model's ratio (var tau^2 R^2)
        Rp = math.exp(pri); vp = (TAU * Rp) ** 2; Rc = (Rh / vR + Rp / vp) / (1 / vR + 1 / vp)
        brest = brest - Rh * Ar + Rc * Ar; vR = 1 / (1 / vR + 1 / vp)
    V = s2 * w[rest].sum() + vR * Ar ** 2
    diff = Bsn + brest - Ad.sum()
    return float(Phi(-diff / math.sqrt(max(V, 1e-300))))
TAU = 0.0
LAM = 3.0
if a.pred and np.isfinite(prior).sum() > 10:
    ok = np.isfinite(prior); TAU = math.sqrt(max(np.var(Lfull[ok] - prior[ok]) - np.median(se0[ok]) ** 2, 1e-6))
    print(f"prior: model log-ratio error tau {TAU:.4f} (sd of measured - predicted, minus the median slice se)")
orders = [order() for _ in range(a.orders)]
ctok = [np.cumsum(tok[o]) / tok.sum() for o in orders]
cut = [[int(np.searchsorted(c, f)) + 1 for f in st] for c in ctok]          # documents seen at each stage
PW = {v: np.ones((a.orders, P, len(st))) for v in VARIANTS}
for oi, o in enumerate(orders):
    for k, nd in enumerate(cut[oi]):
        seen = np.zeros(D, bool); seen[o[:nd]] = True
        for p in range(P):
            for v in VARIANTS:
                if v == "elast" and not hasg[p]: PW[v][oi, p, k] = PW["ratio"][oi, p, k]; continue
                PW[v][oi, p, k] = pwin(A_[p], B_[p], seen, G[p], v, prior[p])
frac_at = np.array([[tok[o[:nd]].sum() / tok.sum() for nd in cut[oi]] for oi, o in enumerate(orders)])  # (O, K)
res = {"label": a.label, "golden": gname, "pairs": P, "accepts": int(acc.sum()), "docs": D, "stages": st,
       "vw": a.vw, "g6": g6, "g3": {}}
print(f"\nG3  racing: {P} candidates, {acc.sum()} accepted on the full slice; stages {st}; {a.orders} orders")
print(f"  {'variant':7s} {'alpha':>6s} {'rej%':>6s} {'wrong/acc%':>10s} {'lost gain':>9s} " +
      " ".join(f"{'saved@T0=' + str(int(t)):>12s}" for t in T0s))
# calibration of P(win) at the first stage: observed accept rate in bins of predicted P
for v in VARIANTS:
    p1 = PW[v][:, :, 0].ravel(); y = np.tile(acc, a.orders)
    cal = [(lo, hi, int(((p1 >= lo) & (p1 < hi)).sum()), float(y[(p1 >= lo) & (p1 < hi)].mean()) if ((p1 >= lo) & (p1 < hi)).any() else None)
           for lo, hi in ((0, .01), (.01, .05), (.05, .2), (.2, .5), (.5, .8), (.8, 1.01))]
    res["g3"][v] = {"calibration_stage1": cal, "by_alpha": {}}
    for al in alphas:
        rej = PW[v] < al                                                    # (O, P, K)
        first = np.where(rej.any(2), rej.argmax(2), -1)                     # stage of rejection, -1 = full eval
        used = np.where(first >= 0, np.take_along_axis(frac_at[:, None, :].repeat(P, 1), np.maximum(first, 0)[..., None], 2)[..., 0], 1.0)
        R_ = first >= 0; wrong = R_ & acc[None]
        gain = np.where(acc, -Lfull, 0.0)
        line = {"rejected": float(R_.mean()), "wrong_per_accept": float(wrong.sum() / max(acc.sum() * a.orders, 1)),
                "lost_gain_share": float((wrong * gain[None]).sum() / max(gain.sum() * a.orders, 1e-300)),
                "tokens_saved": float(1 - used.mean())}
        for t0 in T0s:
            full = t0 + tok.sum() / a.rate
            line[f"time_saved_T0_{int(t0)}"] = float(1 - (t0 + used * tok.sum() / a.rate).mean() / full)
        res["g3"][v]["by_alpha"][str(al)] = line
        print(f"  {v:7s} {al:6.3f} {100 * line['rejected']:6.1f} {100 * line['wrong_per_accept']:10.2f} "
              f"{100 * line['lost_gain_share']:8.2f}% " + " ".join(f"{100 * line[f'time_saved_T0_{int(t)}']:11.1f}%" for t in T0s))
    print(f"  {v:7s} stage-1 calibration (P bin: n, observed accept rate): " +
          ", ".join(f"[{lo:g},{hi:g}) {n}:{'-' if r is None else f'{r:.3f}'}" for lo, hi, n, r in cal))
if a.out:
    with open(a.out, "a") as f: f.write(json.dumps(res) + "\n")
