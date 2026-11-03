#!/usr/bin/env python3
"""f8replay.py RESULTS.jsonl FIT.txt [--w 0.5] [--sim N] -- F8 (ranked.md): posterior-mean vs measured-best selection
replayed on V2's stream, from what is on the VM (data/v2/results.jsonl, data/v2/fit-final.txt). CPU, seconds.

Per fit-slice mix measurement (a / f2 / f3, normalized by fit-final's slice offsets): m = measured KLD, c = kfit2's
cross-validated prediction (fit-final "CV pred": the measurement left out). Posterior estimate in log space:
    log post = w log m + (1 - w) log c        (w = tau^2 / (tau^2 + sigma^2): model vs measurement noise)
Held-out truth: the same mix's ver1 (w-) or ver2 (v2-) eval (not in the fit). Three readings:
 1. calibration: within each (target, held-out slice) group, which w predicts the held-out log KLD best
    (leave-one-group-out, so w is not tuned on the group it is scored on);
 2. selection: per group with >= 2 held-out mixes, held-out KLD of argmin(m) vs argmin(post) (ties: same pick);
 3. simulation (--sim N): per target, the real candidate set (its c values), truth = c + N(0, tau), measurement =
    truth + N(0, sigma), tau / sigma from reading 1's variances; expected held-out KLD of each rule's pick vs the best
    candidate. Whole targets, so it covers what the held-out evals cannot (only finalists got them).
Targets <= MAXT (6.0) bpw: fit-final rounds KLDs to 5 decimals (> 2% rounding below 0.0005)."""
import collections, json, re, sys
import numpy as np

a = sys.argv[1:]
def opt(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return type(d)(v)
    return d
W0, NSIM, MAXT = opt("--w", 0.5), opt("--sim", 2000), opt("--maxt", 6.0)
RES, FIT = a[:2]

fit, off = {}, {"a": 1.0}
for l in open(FIT):
    m = re.search(r"slice offsets ((?:\w+ [\d.]+ ?)+)", l)
    if m and len(off) == 1:
        v = m.group(1).split(); off.update({v[i]: float(v[i + 1]) for i in range(0, len(v), 2)})
    p = l.split()
    if len(p) == 5 and re.match(r"^(a|f2|f3)-knap", p[0]): fit[p[0]] = float(p[4])
res, order = {}, {}
for i, l in enumerate(open(RES)):
    r = json.loads(l)
    if "kld" in r: res[r["tag"]] = r["kld"]; order[r["tag"]] = i
C = {}                                   # mix name -> (target, fit slice, m, c, kind, results index)
for t, cv in fit.items():
    s, n = t.split("-", 1); tg = float(re.search(r"-([\d.]+)$", n).group(1))
    kind = (re.search(r"-(ts|ls|tl|rd)\d", n) or [None, "mean"])[1]
    C[n] = (tg, s, res[t] / off[s], cv / off[s], kind, order[t])
print(f"{len(C)} fit-slice mix measurements, slice offsets {off}")

# ---- 1. calibration on held-out groups
G = collections.defaultdict(list)       # (held slice, target) -> [(name, held KLD)]
for t, k in res.items():
    m = re.match(r"^(w|v2)-(knap-.+)$", t)
    if m and m.group(2) in C and C[m.group(2)][0] <= MAXT: G[(m.group(1), C[m.group(2)][0])].append((m.group(2), k))
G = {g: v for g, v in G.items() if len(v) >= 2}
def cen(v): v = np.log(v); return v - v.mean()
grp = {g: (cen([C[n][2] for n, _ in v]), cen([C[n][3] for n, _ in v]), cen([k for _, k in v])) for g, v in G.items()}
WS = np.round(np.arange(0, 1.0001, 0.05), 2)
def sse(w, gs): return sum(((y - (w * xm + (1 - w) * xc)) ** 2).sum() for xm, xc, y in (grp[g] for g in gs))
n_rows = sum(len(v) for v in G.values())
print(f"\n1. calibration: {len(G)} (held-out slice, target) groups, {n_rows} mixes")
for w in (1.0, 0.75, 0.5, 0.25, 0.0):
    print(f"   w {w:4}: within-group RMS error vs held-out log KLD {np.sqrt(sse(w, G) / n_rows):.4f}")
logo = []
for g in G:
    rest = [h for h in G if h != g]; wb = WS[np.argmin([sse(w, rest) for w in WS])]; logo.append(wb)
print(f"   leave-one-group-out best w: median {np.median(logo):.2f} (range {min(logo):.2f}..{max(logo):.2f})")
xm = np.concatenate([grp[g][0] for g in G]); xc = np.concatenate([grp[g][1] for g in G]); y = np.concatenate([grp[g][2] for g in G])
# variances: e_m = log m - log truth, e_c = log c - log truth; held-out truth y carries its own noise sigma_h
vm, vc, vmc = ((xm - y) ** 2).mean(), ((xc - y) ** 2).mean(), ((xm - y) * (xc - y)).mean()
print(f"   mean sq. deviation from held-out: meas {vm:.5f}, model {vc:.5f}, cross {vmc:.5f} "
      f"(both include the held-out eval's own noise; the cross term estimates it, as m and c share no error otherwise)")
sig2, tau2 = max(vm - vmc, 1e-6), max(vc - vmc, 1e-6)
print(f"   -> sigma (measurement) {np.sqrt(sig2):.4f}, tau (model) {np.sqrt(tau2):.4f}, implied w {tau2 / (tau2 + sig2):.2f}")

# ---- 2. selection among the mixes that have held-out evals
print(f"\n2. selection on held-out groups (w = {W0}): held-out KLD of the pick, ratio post / measured-best")
rat = []
for g, v in sorted(G.items(), key=lambda x: (x[0][0], x[0][1])):
    names = [n for n, _ in v]; hk = dict(v)
    pm = min(names, key=lambda n: C[n][2])
    pp = min(names, key=lambda n: W0 * np.log(C[n][2]) + (1 - W0) * np.log(C[n][3]))
    rat.append(hk[pp] / hk[pm])
    if pm != pp: print(f"   {g[0]:2} {g[1]:5}: n {len(v)}  measured-best {pm} {hk[pm]:.5f}  posterior {pp} {hk[pp]:.5f}  ({hk[pp] / hk[pm] - 1:+.1%})")
rat = np.array(rat)
print(f"   {len(rat)} groups, picks differ in {int((rat != 1).sum())}; geomean ratio {np.exp(np.log(rat).mean()):.4f}, "
      f"posterior better in {int((rat < 1).sum())}, worse in {int((rat > 1).sum())}")
print("   caveat: held-out evals went mostly to each target's measured-best finalists, so these sets favour the measured-best rule")

# ---- 3. simulation on whole targets
if NSIM:
    rng = np.random.default_rng(0)
    print(f"\n3. simulation, {NSIM} draws a target, sigma {np.sqrt(sig2):.4f}, tau {np.sqrt(tau2):.4f}: expected regret "
          f"(pick's true KLD / best candidate's - 1)")
    T = collections.defaultdict(list)
    for n, v in C.items():
        if v[0] <= MAXT: T[v[0]].append(v[3])
    tot = collections.defaultdict(list)
    for tg in sorted(T):
        c = np.log(np.array(T[tg])); N = len(c)
        if N < 3: continue
        tr = c + rng.normal(0, np.sqrt(tau2), (NSIM, N)); me = tr + rng.normal(0, np.sqrt(sig2), (NSIM, N))
        best = tr.min(1); ar = np.arange(NSIM)
        out = {}
        for nm, w in (("measured-best", 1.0), (f"post w={W0}", W0), ("post w*", tau2 / (tau2 + sig2)), ("model-best", 0.0)):
            pick = (w * me + (1 - w) * c).argmin(1); out[nm] = np.exp(tr[ar, pick] - best).mean() - 1
            tot[nm].append(out[nm])
        print(f"   {tg:5}: {N:3d} candidates  " + "  ".join(f"{k} {v:+.2%}" for k, v in out.items()))
    print("   mean over targets: " + "  ".join(f"{k} {np.mean(v):+.2%}" for k, v in tot.items()))
    m0 = np.mean(tot["measured-best"])
    for k, v in tot.items(): print(f"   {k:14}: gain vs measured-best {(1 + np.mean(v)) / (1 + m0) - 1:+.2%} KLD")
