#!/usr/bin/env python3
"""knap.py DIR TARGETS -- the sensitivity-calibrated mix frontier.

DIR holds hproxy*.jsonl (per weight x container: bytes, err = tr(dW H dW^T)/rows on the calib Hessians),
radls.jsonl (every container's entry bytes) and evq-results.jsonl (calibration evals on calib half A).

Model: predicted KLD(mix) = KLD(base) + sum over weights s_w * (err_w(choice) - err_w(base)) + head(choice) + emb(choice)
  s_w   from the group swaps cal-BASE-DONOR-GROUP-LBB: dKLD / sum dErr over the swapped weights, one value per (group,
        band), interpolated linearly in the layer index between band centres (flat past the ends);
  head  measured head swaps where there are any (head-BASE-X), else s_head * dErr with s_head fitted on those;
  emb   measured embedding swaps (emb-BASE-X), the base's own table at 0.
Multiple-choice knapsack by Lagrangian sweep (exact on the convex hull of the frontier), then for each target real
bpw the hull point at or under it. Writes DIR/mixes/<name>.mix (lines "weight donor.rad", base = BASE) and
DIR/frontier.jsonl.

TARGETS: comma list of real bpw, e.g. 2.0,2.5,2.99,3.5,3.99,4.5,5.0,5.49,6.0"""
import json, os, re, sys, glob, collections, bisect
import numpy as np
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")

D, targets = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
PARAMS = 26895998464
F = RXM_MODELS
BASE = os.environ.get("KNAP_BASE", "sc4.00-e4")
EXCLUDE = set(filter(None, os.environ.get("KNAP_EXCLUDE", "").split(",")))   # containers not to use as donors
rad = lambda t: f"{F}/q38-27b-{t}.rad" if not t.startswith("ours-") else f"{RXM_OURS}/{t}.rad"

# ------------------------------------------------------------------ options
opt = collections.defaultdict(dict)            # weight -> container -> (bytes, err)
for p in glob.glob(f"{D}/hproxy*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r["c"] in EXCLUDE: continue
        opt[r["w"]][r["c"]] = (r["bytes"], r["err"])
ls = {}
for l in open(f"{D}/radls.jsonl"):
    r = json.loads(l); t = re.search(r"q38-27b-(.*)\.rad", r["c"]).group(1); ls[t] = r
base_ls = ls[BASE]
exl3 = set(opt)
fixed = base_ls["total"] - sum(base_ls["w"][w] for w in exl3) - base_ls["w"]["token_embd.weight"]

# ------------------------------------------------------------------ calibration
res = {}
for l in open(f"{D}/evq-results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r["kld"]
k0 = res[f"base-{BASE}"]
GROUPS = {"attn_in": (lambda l: l % 4 == 3, ["attn_qg", "attn_k", "attn_v"]),
          "attn_out": (lambda l: l % 4 == 3, ["attn_output"]),
          "ssm_in": (lambda l: l % 4 != 3, ["ssm_qkv", "ssm_z"]),
          "ssm_out": (lambda l: l % 4 != 3, ["ssm_out"]),
          "ffn_gu": (lambda l: True, ["ffn_gate_up"]),
          "ffn_down": (lambda l: True, ["ffn_down"])}
TYPE2G = {t: g for g, (_, ts) in GROUPS.items() for t in ts}
# Linear model, normalised at the base: KLD(mix) = k0 * X(mix) / X(base), X = sum_w s_w err_w. The swaps' absolute
# slopes overstate the base's own error ~4x (big K=2 steps), but the argmin under a byte budget only needs the
# relative s, and the normalisation keeps the predicted KLDs sensible. P stays 1.
swaps = []                                     # (group, band start, band width, k, dErr)
for tag, k in res.items():
    m = re.match(rf"cal-{re.escape(BASE)}-(.+)-({'|'.join(GROUPS)})-L(\d+)(?:w(\d+))?$", tag)
    if not m: continue
    donor, g, b0, bw = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or 8)
    if os.environ.get("KNAP_DONOR") and donor != os.environ["KNAP_DONOR"]: continue
    sel, ts = GROUPS[g]
    ws = [f"blk.{l}.{t}.weight" for l in range(b0, b0 + bw) if sel(l) for t in ts]
    de = sum(opt[w][donor][1] - opt[w][BASE][1] for w in ws)
    if de > 0: swaps.append((g, b0 + (bw - 1) / 2, k, de))
P = 1.0
def points(P):
    pts = collections.defaultdict(list)
    for g, x, k, de in swaps: pts[g].append((x, (k - k0) / de))
    for g in pts: pts[g].sort()
    return pts
pts = None
def s_of(w, pts=None):
    pts = pts if pts is not None else PTS
    l, t = re.match(r"blk\.(\d+)\.(\w+)\.weight", w).groups()
    p = pts.get(TYPE2G[t])
    if not p: return None
    xs, ys = zip(*p)
    return float(np.interp(int(l), xs, ys))
def xbase(P):
    q = points(P); med = float(np.median([s for p in q.values() for _, s in p]))
    for g in GROUPS:
        if g not in q: q[g] = [(31.5, med)]
    return sum(s_of(w, q) * opt[w][BASE][1] for w in exl3 if w != "output.weight")
PTS = points(P); pts = PTS
X0 = xbase(1.0)
toX = lambda k, P=1.0: X0 * k / k0           # a measured KLD -> X units
print(f"X(base) = {X0:.5f}, KLD(base) = {k0:.5f}: slopes scaled by {k0 / X0:.3f}", file=sys.stderr)
missing = sorted({TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight", w).group(1)] for w in exl3 if w != "output.weight" and s_of(w) is None})
if missing:
    # groups not calibrated yet: the median s of the calibrated ones, flagged
    med = float(np.median([s for p in pts.values() for _, s in p]))
    for g in missing: pts[g] = [(31.5, med)]
    print(f"WARNING: no calibration yet for {missing}: using the median sensitivity {med:.3g}", file=sys.stderr)

# head: measured swaps + a fitted s_head for the rest
H = "output.weight"
hmeas = {m.group(1): toX(k, P) - k0 for tag, k in res.items() if (m := re.match(rf"head-{re.escape(BASE)}-(.+)$", tag))}
sh = [(hmeas[c]) / (opt[H][c][1] - opt[H][BASE][1]) for c in hmeas if c in opt[H] and opt[H][c][1] != opt[H][BASE][1]]
s_head = float(np.median(sh)) if sh else None
def head_cost(c):
    if c == BASE: return 0.0
    if c in hmeas: return hmeas[c]
    return s_head * (opt[H][c][1] - opt[H][BASE][1]) if s_head is not None else None

# embedding: measured
emb = {BASE: (base_ls["w"]["token_embd.weight"], 0.0)}
for tag, k in res.items():
    if (m := re.match(rf"emb-{re.escape(BASE)}-(.+)$", tag)):
        emb[m.group(1)] = (ls[m.group(1)]["w"]["token_embd.weight"], toX(k, P) - k0)

# ------------------------------------------------------------------ items: (name, [(choice, bytes, dkld)])
items = []
for w in sorted(exl3):
    ch = []
    for c, (b, e) in opt[w].items():
        if w == H:
            d = head_cost(c)
            if d is None: continue
        else:
            d = s_of(w) * (e - opt[w][BASE][1])
        ch.append((c, b, d))
    items.append((w, ch))
items.append(("token_embd.weight", [(c, b, d) for c, (b, d) in emb.items()]))
# bf16 (unquantized, served by the plugin's gemm_nt fallback): zero quantization error, the bf16 container's bytes
if "bf16" in ls and not os.environ.get("KNAP_NO_BF16"):
    bw = ls["bf16"]["w"]
    for name, ch in items:
        if name in opt and name in bw:
            if name == H:
                d = head_cost("bf16") if "bf16" in hmeas else (s_head * (0 - opt[H][BASE][1]) if s_head is not None else None)
            else:
                d = s_of(name) * (0 - opt[name][BASE][1])
            if d is not None: ch.append(("bf16", bw[name], d))

def solve(lam):
    pick, B, X = {}, fixed, X0
    for name, ch in items:
        c, b, d = min(ch, key=lambda x: x[2] + lam * x[1])
        pick[name] = c; B += b; X += d
    return pick, B, k0 * max(X, 0) / X0
# sweep lambda (KLD per byte) on a log grid, keep the hull points
lams = np.logspace(-18, -4, 600)
hull = []
for lam in lams:
    pick, B, K = solve(lam)
    if not hull or B != hull[-1][1]: hull.append((lam, B, K, pick))
hull.sort(key=lambda h: h[1])
os.makedirs(f"{D}/mixes", exist_ok=True)
out = open(f"{D}/frontier.jsonl", "w")
for t in targets:
    cap = t * PARAMS / 8
    cand = [h for h in hull if h[1] <= cap]
    if not cand: print(f"{t}: nothing fits", file=sys.stderr); continue
    lam, B, K, pick = cand[-1]
    name = f"knap-{BASE}-{t:.2f}"
    with open(f"{D}/mixes/{name}.mix", "w") as f:
        for w, c in sorted(pick.items()):
            if c != BASE: f.write(f"{w} {rad(c)}\n")
    use = collections.Counter(pick.values())
    rec = {"name": name, "target": t, "bpw": B * 8 / PARAMS, "pred_kld": K, "lambda": lam,
           "head": pick[H], "emb": pick["token_embd.weight"], "donors": dict(use.most_common())}
    out.write(json.dumps(rec) + "\n")
    print(f"{name:26} bpw {B*8/PARAMS:.3f}  pred KLD {K:.5f}  head {pick[H]:13} emb {pick['token_embd.weight']:12} "
          + " ".join(f"{c}:{n}" for c, n in use.most_common(6)))
print("calibration points:", {g: [(round(x), f"{s:.3g}") for x, s in p] for g, p in pts.items()}, file=sys.stderr)
print("s_head", s_head, "head measured", {c: round(v, 5) for c, v in hmeas.items()}, file=sys.stderr)
if os.environ.get("KNAP_CHECK"):
    tot = collections.Counter()
    for w in exl3:
        if w == H: continue
        tot[TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight", w).group(1)]] += s_of(w) * opt[w][BASE][1]
    print("sum s*err(base) by group:", {g: round(v, 5) for g, v in tot.items()}, "total", round(sum(tot.values()), 5), "base KLD", k0, file=sys.stderr)
