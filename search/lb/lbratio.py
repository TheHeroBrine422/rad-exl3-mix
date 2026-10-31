#!/usr/bin/env python3
"""lbratio.py RESULTS.jsonl PREFIX [LB.json] -- finalist picker against the leaderboard's category rule (2026-10-30).

The rule (leaderboard help text): an entry's score is its error / the unsloth UD frontier's at the same size, for
KLD and for the top-1 miss rate (1 - top1) separately. The UD frontier = the UD quants no smaller UD quant beats (per
metric), joined by straight lines in log(error) vs bpw (our reading of "a straight line on a log scale"), extended
along the nearest piece beyond the ends. Only entries on the Pareto front can lead (beaten by one no larger, sizes
within 0.25% equal: out); the lowest score leads, the smaller entry on a tie.

Our numbers come from a calib slice (PREFIX: v2- = ver2, w- = ver1, a- / f2- / f3- = fit slices), not the private
set, so each is mapped to the leaderboard scale with the ratio (leaderboard / slice) of the SAME turboderp containers
(PREFIX + ref-*, our EXL3-V1 entries on the board), separately for KLD and miss rate, interpolated linearly in bpw
(clamped at the ends). Then: per category, our candidates (PREFIX + knap-*) ranked by estimated KLD score, our best
next to the board's current leader. Treat the estimates as +-5-10% (the mapping is an approximation).
"""
import json, math, os, re, sys

RES, PFX = sys.argv[1], sys.argv[2]
LBF = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results-2026-10-30.json")
lb = json.load(open(LBF)); P = lb["params"]
ent = [dict(name=e["name"], author=e["author"], group=e["group"], bpw=e["bytes"] * 8 / P, kld=e["scores"]["kld_mean"],
            miss=1 - e["scores"]["top1"]) for e in lb["entries"] if e.get("scores", {}).get("kld_mean") is not None
       and e["scores"].get("top1") is not None]


def front(pts):                      # (bpw, err) -> the points no smaller one beats
    out = []
    for b, v in sorted(pts):
        if not out or v < out[-1][1]: out.append((b, v))
    return out


def loginterp(pts, b):               # log-linear between points, extended along the nearest piece
    if len(pts) == 1: return pts[0][1]
    i = 0 if b <= pts[0][0] else len(pts) - 2 if b >= pts[-1][0] else max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))


def lininterp(pts, b):
    pts = sorted(pts)
    if b <= pts[0][0]: return pts[0][1]
    if b >= pts[-1][0]: return pts[-1][1]
    i = max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return y0 + (b - x0) / (x1 - x0) * (y1 - y0)


ud = [e for e in ent if e["group"] == "ud"]
UDK = front([(e["bpw"], e["kld"]) for e in ud]); UDM = front([(e["bpw"], e["miss"]) for e in ud])
score = lambda b, k, m: (k / loginterp(UDK, b), m / loginterp(UDM, b))

res = {}
for l in open(RES):
    r = json.loads(l)
    if "kld" in r and r.get("bytes") and r["tag"].startswith(PFX): res[r["tag"]] = r
bpw = lambda r: r["bytes"] * 8 / P


def lbname(c):
    m = re.fullmatch(r"(sc|plain)(\d\.\d\d)-e\d", c)
    return None if not m else f"EXL3-{'SC-' if m.group(1) == 'sc' else ''}{m.group(2)}bpw"


byname = {e["name"]: e for e in ent}
cal_k, cal_m = [], []
for t, r in res.items():
    if not t.startswith(PFX + "ref-"): continue
    n = lbname(t[len(PFX) + 4:])
    if n in byname:
        e = byname[n]; cal_k.append((bpw(r), e["kld"] / r["kld"])); cal_m.append((bpw(r), e["miss"] / (1 - r["top1"])))
if len(cal_k) < 2: sys.exit(f"need >= 2 reference containers measured on {PFX} (have {len(cal_k)})")

ours = []
for t, r in res.items():
    if not t.startswith(PFX + "knap-"): continue
    b = bpw(r); k = r["kld"] * lininterp(cal_k, b); m = (1 - r["top1"]) * lininterp(cal_m, b)
    sk, sm = score(b, k, m)
    ours.append(dict(name=t, bpw=b, kld=k, miss=m, sk=sk, sm=sm))

print(f"{PFX}: {len(ours)} mixes, calibration from {len(cal_k)} containers: KLD x" +
      " ".join(f"{b:.2f}:{v:.2f}" for b, v in sorted(cal_k)) + ";  miss x" + " ".join(f"{b:.2f}:{v:.2f}" for b, v in sorted(cal_m)))
for cat in lb["categories"]:
    lo, hi = cat["min"], cat["max"]
    pool = [dict(e, sk=score(e["bpw"], e["kld"], e["miss"])[0], sm=score(e["bpw"], e["kld"], e["miss"])[1], ours=False)
            for e in ent if lo <= e["bpw"] < hi] + [dict(o, ours=True) for o in ours if lo <= o["bpw"] < hi]
    def lead(metric, err, pl=None):
        pl = pool if pl is None else pl
        cand = [x for x in pl if not any(y is not x and y["bpw"] <= x["bpw"] * 1.0025 and y[err] < x[err] for y in pl)]
        return sorted(cand, key=lambda x: (x[metric], x["bpw"]))
    print(f"\n{cat['name']} [{lo}, {hi}) bpw")
    for metric, err, lab in (("sk", "kld", "KLD"), ("sm", "miss", "top-1 miss")):
        L = lead(metric, err)
        if not L: print(f"  {lab}: no entries"); continue
        top = L[0]; ob = next((x for x in L if x["ours"]), None)
        oth = next(iter(lead(metric, err, [x for x in pool if not x["ours"]])), None)   # today's leader, without us
        print(f"  {lab:10} leader: {'OURS ' if top['ours'] else ''}{top['name']} {top['bpw']:.3f} bpw {top[metric]:.2f}x   "
              f"| leader today {oth['name'] if oth else '-'} {oth[metric] if oth else float('nan'):.2f}x   "
              f"| our best on the front {ob['name'] if ob else '- (none on the Pareto front)'}"
              + (f" {ob['bpw']:.3f} bpw {ob[metric]:.2f}x" if ob else ""))
