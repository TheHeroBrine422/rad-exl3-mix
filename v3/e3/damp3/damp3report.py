#!/usr/bin/env python3
"""damp3report.py SCORE.jsonl [OUT.json] -- damping across the K range (damp3.sh), with the decision rule fixed before
the run (2026-11-02, written before any result).

Data: per (tensor, K, sigma) the proxy error of a quant made on H_A, scored on H_A (in-sample) and H_B (held out).
Population for the input-group sweep: q/k/v of layers 3 and 43, in_proj_qkv / z of layer 10 (8 tensors a K).
Bands: low K < 2, mid 2 <= K < 5, high K >= 5.

The ladder's rule picks, per tensor and level, the sigma of its grid with the lowest in-sample error. Here: pick on
H_A, score the pick on H_B, as a ratio to sigma 0.025 on H_B; a grid's value in a band = geometric mean of that ratio
over the band's (tensor, K). Paired bootstrap over tensors (2,000 draws) for grid differences.

RULE (per band):
  1. A 3-value grid replaces the current {0.025, 0.005, 0.001} only if its held-out geo-mean is lower by more than
     max(0.3%, 2 SE) and it is not worse than the current grid at any single K of the band by more than 0.3%.
     Among passing grids, the lowest held-out geo-mean wins. Grid size stays 3 (level time).
  2. Integer vs fractional (mid and high bands): if the best grid on the integer levels and the best on the fractional
     levels differ, and each beats the other's choice on its own levels by more than 2 SE, the levels get grids by
     type; otherwise one grid per band.
  3. Other tensors (o_proj / out_proj, gate / up; fixed sigma in the ladder): a fixed sigma other than 0.025 is
     flagged for the user only if it beats 0.025 on H_B by >= 1% and > 2 SE in a band (sweeping them costs level time).
Also reported: the share of picks at each grid edge (a capped grid shows up there) and the in-sample optimism
(selection gain on H_A minus the same picks' gain on H_B)."""
import collections, itertools, json, math, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
CUR = (0.025, 0.005, 0.001)
INPUT = ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")
data = collections.defaultdict(dict)          # (key, K) -> sigma -> (rel_A, rel_B)
for r in rows:
    K, d = r["arm"][1:].split("-d"); K, d = float(K), float(d)
    data[(r["key"], K)][d] = (r["rel_A"], r["rel_B"])
SIG = sorted({d for v in data.values() for d in v})
band = lambda K: "low" if K < 2 else "mid" if K < 5 else "high"
is_int = lambda K: float(K).is_integer()
kind = lambda key: "input" if key.split(".")[-1] in INPUT else "other"
cells = {kd: v for kd, v in data.items() if all(d in v for d in SIG)}
print(f"{len(rows)} rows, sigmas {SIG}, complete cells {len(cells)} of {len(data)}")


def pick_ratio(v, grid):
    """held-out ratio to 0.025 of the in-sample pick within grid, and the in-sample ratio"""
    d = min(grid, key=lambda s: v[s][0])
    return v[d][1] / v[0.025][1], v[d][0] / v[0.025][0], d


def grid_stats(cs, grid, rng):
    lr = np.array([math.log(pick_ratio(v, grid)[0]) for v in cs.values()])
    return lr


grids = [g for g in itertools.combinations(sorted(SIG, reverse=True), 3)]
rng = np.random.default_rng(0)
report = {"sigmas": SIG, "bands": {}}
for b in ("low", "mid", "high"):
    for sub in (("all", lambda K: True), ("integer", is_int), ("fractional", lambda K: not is_int(K))):
        cs = {kd: v for kd, v in cells.items() if band(kd[1]) == b and kind(kd[0]) == "input" and sub[1](kd[1])}
        if not cs: continue
        keys = sorted({k for k, _ in cs}); Ks = sorted({K for _, K in cs})
        cur = grid_stats(cs, CUR, rng)
        res = []
        for g in grids:
            lr = grid_stats(cs, g, rng)
            diff = lr - cur                                           # per (tensor, K), paired with the current grid
            kk = [k for k, _ in cs]
            boot = [np.mean([diff[i] for i, k in enumerate(kk) if k in set(rng.choice(keys, len(keys)))] or [0])
                    for _ in range(400)]
            se = float(np.std(boot))
            perK = {K: float(np.mean([diff[i] for i, (k, KK) in enumerate(cs) if KK == K])) for K in Ks}
            res.append({"grid": g, "geo": math.exp(lr.mean()) - 1, "vs_cur": math.exp(diff.mean()) - 1, "se": se,
                        "worst_K": max(perK.values()), "pass": (diff.mean() < -max(0.003, 2 * se)) and max(perK.values()) < 0.003})
        res.sort(key=lambda x: x["geo"])
        cur_r = next(x for x in res if x["grid"] == CUR)
        oracle = math.exp(np.mean([math.log(min(v[d][1] for d in SIG) / v[0.025][1]) for v in cs.values()])) - 1
        edge = collections.Counter(pick_ratio(v, CUR)[2] for v in cs.values())
        alls = collections.Counter(pick_ratio(v, tuple(SIG))[2] for v in cs.values())
        ins = math.exp(np.mean([math.log(pick_ratio(v, CUR)[1]) for v in cs.values()])) - 1
        name = f"{b}/{sub[0]}"
        report["bands"][name] = {"n": len(cs), "K": Ks, "current": cur_r, "best": res[:5], "oracle_B": oracle,
                                 "current_picks": dict(edge), "picks_full_grid": dict(alls), "current_in_sample": ins,
                                 "adopt": next((x for x in res if x["pass"]), None)}
        print(f"\n== {name}: {len(cs)} (tensor, K), K {Ks}")
        print(f"   current {CUR}: held-out {cur_r['geo']:+.2%} vs fixed 0.025 (in-sample {ins:+.2%}); oracle {oracle:+.2%}")
        print(f"   current grid picks: {dict(sorted(edge.items()))}; full-grid picks: {dict(sorted(alls.items()))}")
        for x in res[:5]:
            print(f"   {str(x['grid']):28s} held-out {x['geo']:+.2%}  vs current {x['vs_cur']:+.2%} (SE {x['se']:.2%}, worst K "
                  f"{x['worst_K']:+.2%}){'  PASS' if x['pass'] else ''}")
        a = report["bands"][name]["adopt"]
        print(f"   RULE 1 -> {'ADOPT ' + str(a['grid']) if a else 'keep current'}")
# rule 3: other tensors, fixed sigma
for b in ("low", "mid", "high"):
    cs = {kd: v for kd, v in cells.items() if band(kd[1]) == b and kind(kd[0]) == "other"}
    if not cs: continue
    keys = sorted({k for k, _ in cs}); out = []
    for d in SIG:
        lr = np.array([math.log(v[d][1] / v[0.025][1]) for v in cs.values()]); kk = [k for k, _ in cs]
        boot = [np.mean([lr[i] for i, k in enumerate(kk) if k in set(rng.choice(keys, len(keys)))] or [0]) for _ in range(400)]
        out.append((d, math.exp(lr.mean()) - 1, float(np.std(boot))))
    flag = [x for x in out if x[1] <= -0.01 and -x[1] > 2 * x[2]]
    report["bands"][f"{b}/other"] = {"fixed": out, "flag": flag}
    print(f"\n== {b}/other (o/out_proj, gate/up; fixed 0.025 in the ladder), {len(cs)} (tensor, K): " +
          ", ".join(f"{d:g} {m:+.2%}" for d, m, _ in out) + f"  -> RULE 3 {'FLAG ' + str(flag) if flag else 'keep 0.025'}")
# rule 2
for b in ("mid", "high"):
    i, f = report["bands"].get(f"{b}/integer"), report["bands"].get(f"{b}/fractional")
    if i and f:
        same = i["best"][0]["grid"] == f["best"][0]["grid"]
        print(f"\nRULE 2 {b}: best integer {i['best'][0]['grid']}, best fractional {f['best'][0]['grid']} -> "
              f"{'same grid' if same else 'differ: check SEs above before splitting by type'}")
if OUT: json.dump(report, open(OUT, "w"), indent=1, default=str)
