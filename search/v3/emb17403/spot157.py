#!/usr/bin/env python3
"""spot157.py DOCSUMS.json REGISTRY REL.jsonl... [--out OUT.json] -- (w157, #174-03) RULES.md §1 / §5 readout on
golden-s173e: identity (T's rows == R's rows, sha256), and every arm vs B (the pack's bf16 embedding): per part (c / x)
R = sum KL(arm) / sum KL(B) over the part's documents (same positions), pooled = geometric mean of the part ratios, 95 %
CI and SE of log pooled from 2,000 part-stratified paired document bootstraps (seed 17403). Prediction (§5): log r_pred =
log(1 + (err_arm - F) / KL_B,c), err = F + a rel (embprices174 floor model), KL_B,c = B's calib-part mean KL; agrees when
|log R_pooled - log r_pred| <= 2 SE."""
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import slices as SL  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("docsums"); ap.add_argument("registry"); ap.add_argument("rel", nargs="+")
ap.add_argument("--out"); ap.add_argument("--prices", default=os.path.join(HERE, "..", "..", "..", "data", "emb174", "embprices174.json"))
a = ap.parse_args()
S = json.load(open(a.docsums)); fm = json.load(open(a.prices))["floor_model"]; F, A = fm["F"], fm["a"]
rel = {}
for p in a.rel:
    for l in open(p):
        r = json.loads(l); lv = r["level"].rstrip("/").split("/")
        if lv[-2] == "t-i8" and "/rerun/" not in r["level"]: rel[float(lv[-1][1:])] = r["rel"]
parts = np.array(SL.Registry(a.registry).parts_of("s173e")); P = sorted(set(parts.tolist()))
grp = {p: np.where(parts == p)[0] for p in P}
rng = np.random.default_rng(17403)
draws = [{p: g[rng.integers(0, len(g), len(g))] for p, g in grp.items()} for _ in range(2000)]
B = np.array([d[0] for d in S["B"]["docs"]]); n = np.array([d[1] for d in S["B"]["docs"]])
assert len(B) == len(parts), (len(B), len(parts))
klBc = B[grp["c"]].sum() / n[grp["c"]].sum()
out = {"identity": {"T_rows_sha256": S["T"]["rows_sha256"], "R_rows_sha256": S.get("R", {}).get("rows_sha256"),
                    "pass": "R" in S and S["T"]["rows_sha256"] == S["R"]["rows_sha256"]},
       "B": {"kld": S["B"]["kld"], "kl_c": klBc, "top1": S["B"]["top1"]}, "arms": {}}
for arm, v in S.items():
    if arm == "B": continue
    X = np.array([d[0] for d in v["docs"]]); assert [d[1] for d in v["docs"]] == n.tolist(), arm
    pr = {p: X[g].sum() / B[g].sum() for p, g in grp.items()}
    lp = float(np.mean([math.log(pr[p]) for p in P]))
    bl = np.array([np.mean([math.log(X[d[p]].sum() / B[d[p]].sum()) for p in P]) for d in draws])
    se = float(bl.std(ddof=1)); lo, hi = np.percentile(np.exp(bl), [2.5, 97.5])
    r = {"kld": v["kld"], "top1": v["top1"], "parts": {p: float(pr[p]) for p in P}, "pooled": math.exp(lp), "ci95": [float(lo), float(hi)],
         "se_log": se}
    k = {"T": 2.5}.get(arm, float(arm[1:]) if arm.startswith("t") else None)
    if k is not None and k in rel:
        e = F + A * rel[k]; pred = math.log1p((e - F) / klBc)
        r.update(K=k, rel=rel[k], err=e, logr_pred=pred, logr_meas=lp, z=(lp - pred) / se, agree=abs(lp - pred) <= 2 * se)
    out["arms"][arm] = r
ec = [x for x in ("t1.625", "t2.125", "t3.375") if x in out["arms"]]
out["summary"] = {"identity": out["identity"]["pass"], "eighths_agree": f'{sum(out["arms"][x]["agree"] for x in ec)}/{len(ec)}',
                  "flag_eighth_prices": sum(not out["arms"][x]["agree"] for x in ec) >= 2}
if a.out: json.dump(out, open(a.out, "w"), indent=1)
print(f'identity T rows == R rows: {out["identity"]["pass"]}   B: KL {S["B"]["kld"]:.5f} (c {klBc:.5f}) top-1 {S["B"]["top1"]}')
print(f'{"arm":>7} {"KL":>8} {"R_c":>7} {"R_x":>7} {"pooled":>7} {"95% CI":>16} {"log pred":>9} {"log meas":>9} {"z":>6} agree')
for arm, r in out["arms"].items():
    print(f'{arm:>7} {r["kld"]:8.5f} {r["parts"]["c"]:7.4f} {r["parts"]["x"]:7.4f} {r["pooled"]:7.4f} [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}] '
          + (f'{r["logr_pred"]:+9.4f} {r["logr_meas"]:+9.4f} {r["z"]:+6.2f} {r["agree"]}' if "agree" in r else ""))
print("summary:", out["summary"])
