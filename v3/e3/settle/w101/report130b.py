#!/usr/bin/env python3
"""report130b.py STAB.jsonl [--json F] -- w101 #130-02 verdict (../w101.md): kept = G_out / G_in per family, P16 (and B16,
reported). Per tensor and direction (A, B) in {(E, O), (O, E)}: in-sample ratio c_P(A->A) / c_N(A->A), out-of-sample
c_P(A->B) / c_N(A->B); per family the geo mean over tensors x both directions; G = 1 - ratio. Pass: kept(qkv) >= 0.75.
Also: the damped-metric full-set ratio (perm2.py's pivsum g16 / nat) and the head-mean Spearman between halves."""
import json, math, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
fam = lambda k: "qkv" if "in_proj_qkv" in k else "z" if "in_proj_z" in k else "q_proj"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
gate = max(r["gate_rel"] for r in rows)
print(f"{len(rows)} tensors; gate G-b max rel {gate:.2e} ({'PASS' if gate <= 1e-5 else 'FAIL'})")
agg = defaultdict(lambda: defaultdict(list)); per = []
for r in rows:
    f = fam(r["key"]); t = {"key": r["key"].replace("model.language_model.layers.", "L"), "fam": f}
    for arm in ("P16", "B16"):
        for A, B in (("E", "O"), ("O", "E")):
            i = r[f"{arm}:{A}{A}"] / r[f"N:{A}{A}"]; o = r[f"{arm}:{A}{B}"] / r[f"N:{A}{B}"]
            agg[(f, arm)]["in"].append(i); agg[(f, arm)]["out"].append(o)
            t[f"{arm}_{A}{B}"] = [round(i, 4), round(o, 4)]
        full = r[f"{arm}:FF"] / r["N:FF"]; fd = (r[f"{arm}:FF"] + r[f"{arm}:Ffl"]) / (r["N:FF"] + r["N:Ffl"])
        agg[(f, arm)]["full"].append(full); agg[(f, arm)]["full_damped"].append(fd)
        t[f"{arm}_full"] = round(full, 4); t[f"{arm}_full_damped"] = round(fd, 4)
    t["rho_head"] = round(r["rho_head"], 3); agg[(f, "rho")]["rho"].append(r["rho_head"])
    per.append(t)
out = {"gate": gate, "families": {}, "per_tensor": per}
print(f"\n{'family':7s} {'arm':4s} {'n':>3s}  {'in-sample':>9s} {'held-out':>9s}  {'G_in':>6s} {'G_out':>6s} {'kept':>6s}   "
      f"{'full':>6s} {'full damped (perm2)':>19s}   rho_head (mean, min)")
for f in ("qkv", "z", "q_proj"):
    for arm in ("P16", "B16"):
        a = agg[(f, arm)]; ri, ro = geo(a["in"]), geo(a["out"]); kept = (1 - ro) / (1 - ri)
        rho = agg[(f, "rho")]["rho"]
        out["families"][f"{f}:{arm}"] = {"n": len(a["in"]) // 2, "ratio_in": ri, "ratio_out": ro, "kept": kept,
                                         "full": geo(a["full"]), "full_damped": geo(a["full_damped"]),
                                         "rho_mean": sum(rho) / len(rho), "rho_min": min(rho)}
        print(f"{f:7s} {arm:4s} {len(a['in']) // 2:3d}  {ri:9.4f} {ro:9.4f}  {1 - ri:6.3f} {1 - ro:6.3f} {kept:6.3f}   "
              f"{geo(a['full']):6.4f} {geo(a['full_damped']):19.4f}   {sum(rho) / len(rho):.3f}, {min(rho):.3f}")
k = out["families"]["qkv:P16"]["kept"]
ok = {f: out["families"][f"{f}:P16"]["kept"] >= 0.75 for f in ("qkv", "z", "q_proj")}
print(f"\nVERDICT (b): kept(qkv, P16) = {k:.3f} vs >= 0.75 -> {'PASS' if k >= 0.75 and gate <= 1e-5 else 'FAIL'}; "
      f"z {'ok' if ok['z'] else 'no'}, q_proj {'ok' if ok['q_proj'] else 'no'} (may join (c) with (a)'s scope rule)")
out["verdict"] = "PASS" if k >= 0.75 and gate <= 1e-5 else "FAIL"; out["family_ok"] = ok
worst = sorted(per, key=lambda t: t["P16_EO"][1] / t["P16_EO"][0] if t["P16_EO"][0] < 1 else 0, reverse=True)[:5]
print("largest in -> out loss (P16, E->O):", [(t["key"], t["P16_EO"]) for t in worst])
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
