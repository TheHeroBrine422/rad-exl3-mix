#!/usr/bin/env python3
"""report130a.py OUT [--json F] -- w101 #130-01 verdict (../w101.md): r = f2_P / f2_N per tensor, K, seed; R_t = geo
mean over K2.5 / K4 x seeds 0 / 1. Pass: R(L10 qkv) <= 0.90. Scope of (c): q_proj if geo(R L11, R L39) <= 0.95, z if
R(L10 z) <= 0.95. Also the composable model's ratio (OUT/model.json) and the realized share ln r / ln r_model."""
import glob, json, math, os, sys

OUT = sys.argv[1]
f2 = {}
for fp in glob.glob(f"{OUT}/*/f2.json"):
    arm, K, s = os.path.basename(os.path.dirname(fp)).split("-")
    for k, o in json.load(open(fp)).items(): f2[(arm, K[1:], s[1:], k)] = o
model = json.load(open(f"{OUT}/model.json")) if os.path.exists(f"{OUT}/model.json") else {}
short = lambda k: k.replace("model.language_model.layers.", "L").replace(".linear_attn.", " ").replace(".self_attn.", " ")
keys = sorted({k for (_, _, _, k) in f2}, key=lambda k: (int(k.split(".")[3]), k))
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
out = {"r": {}, "R": {}, "model": {}}
print("tensor                     K    seed  f2_N        f2_P        r=P/N    f1 r")
for k in keys:
    rs = []
    for K in ("2.5", "4"):
        for s in ("0", "1"):
            n, p = f2.get(("N", K, s, k)), f2.get(("P", K, s, k))
            if not (n and p): continue
            r = p["f2"] / n["f2"]; rs.append(r); out["r"][f"{short(k)} K{K} s{s}"] = r
            print(f"{short(k):26s} {K:4s} {s:4s}  {n['f2']:.4e}  {p['f2']:.4e}  {r:.4f}   {p['f1'] / n['f1']:.4f}")
    out["R"][short(k)] = geo(rs) if len(rs) == 4 else None
    m = model.get(k, {}).get("ratio"); out["model"][short(k)] = m
    if rs:
        R = geo(rs)
        print(f"  -> R {R:.4f} over {len(rs)} (of 4); model g16/nat {m if m is None else round(m, 4)}; realized share "
              f"{(math.log(R) / math.log(m)) if m and m < 1 else float('nan'):.2f}")
R = out["R"]; q = [R.get("L11 q_proj"), R.get("L39 q_proj")]
qkv = R.get("L10 in_proj_qkv")
if qkv is None: print("\nVERDICT: incomplete"); out["verdict"] = "incomplete"
else:
    ok = qkv <= 0.90
    scope = ["in_proj_qkv"] + (["q_proj"] if None not in q and geo(q) <= 0.95 else []) + (["in_proj_z"] if R.get("L10 in_proj_z") and R["L10 in_proj_z"] <= 0.95 else [])
    print(f"\nVERDICT (a): R(L10 qkv) = {qkv:.4f} vs <= 0.90 -> {'PASS' if ok else 'FAIL'}; q_proj geo "
          f"{geo([x for x in q if x]):.4f}, z {R.get('L10 in_proj_z')}; scope of (c) if (b) passes: {scope if ok else 'none'}")
    out.update(verdict="PASS" if ok else "FAIL", scope=scope if ok else [])
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
