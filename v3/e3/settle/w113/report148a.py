#!/usr/bin/env python3
"""report148a.py OUT -- w113 #148-01 table and gate (../w113.md): per tensor r_X = f2_X / f2_N (X = P, T, V) per K and
seed; per family R_X = geo over its tensors x K x seeds; keep = ln R_T / ln R_P (and ln R_V / ln R_P for v_proj);
gate per family: keep >= 0.75 and R_P < 1. Last line = the (b) scope."""
import json, math, os, sys
from collections import defaultdict

OUT = sys.argv[1]
FAM = {"in_proj_qkv": "ssm_qkv", "in_proj_z": "ssm_z", "q_proj": "attn_qg", "v_proj": "attn_v"}
f2 = {}
for d in sorted(os.listdir(OUT)):
    fp = f"{OUT}/{d}/f2.json"
    if not os.path.exists(fp): continue
    arm, K, s = d.split("-"); K = K[1:]; s = s[1:]
    for k, o in json.load(open(fp)).items(): f2[(arm, k, K, s)] = o["f2"]
model = json.load(open(f"{OUT}/model.json")) if os.path.exists(f"{OUT}/model.json") else {}
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
keys = sorted({k for (_, k, _, _) in f2}, key=lambda k: (FAM[k.split(".")[-1]], int(k.split(".")[3])))
logs = defaultdict(lambda: defaultdict(list))
print(f"{'tensor':22s} {'K':>4s} {'s':>2s} {'f2_N':>11s} {'P/N':>7s} {'T/N':>7s} {'V/N':>7s}")
for k in keys:
    fam = FAM[k.split(".")[-1]]; tag = f"L{k.split('.')[3]} {k.split('.')[-1]}"; per = defaultdict(list)
    for K in ("2.5", "4"):
        for s in ("0", "1"):
            n = f2.get(("N", k, K, s))
            if n is None: continue
            r = {a: f2[(a, k, K, s)] / n for a in "PTV" if (a, k, K, s) in f2}
            for a, v in r.items(): per[a].append(v); logs[fam][a].append(math.log(v))
            print(f"{tag:22s} {K:>4s} {s:>2s} {n:11.4e} " + " ".join(f"{r[a]:7.4f}" if a in r else f"{'':7s}" for a in "PTV"))
    m = model.get(k, {})
    print(f"  -> R " + ", ".join(f"{a} {geo(per[a]):.4f} (model {m.get('ratio_' + a, float('nan')):.4f})" for a in "PTV" if per[a])
          + (f"; tensor keep T {math.log(geo(per['T'])) / math.log(geo(per['P'])):.3f}" if per["P"] and per["T"] else ""))
print()
scope = []
for fam in ("ssm_qkv", "ssm_z", "attn_qg", "attn_v"):
    L = logs[fam]
    if not L["P"] or not L["T"]: print(f"{fam}: incomplete"); continue
    RP, RT = math.exp(sum(L["P"]) / len(L["P"])), math.exp(sum(L["T"]) / len(L["T"]))
    keep = math.log(RT) / math.log(RP) if RP < 1 else float("nan"); ok = RP < 1 and keep >= 0.75
    extra = ""
    if L["V"]:
        RV = math.exp(sum(L["V"]) / len(L["V"])); extra = f"; V (nest 2, report) R {RV:.4f}, keep {math.log(RV) / math.log(RP):.3f}"
    print(f"{fam:8s} n {len(L['P']):2d}: R_P {RP:.4f}, R_T {RT:.4f}, R_T/R_P {RT / RP:.4f}, keep {keep:.3f} -> "
          f"{'PASS' if ok else 'FAIL'}{extra}")
    if ok: scope.append(fam)
print(f"W113 #148-01 GATE: (b) scope = {','.join(scope) if scope else 'none (b does not run)'}")
