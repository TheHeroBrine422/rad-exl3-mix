#!/usr/bin/env python3
"""report98.py A FS0 -- #138-01's read (../w98.md): checks c1-c3, the f2 table of the refit arms, the verdict; #139's
per-level diag bytes and a trajectory / overload summary. A = /data/w98 (out/K<K>/L<l>-g<g>/ job dirs, arms/K<K>/<arm>/
f2.json from f2calc98.py), FS0 = w92's fs0 job dirs (the c1 no-op reference at K2.5). Last line: the verdict."""
import glob, json, math, os, sys

import torch
from safetensors import safe_open

A, FS0 = sys.argv[1:3]
KS = ["0.5", "2.5", "4"]
ARMS = ["none", "hout", "undamped", "hout+undamped"]
TARGET = ("layers.10.linear_attn.in_proj_qkv", "layers.10.linear_attn.in_proj_z", "layers.39.self_attn.q_proj")
short = lambda k: k.replace("model.language_model.layers.", "L").replace("linear_attn.", "").replace("self_attn.", "").replace("mlp.", "")
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))

rec, f2c, dbytes, jdir = {}, {}, {}, {}
for K in KS:
    for d in sorted(glob.glob(f"{A}/out/K{K}/L*-g*")):
        for l in open(f"{d}/log.jsonl"):
            r = json.loads(l); rec[(K, r["key"])] = r; jdir[(K, r["key"])] = d
            p = f"{d}/diag-{r['key']}.safetensors"
            if os.path.exists(p): dbytes[(K, r["key"])] = os.path.getsize(p)
    for arm in ["cur"] + ARMS:
        p = f"{A}/arms/K{K}/{arm}/f2.json"
        if os.path.exists(p):
            for k, v in json.load(open(p)).items(): f2c[(K, arm, k)] = v
keys = sorted({k for _, k in rec}, key=lambda k: (int(k.split(".")[3]), k))
print(f"# w98 #138-01 report: {len(keys)} tensors x {len(KS)} K, {len(rec)} job records, {len(f2c)} f2calc98 rows")

# ---- checks
c1n = c1t = 0; c1bad = []
for k in keys:
    l = int(k.split(".")[3])
    ref = glob.glob(f"{FS0}/L{l:02d}-g*/t-{k}.safetensors")
    if not ref or ("2.5", k) not in jdir: continue
    with safe_open(ref[0], "pt") as x, safe_open(f"{jdir[('2.5', k)]}/t-{k}.safetensors", "pt") as y:
        same = sorted(x.keys()) == sorted(y.keys()) and all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())
        ms = {m: (x.metadata().get(m), y.metadata().get(m)) for m in ("rxq_settings", "K", "codebook", "level")}
    c1t += 1; c1n += same
    if not same: c1bad.append(short(k))
c1 = c1t > 0 and c1n == c1t
print(f"c1 no-op vs w92 fs0 (K2.5 t-file tensors): {c1n}/{c1t} identical {c1bad[:4]}; metadata (settings, K, codebook, level) {ms if c1t else '-'}")
c2w, c2bad = 0.0, []
for K in KS:
    for k in keys:
        a = rec[(K, k)]["diag"]["arms"]["cur"]["f2"]; b = f2c.get((K, "cur", k), {}).get("f2")
        if b is None: c2bad.append(f"K{K} {short(k)} missing"); continue
        c2w = max(c2w, abs(a / b - 1))
c2 = c2w <= 2e-3 and not c2bad
print(f"c2 in-job f2 (cur) vs f2calc98: max rel {c2w:.2e} (<= 2e-3) {c2bad[:3]}")
c3w = 0.0
for (K, k), r in rec.items():
    with safe_open(f"{jdir[(K, k)]}/diag-{k}.safetensors", "pt") as f:
        e1, r1 = f.get_tensor(f"{k}.e1").double(), f.get_tensor(f"{k}.r1").double()
    c3w = max(c3w, abs(float(e1.sum() / r1.sum()) / r["proxy_err"] - 1), r["diag"]["check"]["rel"])
c3 = c3w <= 1e-6
print(f"c3 sum(e1) / sum(r1) vs logged proxy_err: max rel {c3w:.2e} (<= 1e-6)")

# ---- table: rho = f2(arm) / f2(cur), f2calc98 (fp64)
rho = {}
print("\n## f2 (w81 metric, fp64) of the stored arm and rho = f2(arm) / f2(cur); f1 ratio in brackets")
print(f"{'tensor':26s} {'K':>4s} {'f2 cur':>11s} " + " ".join(f"{a:>20s}" for a in ARMS))
for k in keys:
    for K in KS:
        cur = f2c[(K, "cur", k)]
        row = f"{short(k):26s} {K:>4s} {cur['f2']:11.4e} "
        for a in ARMS:
            v = f2c[(K, a, k)]; rho[(a, K, k)] = v["f2"] / cur["f2"]
            row += f" {rho[(a, K, k)]:9.5f} [{v['f1'] / cur['f1']:7.4f}]"
        print(row)

# ---- verdict per rule
print("\n## rule: (a) geo over T (qkv, z, qg x 3 K) <= 0.990; (b) every tensor's geo over K <= 1.001, no cell > 1.005")
tcells = [(K, k) for K in KS for k in keys if any(k.endswith(t) for t in TARGET)]
assert len(tcells) == 9, tcells
res = {}
for a in ARMS:
    gt = geo([rho[(a, K, k)] for K, k in tcells])
    fam = {short(k): geo([rho[(a, K, k)] for K in KS]) for k in keys}
    worst_fam = max(fam.items(), key=lambda x: x[1]); worst_cell = max(rho[(a, K, k)] for K in KS for k in keys)
    ok_a, ok_b = gt <= 0.990, worst_fam[1] <= 1.001 and worst_cell <= 1.005
    res[a] = (gt, ok_a and ok_b)
    print(f"{a:14s} T-geo {gt:.5f} ({'ok' if ok_a else 'no'})  worst tensor geo {worst_fam[0]} {worst_fam[1]:.5f}, worst cell {worst_cell:.5f} "
          f"({'ok' if ok_b else 'no'}) -> {'PASS' if ok_a and ok_b else 'fail'}   by K: " +
          " ".join(f"K{K} {geo([rho[(a, K, k)] for K2, k in tcells if K2 == K]):.4f}" for K in KS))
passing = [a for a in ("none", "hout", "undamped") if res[a][1]]
cand = min(passing, key=lambda a: res[a][0]) if passing else None
if "hout" in passing and "undamped" in passing and res["hout+undamped"][1] and res["hout+undamped"][0] < min(res["hout"][0], res["undamped"][0]):
    cand = "hout+undamped"
best = min(("none", "hout", "undamped"), key=lambda a: res[a][0])

# ---- #139: bytes, trajectory, overload
print("\n## #139 diag files: bytes per layer and per level (48 GDN layers like L10 + 16 attention layers like L39)")
for K in KS:
    b10 = sum(v for (K2, k), v in dbytes.items() if K2 == K and ".layers.10." in k)
    b39 = sum(v for (K2, k), v in dbytes.items() if K2 == K and ".layers.39." in k)
    print(f"K{K}: L10 {b10 / 1e6:.3f} MB, L39 {b39 / 1e6:.3f} MB -> level {(48 * b10 + 16 * b39) / 1e6:.1f} MB")
print("\n## trajectory f2 (pre-refit -> post-refit -> post-C9) and overload (|q| max, share >= 0.9 max, targets beyond, tiles > 3x median)")
for K in KS:
    for k in keys:
        d = rec[(K, k)]["diag"]; tj = d["traj"]; ov = d.get("overload", {})
        c9 = tj.get("post_c9", {}).get("f2"); cd = tj.get("tilecd_q")
        print(f"K{K:>3s} {short(k):26s} f2 {tj['pre_refit']['f2']:.4e} -> {tj['post_refit']['f2']:.4e}" + (f" -> C9 {c9:.4e}" if c9 else "") +
              (f"  CD(q) {cd['ldlq']:.4e}->{cd[max(x for x in cd if x.startswith('cd'))]:.4e}" if cd else "") +
              f"  g {d.get('g_scale') or 0:.4f}  ov {ov.get('q_absmax', 0):.3f} {ov.get('q_outer', 0):.2e} {ov.get('w_over', 0):.2e} {ov.get('tile_3x') or 0:.2e}")
checks = c1 and c2 and c3
v = (f"VERDICT #138-01: checks {'OK' if checks else 'FAILED (c1 ' + str(c1) + ', c2 ' + str(c2) + ', c3 ' + str(c3) + ')'}; " +
     (f"candidate {cand} (T-geo {res[cand][0]:.5f})" + (" = polish harmful under H_out" if cand == "none" else "") if cand else
      f"no arm passes (best {best}, T-geo {res[best][0]:.5f}{', drop (< 0.3%)' if 0.997 < res[best][0] <= 1.0 else ''}): refit = true stays"))
print("\n" + v)
