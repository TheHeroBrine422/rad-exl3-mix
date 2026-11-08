#!/usr/bin/env python3
"""price82.py MODEL.json HPROXY.jsonl LEVELS_DIR NAME=PROXY.jsonl[:cb] ... -- #118 2(b) (../w82.md): measured V3 level
proxies (ourq proxy_err: one-sided, after the scale refit, undamped calib H) against knap5's KN_V3R price of the same
level: base(K) (the model's V2 ours-K hproxy errors, fractional K at the measured ladder positions of LEVELS_DIR; knap5's
phi interpolation only with W82_PHI=1: it is a KL calibration step, not a proxy error) x R8 (i8) / R4 (i4) x DAMPF
(attn q/k/v, KN_DAMP=1), divided by the tensor's hproxy ref (relative error).
gate / up: geometric mean of the two parts against the fused ffn_gate_up entry (knap5's KN_LEVELS convention).
Prints m = measured / price per level: geometric mean over tensors, per family, p10 / p90; and H_out / plain
one-sided ratios for the named pairs (PAIRS env "a/b,c/d")."""
import collections, glob, json, math, os, sys
import numpy as np

MJ, HP, LD = sys.argv[1:4]
md = json.load(open(MJ)); PHI = md["phi"]
REF = {}
for l in open(HP):
    d = json.loads(l); REF[d["w"]] = d["ref"]
HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z",
         "linear_attn.out_proj": "ssm_out", "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up",
         "mlp.down_proj": "ffn_down"}
DAMPF = {2: 0.994, 4: 0.981, 6: 0.991, 8: 0.955}
R8 = {1: 1.0, 3: 1.0, 4: 1.001, 5: 1.002, 6: 1.004, 6.5: 1.018, 7: 1.27}
R4 = {1: 1.001, 1.125: 1.0009, 1.25: 1.0016, 1.375: 1.0010, 1.5: 1.002, 1.625: 1.0037, 1.75: 1.0046, 1.875: 1.0070,
      2: 1.0088, 2.25: 1.0130, 2.5: 1.018, 2.75: 1.029, 3: 1.062}


def loglin(tab, K):
    ks = sorted(tab); K = min(max(K, ks[0]), ks[-1])
    if K in tab: return tab[K]
    lo = max(k for k in ks if k < K); hi = min(k for k in ks if k > K); f = (K - lo) / (hi - lo)
    return math.exp(math.log(tab[lo]) + f * (math.log(tab[hi]) - math.log(tab[lo])))


def read(pj):
    acc = collections.defaultdict(list)
    for l in open(pj):
        d = json.loads(l); k = d["key"].split(".")
        if len(k) < 6 or k[2] != "layers" or d.get("proxy_err") is None or d["proxy_err"] <= 0: continue
        n = HF2KN.get(".".join(k[4:6]))
        if n: acc[(f"blk.{k[3]}.{n}.weight", float(d["K"]))].append(d["proxy_err"])
    return {key: math.exp(sum(map(math.log, v)) / len(v)) for key, v in acc.items()}


LEV = collections.defaultdict(dict)
for pj in glob.glob(f"{LD}/K*/proxy.jsonl"):
    for (n, K), v in read(pj).items(): LEV[n][K] = v


def level_pos(w, K):
    m = LEV.get(w); a, b = float(math.floor(K)), float(math.ceil(K))
    if not m or K not in m or a not in m or b not in m or not (m[b] < m[K] < m[a]): return None
    return math.log(m[K] / m[a]) / math.log(m[b] / m[a])


T = {t["w"]: t for t in md["tensors"]}


def price(w, K, cb):
    t = T[w]; o = t["opts"]; a, b = math.floor(K), math.ceil(K)
    ea, eb = o[f"ours-K{a}"][1], o[f"ours-K{b}"][1]
    if a == b: e = ea
    else:
        f = K - a; fm = level_pos(w, K); fe = f if fm is None else fm
        e = math.exp(math.log(ea) + fe * (math.log(eb) - math.log(ea)))
        fa, fb = t["family_of"][f"ours-K{a}"], t["family_of"][f"ours-K{b}"]
        if fa != fb and os.environ.get("W82_PHI") == "1": e *= math.exp(f * (PHI[fb] - PHI[fa]))
    e *= loglin(R4, K) if cb == "i4" else loglin(R8, K)
    if w.split(".")[-2] in ("attn_qg", "attn_k", "attn_v"): e *= loglin(DAMPF, K)
    return e / REF[w]


FAM = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]
g = lambda v: math.exp(np.mean(np.log(v)))
sets = {}
print("level | K | cb | n | geo m | p10 | p90 | " + " | ".join(FAM))
for arg in sys.argv[4:]:
    name, rest = arg.split("=", 1); pj, cb = (rest.split(":") + ["i8"])[:2]
    meas = read(pj); sets[name] = meas
    byK = collections.defaultdict(list)
    for (w, K), v in meas.items():
        if w in T and w in REF: byK[K].append((w, v / price(w, K, cb)))
    for K, rows in sorted(byK.items()):
        ms = np.array([r for _, r in rows])
        fam = {f: [r for w, r in rows if w.split(".")[-2] == f] for f in FAM}
        print(f"{name} | {K:g} | {cb} | {len(ms)} | {g(ms):.3f} | {np.percentile(ms, 10):.3f} | {np.percentile(ms, 90):.3f} | "
              + " | ".join(f"{g(v):.3f}" if v else "–" for v in fam.values()))
if os.environ.get("PAIRS"):
    print("\npair (one-sided proxy a / b) | n | geo | p10 | p90 | " + " | ".join(FAM))
    for pr in os.environ["PAIRS"].split(","):
        a, b = pr.split("/"); A, B = sets[a], sets[b]
        ks = [k for k in A if k in B]
        r = np.array([A[k] / B[k] for k in ks])
        fam = {f: [A[k] / B[k] for k in ks if k[0].split(".")[-2] == f] for f in FAM}
        print(f"{a} / {b} | {len(r)} | {g(r):.3f} | {np.percentile(r, 10):.3f} | {np.percentile(r, 90):.3f} | "
              + " | ".join(f"{g(v):.3f}" if v else "–" for v in fam.values()))
