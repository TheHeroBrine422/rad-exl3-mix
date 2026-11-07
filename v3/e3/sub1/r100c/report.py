#!/usr/bin/env python3
"""report.py O -- #100c quality report (r100c.md rules) from run4070.sh's O/pairs.jsonl and the arms' proxy.jsonl:
per K and seed the hash / i8 KL ratios on search (deciding) and verify, geo over seeds, the quality verdict, the d1112
reproduction check, the hiqr readout, and per-class proxy-error ratios (cf. w27's CPU table). Markdown on stdout."""
import json, math, os, sys, collections
O = sys.argv[1]
P = {}
for l in open(os.path.join(O, "pairs.jsonl")):
    d = json.loads(l); P[d["pair"]] = d
sfx = lambda s: "" if s == 0 else f"_s{s}"
geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None
f4 = lambda x: "—" if x is None else f"{x:.4f}"
Ks, S = (2, 3, 4), (0, 1)

def row(a, b, pre=""):
    d = P.get(f"{pre}{a} -> {pre}{b}"); return d

print("## Quality (0.8B, fake-quant, edy trunk quant)\n")
print("hash / i8 = gen-masked mean KL ratio, per seed (z = paired diff / per-document bootstrap SE), geo = geometric mean over seeds.\n")
print("| K | golden | KL i8 s0 / s1 | KL hash s0 / s1 | ratio s0 (z) | ratio s1 (z) | geo |")
print("|---|---|---|---|---|---|---|")
G = {}
for k in Ks:
    for gname, pre in (("search", ""), ("verify", "vf-")):
        ds = [row(f"i8k{k}{sfx(s)}", f"hashk{k}{sfx(s)}", pre) for s in S]
        rs = [d["ratio"] for d in ds if d]
        G[(k, gname)] = geo(rs) if len(rs) == len(S) else None
        cell = lambda d, key: "—" if not d else f"{d[key]:.4f}"
        print(f"| {k} | {gname} | {' / '.join(cell(d, 'kld_A') for d in ds)} | {' / '.join(cell(d, 'kld_B') for d in ds)} | "
              + " | ".join("—" if not d else f"{d['ratio']:.4f} ({d['z']:+.1f})" for d in ds) + f" | **{f4(G[(k, gname)])}** |")
print()
srch = [G[(k, "search")] for k in Ks]; ver = [G[(k, "verify")] for k in Ks]
if None in srch:
    verdict = "INCOMPLETE (missing pairs)"
else:
    if any(g > 1.05 for g in srch): verdict = "FAIL, and > ×1.05 at some K (reject regardless of speed)"
    elif all(g <= 1.010 for g in srch):
        frag = [k for k, v in zip(Ks, ver) if v is not None and v > 1.020]
        verdict = f"PASS but FRAGILE (verify geo > 1.020 at K{frag})" if frag else "PASS (search geo ≤ 1.010 at K2, K3, K4)"
    else: verdict = "FAIL (search geo > 1.010 at K" + ", K".join(str(k) for k, g in zip(Ks, srch) if g > 1.010) + ")"
    near = [k for k, g in zip(Ks, srch) if abs(g - 1.010) <= 0.005]
    if near: verdict += f"; at the bar (±0.5%) at K{near}"
print(f"**Quality verdict:** {verdict}\n")

print("## Side checks\n")
print("Reproduction (d1112's edy `k{K}ste` vs this run's `i8`, search; same recipe, expected 1.0000):\n")
for k in Ks:
    xs = [P.get(f"d1112:k{k}ste{sfx(s)} -> i8k{k}{sfx(s)}") for s in S]
    print(f"- K{k}: " + ", ".join("—" if not d else f"s{s} {d['ratio']:.5f} (KL {d['kld_A']:.5f} → {d['kld_B']:.5f})" for s, d in zip(S, xs)))
print("\nIQR's A (`hiqr`, 0xB5297B, seed 0) — readout:\n")
print("| K | hiqr / i8 search | hiqr / i8 verify | hiqr / hash (searched A) search |")
print("|---|---|---|---|")
for k in Ks:
    c = lambda key: "—" if key not in P else f"{P[key]['ratio']:.4f} ({P[key]['z']:+.1f})"
    print(f"| {k} | {c(f'i8k{k} -> hiqrk{k}')} | {c(f'vf-i8k{k} -> vf-hiqrk{k}')} | {c(f'hashk{k} -> hiqrk{k}')} |")

# proxy error (exllamav3's own, per tensor) by class
def cls(key):
    for c in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if key.endswith(c): return c
    return key.rsplit(".", 1)[-1]
def perr(tag):
    f = os.path.join(O, f"q-{tag}", "proxy.jsonl")
    if not os.path.exists(f): return None
    r = {}
    for l in open(f):
        d = json.loads(l); r[d["key"]] = d["proxy_err"]
    return r
print("\n## Per-tensor proxy error (exllamav3's, after the scale refit): hash / i8, geo over tensors and seeds\n")
cl = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
print("| K | all | " + " | ".join(cl) + " |"); print("|---|---|" + "---|" * len(cl))
for k in Ks:
    R = collections.defaultdict(list)
    for s in S:
        a, b = perr(f"i8k{k}{sfx(s)}"), perr(f"hashk{k}{sfx(s)}")
        if not a or not b: continue
        for key in a.keys() & b.keys():
            if a[key] > 0 and b[key] > 0: R[cls(key)].append(b[key] / a[key]); R["all"].append(b[key] / a[key])
    print(f"| {k} | {f4(geo(R['all']))} (n {len(R['all'])}) | " + " | ".join(f4(geo(R[c])) for c in cl) + " |")
