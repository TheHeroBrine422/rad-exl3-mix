#!/usr/bin/env python3
"""report71.py OUTDIR[,OUTDIR..] [W64_CD_DIR] > report.md -- w71's tables (../w71.md, #113) from q71.py's out/ dirs
(colo and Thor: disjoint arm groups, merged per (tensor, sample, K); final files and the .arms lines of jobs still
running, so it can be read mid-run: n says how many (tensor, sample) pairs a cell has). `pull71.sh` copies both boxes'
out/ to data/colo and data/thor here.

r(arm) = rel(arm) / rel(V16) per (tensor, sample, K); G = geometric mean over the pairs that have both; t = per-tensor
geometric mean over its samples (only tensors with both samples count for "no tensor worse" once complete).
W64_CD_DIR: w64's out/cd/mul1_i8 (gate 2: S16 == w64's settle64 LDLQ rel)."""
import glob, json, math, os, re, sys
from collections import defaultdict

FAM = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]
out, w64 = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
rel = defaultdict(dict); sec = defaultdict(dict); kd = {}
files = [f for d in out.split(",") for f in glob.glob(os.path.join(d, "s*", "*.jsonl")) + glob.glob(os.path.join(d, "s*", "*.jsonl.arms"))]
for f in files:
    m = re.match(r"(\d+)-(\w+?)-K(\d+)-(\w+)\.jsonl", os.path.basename(f))
    s = int(os.path.basename(os.path.dirname(f))[1:]); key = (int(m[1]), m[2], int(m[3]), s)
    for l in open(f):
        try: o = json.loads(l)
        except ValueError: continue
        if "arms" in o:                                   # a final file
            kd[m[2]] = o["K"]
            for a in o["arms"]: rel[key][a] = o["rel"][a]; sec[key][a] = o["sec_ldlq"][a]
        else: rel[key][o["arm"]] = o["rel"]; sec[key][o["arm"]] = o["sec_ldlq"]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
P = print

P("## Gate\n")
g2 = []
for key in sorted(k for k in rel if "S16" in rel[k]):
    r = rel[key]; L, w, K, s = key
    a = "F16 == S16 " + ("yes" if r.get("F16") == r["S16"] else f"NO ({r.get('F16')} vs {r['S16']})") if "F16" in r else "F16 -"
    b = ("T16 == V16 " + ("yes" if r["T16"] == r["V16"] else f"NO ({r['T16']} vs {r['V16']})")) if "T16" in r and "V16" in r else "T16/V16 -"
    c = ""
    if w64:
        f = os.path.join(w64, f"s{s}", f"{L}-{w}-K{K}.jsonl")
        if os.path.exists(f):
            v = json.loads(open(f).readline())["rel"]["ldlq"]
            c = "; S16 == w64 settle64 ldlq " + ("yes" if v == r["S16"] else f"NO ({v} vs {r['S16']})")
    P(f"- {L} {w} K{K} s{s}: S16 {r['S16']:.6g}; {a}; {b}{c}")

arms_all = sorted({a for k in rel for a in rel[k]} - {"S16", "F16", "T16"},
                  key=lambda a: (a.endswith("r"), int(re.sub(r"\D.*", "", a[1:]) or 0), a[0] != "V", int(a.split("-")[1].rstrip("r")) if "-" in a else 0))
Ks = sorted({k[2] for k in rel})


def ratios(arm, K):
    """{(layer, w): [r per sample]}"""
    d = defaultdict(list)
    for (L, w, KK, s), r in rel.items():
        if KK == K and arm in r and "V16" in r and arm != "V16": d[(L, w)].append(r[arm] / r["V16"])
    return d


def stat(arm, K):
    d = ratios(arm, K)
    allr = [x for v in d.values() for x in v]
    t = {k: gm(v) for k, v in d.items() if len(v) == 2}
    worst = max(t.items(), key=lambda kv: kv[1]) if t else (None, float("nan"))
    return gm(allr), len(allr), worst, sum(v > 1.0 for v in t.values()), len(t)


P("\n## G = geometric mean of rel(arm) / rel(V16) per K (n pairs; worst tensor t; tensors with t > 1 of complete)\n")
P("| arm | " + " | ".join(f"K{K}" for K in Ks) + " |"); P("|---|" + "---|" * len(Ks))
for a in arms_all:
    if a == "V16": continue
    cells = []
    for K in Ks:
        G, n, (wk, wv), nbad, nt = stat(a, K)
        cells.append("" if n == 0 else f"{G:.4f} (n {n}; worst {wv:.3f} {wk[0]}:{wk[1]}; {nbad}/{nt} > 1)" if wk else f"{G:.4f} (n {n})")
    P(f"| {a} | " + " | ".join(cells) + " |")

P("\n## Per family G (both samples, 3 tensors)\n")
for K in Ks:
    arms = [a for a in arms_all if a != "V16" and any(ratios(a, K).values())]
    if not arms: continue
    P(f"\n**K{K}**\n"); P("| family | " + " | ".join(arms) + " |"); P("|---|" + "---|" * len(arms))
    for fam in FAM:
        row = []
        for a in arms:
            v = [x for (L, w), xs in ratios(a, K).items() if w == fam for x in xs]
            row.append(f"{gm(v):.4f} ({len(v)})" if v else "")
        P(f"| {fam} | " + " | ".join(row) + " |")

P("\n## Rule (#113)\n")
for Lw, cands in ((20, ["B20-256", "B20-1024", "B20-4096"]), (18, ["V18", "B18-1024", "B18-4096"])):
    npass = 0; lines = []
    for K in (2, 3, 4):
        st = {a: stat(a, K) for a in cands if stat(a, K)[1]}
        if not st: lines.append(f"K{K}: no data"); continue
        b = min(st, key=lambda a: st[a][0]); G, n, (wk, wv), nbad, nt = st[b]
        ok = G <= 0.970 and nbad == 0
        npass += ok
        lines.append(f"K{K}: best {b} G {G:.4f} (n {n}), tensors with t > 1: {nbad}/{nt} (worst {wv:.3f}) -> {'pass' if ok else 'no'}")
    P(f"- **L = {Lw}**: " + "; ".join(lines) + f". **{'ADOPT-FOR-27B-CHECK' if npass >= 2 else 'NOT ADOPTED'}** "
      f"({npass} of 3 K pass{', provisional: data incomplete' if any(stat(c, K)[1] < 54 for K in (2, 3, 4) for c in cands[-1:]) else ''})")

P("\n## Encode time (LDLQ pass seconds, sum over the pairs that have both arms, ratio to V16; s per tile = 2 tail-biting runs)\n")
P("| arm | " + " | ".join(f"K{K}" for K in Ks) + " |"); P("|---|" + "---|" * len(Ks))
for a in arms_all:
    cells = []
    for K in Ks:
        ks = [k for k in sec if k[2] == K and a in sec[k] and "V16" in sec[k] and k[1] in kd]
        if not ks: cells.append(""); continue
        sa = sum(sec[k][a] for k in ks); sv = sum(sec[k]["V16"] for k in ks); nt = sum(128 * kd[k[1]] / 256 for k in ks)
        cells.append(f"×{sa / sv:.1f} ({1e3 * sa / nt:.0f} ms/tile)")
    P(f"| {a} | " + " | ".join(cells) + " |")
