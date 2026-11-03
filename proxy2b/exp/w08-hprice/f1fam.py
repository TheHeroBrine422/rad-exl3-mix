#!/usr/bin/env python3
"""f1fam.py ROWERR2.jsonl DIR[,DIR..] [--json OUT] -- F1 per family: after w10-hout-attrib (B3) has measured the Σdy²
H_out on one family at a time, does the two-sided price (e2) or the one-sided one (e1) get each family's KL change
right? ROWERR2: rowerr2.py over k{3,2}base{,_s1,_s2} and k{3,2}edy{,_s1,_s2} with h = c18 hdiag-snap1024-dy (a family's
tensors in w10's subset arm k{K}h-<fam> are those of the full edy arm: ourq quantizes every linear on its own, same
seed). DIRs hold evalq --out rows <tag>.f32 ([kl, top1, gen, 0]); the first that has a tag wins.
Per family f and K: e1_f, e2_f = geometric mean over f's tensors and the seed pairs of the H / base error ratio;
g_f = -mean_s ln(KL[k{K}h-f_s] / KL[k{K}base_s]) over the seeds both exist for, SE = max(SD / sqrt(n), 0.005).
Rule (README, draft): for every (f, K) where the proxies disagree (e1_f > 1.05 and e2_f < 0.95), e2 is right if
g_f / SE >= 2 and e1 is right if g_f / SE <= -2; F1's oursH pricing passes if e2 is right in every disagreeing case
and e1 in none."""
import collections, json, math, os, sys
import numpy as np

a = sys.argv[1:]
OUT = a[a.index("--json") + 1] if "--json" in a else None
RW, DIRS = a[0], a[1].split(",")
FAM = {"gdn_in": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"), "attn_in": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
       "mlp_in": ("mlp.gate_proj", "mlp.up_proj"), "writers": ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")}
R = collections.defaultdict(dict)
for l in open(RW):
    r = json.loads(l); R[r["arm"][2:]][r["key"]] = r
def sfx(s): return "" if s == 0 else f"_s{s}"
def kl(tag):
    for d in DIRS:
        f = f"{d}/{tag}.f32"
        if os.path.exists(f):
            x = np.fromfile(f, np.float32).reshape(-1, 4); g = x[:, 2] > 0
            return float(x[g, 0].astype(np.float64).mean())
    return None
res, verdict = {}, []
for K in (3, 2):
    for f, types in FAM.items():
        e = {m: [] for m in ("e1", "e2")}
        for s in range(3):
            b, h = R.get(f"k{K}base{sfx(s)}"), R.get(f"k{K}edy{sfx(s)}")
            if not b or not h: continue
            for k in b:
                if k.split(".", 4)[4] in types and k in h:
                    for m in e: e[m].append(math.log(h[k][m] / b[k][m]))
        if not e["e1"]: continue
        e1, e2 = math.exp(np.mean(e["e1"])), math.exp(np.mean(e["e2"]))
        ls = []
        for s in range(4):
            x, y = kl(f"k{K}h-{f}{sfx(s)}"), kl(f"k{K}base{sfx(s)}")
            if x and y: ls.append(math.log(x / y))
        g = -np.mean(ls) if ls else None
        se = max(np.std(ls, ddof=1) / math.sqrt(len(ls)), 0.005) if len(ls) > 1 else None
        dis = e1 > 1.05 and e2 < 0.95
        who = None if g is None or se is None or not dis else ("e2" if g / se >= 2 else "e1" if g / se <= -2 else "unclear")
        if dis: verdict.append(who)
        res[f"K{K} {f}"] = {"e1": e1, "e2": e2, "seeds": len(ls), "gain": g, "se": se, "disagree": dis, "right": who}
        print(f"K{K} {f:8}: e1 x{e1:.3f}  e2 x{e2:.3f}  measured " +
              (f"gain {g:+.2%} +- {se:.2%} ({len(ls)} seeds)" if se else "n/a") + (f"  -> {who}" if dis else ""))
ok = bool(verdict) and all(v == "e2" for v in verdict)
print(f"F1 per-family: {'PASS' if ok else 'FAIL' if any(v == 'e1' for v in verdict) else 'INCONCLUSIVE'} "
      f"({verdict.count('e2')} e2 / {verdict.count('e1')} e1 / {sum(v in (None, 'unclear') for v in verdict)} open, of {len(verdict)} disagreeing cases)")
if OUT: json.dump({"cases": res, "pass": ok}, open(OUT, "w"), indent=1)
