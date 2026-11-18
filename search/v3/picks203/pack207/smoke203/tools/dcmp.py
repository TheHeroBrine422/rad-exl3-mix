#!/usr/bin/env python3
"""dcmp.py A.jsonl B.jsonl -- decode numerics of two engines on the same prompts (gen.py outputs, greedy): per prompt the
steps until the first different token, and over every step before it KL(A || B) on the union of the two top-20 lists
(each renormalised over that union; a token missing from one side's list gets the side's smallest listed logprob, an
upper bound). Prints the mean / max per-step KL, the share of identical tokens and of prompts identical to the end."""
import json, math, sys

A = {j["id"]: j for j in map(json.loads, open(sys.argv[1]))}
B = {j["id"]: j for j in map(json.loads, open(sys.argv[2]))}
kls, same_steps, total_steps, full = [], 0, 0, 0
for k in A:
    a, b = A[k], B.get(k)
    if not b or a.get("error") or b.get("error") or not a.get("tokens") or not b.get("tokens"):
        print(f"{k}: missing"); continue
    n = min(len(a["tokens"]), len(b["tokens"]))
    d = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), n)
    same_steps += d; total_steps += n; full += d == n
    for i in range(d):
        ta, tb = a["top"][i] or {}, b["top"][i] or {}
        if not ta or not tb:
            continue
        keys = set(ta) | set(tb)
        fa, fb = min(ta.values()), min(tb.values())
        pa = {t: math.exp(ta.get(t, fa)) for t in keys}; pb = {t: math.exp(tb.get(t, fb)) for t in keys}
        za, zb = sum(pa.values()), sum(pb.values())
        kls.append(sum(pa[t] / za * math.log((pa[t] / za) / (pb[t] / zb)) for t in keys if pa[t] > 0))
    print(f"{k}: identical for {d} of {n} steps")
if kls:
    kls.sort()
    print(f"steps compared {len(kls)}: per-step KL(A||B) mean {sum(kls) / len(kls):.3g}, median {kls[len(kls) // 2]:.3g}, "
          f"max {kls[-1]:.3g}; identical tokens {same_steps} / {total_steps} ({same_steps / max(total_steps, 1):.3f}); "
          f"prompts identical to the end {full} / {len(A)}")
