#!/usr/bin/env python3
"""tiemargin.py A.jsonl B.jsonl -- (w153, #180) where two greedy runs of the same prompts (gen.py / genimg.py outputs)
first pick different tokens: per diverging prompt the step, both picks, and A's top-1 - top-2 logprob margin there and
B's (a near-tie on both sides means the runs part on numerics, not on a wrong token)."""
import json, sys

A = {j["id"]: j for j in map(json.loads, open(sys.argv[1]))}
B = {j["id"]: j for j in map(json.loads, open(sys.argv[2]))}


def margin(top):
    v = sorted((top or {}).values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else float("nan")


for k, a in A.items():
    b = B.get(k)
    if not b or not a.get("tokens") or not b.get("tokens"):
        continue
    n = min(len(a["tokens"]), len(b["tokens"]))
    d = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    if d is None:
        continue
    print(f"{k}: step {d}: A {a['tokens'][d]!r} B {b['tokens'][d]!r}; top-1 - top-2 logprob margin "
          f"A {margin(a['top'][d]):.4f} B {margin(b['top'][d]):.4f}")
