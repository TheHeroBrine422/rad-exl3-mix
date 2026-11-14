#!/usr/bin/env python3
"""gdec.py run PORT OUT | gdec.py cmp A.jsonl B.jsonl -- decode numerics check (w152, #179): `run` sends 8 fixed prompts
(below, written for this check; not from any corpus), 96 greedy tokens each with top-20 logprobs, to a radiance server and
writes one JSON line a prompt (tokens, chosen-token logprobs, top-20 lists). `cmp`: per prompt the steps before the first
different token, then over the common prefix: exact logprob equality (bit for bit as JSON floats), max |dlogprob| of the
chosen token, and a per-step KL estimate over the top-20 (renormalised) as a scale. Standard library only."""
import json, math, sys, urllib.request
PROMPTS = [
    "Explain step by step how a bicycle gear system changes the effort needed to climb a hill.",
    "Write a short Python function that merges two sorted lists, then explain its running time.",
    "Describe the water cycle to a ten-year-old, using an analogy from the kitchen.",
    "List five practical tips for keeping a sourdough starter healthy over a long holiday.",
    "Summarise the main trade-offs between electric and gas stoves for a small apartment.",
    "Translate into French and explain any idioms: 'It's raining cats and dogs, so let's stay in.'",
    "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the work.",
    "Draft a polite email asking a landlord to fix a leaking kitchen tap within the week.",
]
def run(port, out):
    with open(out, "w") as f:
        for i, p in enumerate(PROMPTS):
            req = {"model": "m", "prompt": p, "max_tokens": 96, "temperature": 0, "ignore_eos": True, "logprobs": 20}
            r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(req).encode(),
                                       {"Content-Type": "application/json"})
            j = json.loads(urllib.request.urlopen(r, timeout=600).read())
            lp = j["choices"][0]["logprobs"]
            f.write(json.dumps({"i": i, "tokens": lp["tokens"], "lp": lp["token_logprobs"], "top": lp["top_logprobs"]}) + "\n")
def cmp(a, b):
    A = [json.loads(l) for l in open(a)]; B = [json.loads(l) for l in open(b)]
    tot = same = exact = 0; dmax = 0.0; kls = []
    for x, y in zip(A, B):
        n = 0
        while n < min(len(x["tokens"]), len(y["tokens"])) and x["tokens"][n] == y["tokens"][n]: n += 1
        tot += len(x["tokens"]); same += n
        for t in range(n):
            exact += x["top"][t] == y["top"][t] and x["lp"][t] == y["lp"][t]
            dmax = max(dmax, abs(x["lp"][t] - y["lp"][t]))
            P, Q = x["top"][t], y["top"][t]; ks = set(P) & set(Q)
            zp = sum(math.exp(P[k]) for k in ks); zq = sum(math.exp(Q[k]) for k in ks)
            kls.append(sum(math.exp(P[k]) / zp * ((P[k] - math.log(zp)) - (Q[k] - math.log(zq))) for k in ks))
        print(f"prompt {x['i']}: {n} / {len(x['tokens'])} tokens before the first difference")
    kls.sort()
    print(f"tokens {same} / {tot} in common prefixes; bit-exact top-20 + chosen logprob at {exact} / {same} steps; "
          f"max |dlogprob| {dmax:.3g}; top-20 KL median {kls[len(kls)//2] if kls else 0:.3g} max {kls[-1] if kls else 0:.3g}")
if __name__ == "__main__":
    if sys.argv[1] == "run": run(int(sys.argv[2]), sys.argv[3])
    else: cmp(sys.argv[2], sys.argv[3])
