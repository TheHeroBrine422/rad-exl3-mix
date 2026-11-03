#!/usr/bin/env python3
"""selftest_f7.py DIR -- synthetic V2-stream layout for f7replay.py: 300 tensors, 3 targets, 40 fits; candidates change
n ~ 1..60 tensors from the current best; true change in log KLD = sum of per-tensor effects (mean slightly positive
for big moves: the model's composition error), prediction = true - noise growing with sqrt(n). Expected: RMS error
rising with n, win rate falling at large n, positive cap estimates at small c."""
import json, math, os, random, sys
D = sys.argv[1]; random.seed(0)
for s in ("evq", "knap/mixes", "auto"): os.makedirs(f"{D}/{s}", exist_ok=True)
W = [f"blk.{i}.w" for i in range(300)]
res = open(f"{D}/evq/results.jsonl", "w")
best = {T: ({}, 0.0, 9.0) for T in ("2.50", "3.00", "4.00")}     # mix, true log kld, measured
for N in range(40):
    open(f"{D}/auto/fit-s{N}.txt", "w").write("model ... slice offsets f2 0.85 f3 0.95\n")
    fr = open(f"{D}/auto/frontier-s{N}.txt", "w")
    for T in best:
        for kind, nmax in (("", 60), ("-ts0", 30), ("-ls0", 4)):
            mix0, k0, _ = best[T]; mix = dict(mix0)
            n = random.randint(1, nmax)
            for w in random.sample(W, n): mix[w] = f"c{random.randint(0, 5)}"
            eff = sum(random.gauss(0.004, 0.01) for _ in range(n)) - 0.01     # small moves help a little on average
            k = k0 + eff; name = f"knap-s{N}{kind}-sc4.00-e4-{T}"
            p = k - random.gauss(0, 0.005 * math.sqrt(n))
            fr.write(f"{name:30} bpw {T}  pred KLD {math.exp(p - 3):.6f}\n" if not kind else
                     f"{name:30} bpw {T}  pred (mean model) {math.exp(p - 3):.6f}  differs in {n} tensors\n")
            with open(f"{D}/knap/mixes/{name}.mix", "w") as f:
                for w, c in sorted(mix.items()): f.write(f"{w} {c}\n")
            sl = random.choice(["a", "f2", "f3"]); o = {"a": 1.0, "f2": 0.85, "f3": 0.95}[sl]
            meas = k + random.gauss(0, 0.01)
            res.write(json.dumps({"tag": f"{sl}-{name}", "kld": math.exp(meas - 3) * o}) + "\n")
            if meas < best[T][2]: best[T] = (mix, k, meas)
print("synthetic stream written")
