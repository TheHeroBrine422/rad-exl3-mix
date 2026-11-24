#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plancheck226.py TRAINER.py OUT_DIR [qad220 args...] -- run only the trainer's build_data (no GPU, no model load) with a job's data flags and print where
each front item of --plan-ext starts and the depth it reaches by the last budget (w226: checks the plan the run will use before it takes cards)."""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("q220", sys.argv[1]); Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
a = Q.parser().parse_args([sys.argv[2]] + sys.argv[3:]); __import__("os").makedirs(a.out, exist_ok=True)
class D1:
    rank = 0
    def barrier(self): pass
P = Q.build_data(a, D1(), print)
C, steps, cum = P["C"], P["steps"], P["cum"]
B = float(P["budgets"][-1]); last = int(np.searchsorted(cum, B)) + 1
px = json.load(open(a.plan_ext)); nf = len(px.get("front_items", [])); n = len(C.items)
front = set(range(n - nf, n)) if nf else set()
print(f"plan: {n} items ({nf} front), {len(steps)} steps, {int(cum[-1])} rows; budget {B:.3g} at step {last}; first order entries {list(P['order'][:4])}")
reach = {}
for k, st in enumerate(steps[:last]):
    for s, segs in enumerate(st):
        for (it, sa, sb) in segs:
            if it in front: r = reach.setdefault(it, [k, s, 0]); r[2] = max(r[2], sb)
for it in sorted(front, key=lambda i: -C.items[i][2]):
    gi, i, ntok, nr = C.items[it]; r = reach.get(it)
    print(f"  {C.g[gi].name}:{i} tokens {ntok} rows {nr}: " + (f"stream {r[1]} from step {r[0]}, depth {r[2]} by the budget" if r else "not reached"))
# carried KV state (all streams, 64 KB a token: 16 attention layers x 4 KV heads x 256 x bf16 K + V) at the end of each step up to the budget: the host RAM a
# run needs for parked states (minus <= 2 x --state-card-gb kept on the cards); D256-25 measured RSS ~24 GB + this
car = []
for k, st in enumerate(steps[:last]):
    t = 0
    for segs in st:
        if segs:
            it, sa, sb = segs[-1]
            if sb < C.items[it][2]: t += sb
    car.append(t * 65536 / 2**30)
pk = int(np.argmax(car)); print(f"carried KV state: peak {car[pk]:.1f} GiB at step {pk + 1}; at steps 50 / 100 / 200 / 300 / 400 / 500: " +
                              " / ".join(f"{car[s - 1]:.1f}" for s in (50, 100, 200, 300, 400, 500) if s <= len(car)))
