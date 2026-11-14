#!/usr/bin/env python3
"""bsum.py RES_DIR TAG... -- (w153, #180) bench.py summaries (bench-TAG.txt, warm runs dropped): per ctx the client
tok/s, the server step and tokens a step, the TTFT; plus the drafter's acceptance ratio from metrics-TAG.txt."""
import re, sys

R = re.compile(r"^(\S+) ctx (\S+) run \d+: .*?ttft (\d+) ms, client ([\d.]+) ms/tok, server ([\d.]+) ms/step x ([\d.]+) tok")
d = sys.argv[1]
for t in sys.argv[2:]:
    per = {}
    for line in open(f"{d}/bench-{t}.txt"):
        m = R.match(line)
        if m:
            per.setdefault(m.group(2), []).append([float(m.group(i)) for i in (3, 4, 5, 6)])
    acc = next((l.split()[-1] for l in open(f"{d}/metrics-{t}.txt") if l.startswith("radiance:draft_acceptance_ratio{")), "-")
    cols = []
    for c, v in per.items():
        ttft, cl, st, tk = (sum(x[i] for x in v) / len(v) for i in range(4))
        cols.append(f"ctx {c}: {1000 / cl:.1f} tok/s, step {st:.2f} ms x {tk:.2f} tok, ttft {ttft:.0f} ms")
    print(f"| {t} | " + " | ".join(cols) + f" | {acc if acc == '-' else f'{float(acc):.3f}'} |")
