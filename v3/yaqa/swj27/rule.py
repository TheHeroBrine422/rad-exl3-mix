#!/usr/bin/env python3
"""rule.py RESULT.txt ARM... -- SWJ27.md's verdict from swapfit.sh's "pair <arm> -> <arm>swj: {paired.py json}" lines
(the last line per arm). Gating pairs: base and s1 (K2.5, no H_out, rotation seeds 0 and 1): PASS if the geometric mean
of the two KLD ratios is <= 0.98 and both ratios are < 1. bn / bns1 (on H_out) are reported, not gating."""
import json, math, re, sys

res, arms = sys.argv[1], sys.argv[2:]
pr = {}
for line in open(res):
    m = re.search(r"pair (\S+) -> (\S+)swj: (\{.*\})", line)
    if m and m.group(1) == m.group(2): pr[m.group(1)] = json.loads(m.group(3))
def ratio(p): return p["kld_b"] / p["kld_a"]
for a in arms:
    if a in pr: print(f"{a} -> {a}swj: ratio {ratio(pr[a]):.4f} (paired z {pr[a]['z']:.2f}, {pr[a]['docs']} docs)")
for lab, (x, y), gate in (("K2.5 no H_out", ("base", "s1"), True), ("K2.5 on bn H_out", ("bn", "bns1"), False),
                          ("K4 no H_out", ("base4", "base4s1"), False)):
    if x not in pr or y not in pr:
        if gate: print(f"SWJ27 RULE: incomplete ({', '.join(a for a in (x, y) if a not in pr)} missing)")
        continue
    r = [ratio(pr[x]), ratio(pr[y])]; gm = math.sqrt(r[0] * r[1])
    ok = gm <= 0.98 and max(r) < 1
    print(f"SWJ27 {'RULE' if gate else 'REPORT'} {lab}: ratios {r[0]:.4f} / {r[1]:.4f}, geo mean {gm:.4f} -> "
          f"{'PASS' if ok else 'FAIL'}{'' if gate else ' (not gating)'}")
