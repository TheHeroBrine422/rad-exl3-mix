#!/usr/bin/env python3
"""rules.py O -- d1109's three rules (README.md: base, amendment 1, amendment 2) from O/result.txt's "pair" lines.
Ratios are the paired ratios ARM / BASE (seed suffixes "", _s1, _s2); "mean ratio" is their arithmetic mean (summ.py's),
"gain" = -mean log ratio. Amendment rules use seeds 0 and 1 (the healed and rqp arms' seeds); the base rule uses every
seed run.sh made (0-2). A rule with missing pairs prints "incomplete" and counts as not passed."""
import math, re, sys
o = sys.argv[1]
P = {}
for l in open(f"{o}/result.txt"):
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+)", l)
    if m: P[(m[1], m[2])] = float(m[3])
SFX = {0: "", 1: "_s1", 2: "_s2"}

def rs(base, arm, seeds):
    r = [P.get((base + SFX[s], arm + SFX[s])) for s in seeds]
    return None if any(x is None for x in r) else r

def show(r): return "incomplete" if r is None else " ".join(f"{x:.4f}" for x in r)

def passes(base, arm, seeds, k):
    r = rs(base, arm, seeds)
    ok = r is not None and sum(r) / len(r) <= 0.95 and all(x < 1 for x in r)
    mr = "" if r is None else f" mean {sum(r) / len(r):.4f}"
    print(f"  K{k} {arm} vs {base} (seeds {','.join(map(str, seeds))}): {show(r)}{mr} -> {'pass' if ok else 'fail'}")
    return ok

def gain(base, arm, seeds):
    r = rs(base, arm, seeds)
    return None if r is None else -sum(math.log(x) for x in r) / len(r)

print("BASE RULE (rq vs stack, mean ratio <= 0.95 and every seed < 1, K3 and K2):")
base_rq = all([passes(f"k{k}edyswjasym_r0", f"k{k}rqswjasym_r0", [0, 1, 2], k) for k in (3, 2)])
print("AMENDMENT 1 (rq+heal vs stack+heal, the same test, seeds 0 / 1):")
a1_rq = all([passes(f"k{k}edyswjasym_r0_sp", f"k{k}rqswjasym_r0_sp", [0, 1], k) for k in (3, 2)])
rq_ok = base_rq and a1_rq
print(f"=> rq earns a 27B test: {rq_ok}" + ("" if rq_ok else " (base rule passed, healing absorbs the gain)" if base_rq else ""))
print("AMENDMENT 2 (rqp):")
keep = True
for h in ("", "_sp"):
    for k in (3, 2):
        g_rq = gain(f"k{k}edyswjasym_r0{h}", f"k{k}rqswjasym_r0{h}", [0, 1])
        g_rqp = gain(f"k{k}edyswjasym_r0{h}", f"k{k}rqpswjasym_r0{h}", [0, 1])
        f = None if g_rq is None or g_rqp is None or g_rq <= 0 else g_rqp / g_rq
        keep &= f is not None and f >= 0.75
        gs = lambda g: "incomplete" if g is None else f"{g:+.4f}"
        print(f"  K{k} {'healed' if h else 'unhealed'}: gain vs stack rq {gs(g_rq)}, rqp {gs(g_rqp)}, "
              f"kept {'n/a (no rq gain)' if f is None else f'{f:.0%}'}")
if rq_ok:
    print(f"=> rq passed: rqp replaces rq (per-level ladder method) if it keeps >= 75% everywhere: {keep}")
else:
    print("  rq failed, so rqp alone faces the base rule and amendment 1's rule:")
    b = all([passes(f"k{k}edyswjasym_r0", f"k{k}rqpswjasym_r0", [0, 1], k) for k in (3, 2)])
    a = all([passes(f"k{k}edyswjasym_r0_sp", f"k{k}rqpswjasym_r0_sp", [0, 1], k) for k in (3, 2)])
    print(f"=> rqp earns the 27B test instead: {b and a}")
