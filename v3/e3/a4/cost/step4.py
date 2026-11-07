#!/usr/bin/env python3
"""step4.py OUTDIR ARM [REF] -- #46-03 step 4 (v3/e3/a4/README.md) from knap5a solves in OUTDIR (<arm>-f-<size>-l<lambda>.jsonl,
mkjobs.py): per size 2.0 / 2.5 / 3.0, REF (default noa4)'s fast variant F (w58 report.py's rule: the lowest predicted KLD
among the lambda grid's solutions whose GEMM time is within 2% of the fastest) against ARM:
  time test   ARM's least GEMM time among its same-size solutions with KLD <= F's (no interpolation: a solution that exists)
              -> 1 - t / t_F; PASS at >= 10%. Also the best such reduction anywhere on REF's curve (each REF point).
  bytes test  ARM at size - 0.03 / - 0.06: a solution with KLD <= F's and time <= F's -> PASS (>= 0.03 bpw smaller).
Context (not the rule): the time cut if ARM may sit 1% / 3% above F's KLD.
Prints the table, ARM's A4 use (tensors, families, param share) at the matching point, and one 'STEP4' line per size."""
import collections, glob, json, os, re, sys

D, ARM = sys.argv[1], sys.argv[2]; REF = sys.argv[3] if len(sys.argv) > 3 else "noa4"
P = collections.defaultdict(list)
for p in glob.glob(f"{D}/*.jsonl"):
    m = re.fullmatch(r"(.+?)-f-([\d.]+)-l([\de.+-]+)", os.path.basename(p)[:-6])
    if not m: continue
    for l in open(p):
        if l.strip(): r = json.loads(l); P[(m[1], float(m[2]))].append(r)
def fast(pts):
    tmin = min(r["t_lin_ms"] for r in pts)
    return min((r for r in pts if r["t_lin_ms"] <= 1.02 * tmin), key=lambda r: r["kld"])
def best_at(pts, kld, tmax=None):
    ok = [r for r in pts if r["kld"] <= kld * (1 + 1e-9) and (tmax is None or r["t_lin_ms"] <= tmax)]
    return min(ok, key=lambda r: r["t_lin_ms"]) if ok else None
def a4use(r):
    if not r or "picks" not in r: return "-"
    c = collections.Counter(); n = 0
    for w, (opt, mode) in r["picks"].items():
        if mode == "a4" or mode.startswith("a4"): n += 1; c[w.split(".")[-2]] += 1
    return f"{n} tensors {dict(c)}" if n else "none"
print(f"# step 4: {ARM} vs {REF} (knap5a predictions)\n")
print("| size | ref fast KLD | ref t ms | arm t at <= KLD | time -% | best -% on ref curve | -0.03 bpw | -0.06 bpw | -% at KLD +1% / +3% | A4 use |")
print("|---|---|---|---|---|---|---|---|---|---|")
lines = []
for s in (2.0, 2.5, 3.0):
    ref, arm = P.get((REF, s)), P.get((ARM, s))
    if not ref or not arm: print(f"| {s} | missing ({len(ref or [])} / {len(arm or [])}) |"); continue
    F = fast(ref); b = best_at(arm, F["kld"])
    red = 1 - b["t_lin_ms"] / F["t_lin_ms"] if b else None
    curve = [(1 - x["t_lin_ms"] / r["t_lin_ms"]) for r in ref for x in [best_at(arm, r["kld"])] if x]
    bt = {}
    for d in (0.03, 0.06):
        sm = P.get((ARM, round(s - d, 2)))
        bt[d] = None if not sm else best_at(sm, F["kld"], F["t_lin_ms"])
    tol = [best_at(arm, F["kld"] * (1 + e)) for e in (0.01, 0.03)]
    tols = " / ".join(f"{100 * (1 - x['t_lin_ms'] / F['t_lin_ms']):.1f}" if x else "-" for x in tol)
    ok_t = red is not None and red >= 0.10; ok_b = bt[0.03] is not None
    print(f"| {s} | {F['kld']:.5f} | {F['t_lin_ms']:.1f} | {b['t_lin_ms'] if b else float('nan'):.1f} | "
          f"{100 * red if red is not None else float('nan'):.1f} | {100 * max(curve) if curve else float('nan'):.1f} | "
          f"{'yes' if bt[0.03] else 'no' if (ARM, round(s - 0.03, 2)) in P else '-'} | "
          f"{'yes' if bt[0.06] else 'no' if (ARM, round(s - 0.06, 2)) in P else '-'} | {tols} | {a4use(b)} |")
    lines.append(f"STEP4 {s}: time -{100 * red if red is not None else float('nan'):.1f}% ({'PASS' if ok_t else 'fail'}), "
                 f"bytes -0.03 {'PASS' if ok_b else 'fail'}")
print(); print("\n".join(lines))
print("STEP4 VERDICT:", "BUILD (rule met at " + ", ".join(l.split(":")[0][6:] for l in lines if "PASS" in l) + ")"
      if any("PASS" in l for l in lines) else "A4 stays out (rule not met at 2.0 / 2.5 / 3.0)")
