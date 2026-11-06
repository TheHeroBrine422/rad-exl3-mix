#!/usr/bin/env python3
"""verdict.py A ARM [ARM ...] -- the ASYM27.md rule over the seed pairs asym27.sh wrote (A/<ARM>/pair.json: control ->
<ARM>a on golden-search27, tools/paired.py: kld_a = control, kld_b = asym, bootstrap SE over documents).
  r_i = kld_b / kld_a per pair; g = exp(mean log r_i) (geometric mean)
  PASS          g <= 0.97 and every r_i < 1         -> adopt the writers' asym target (rho 0) as a quantizer setting
  FAIL          g >= 1.00                           -> proxy negative (BACKLOG "proxy negatives to reconsider")
  INCONCLUSIVE  otherwise                           -> not adopted (ASYM27.md: the one pre-registered third pair)
--paired TOOL --golden G: also the asym arms' own seed floor (first two arms' <ARM>a rows, paired) beside the controls'."""
import argparse, json, math, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("A"); ap.add_argument("arms", nargs="+")
ap.add_argument("--paired", default=None); ap.add_argument("--golden", default=None)
a = ap.parse_args()
rs = []
for arm in a.arms:
    p, m = f"{a.A}/{arm}/pair.json", f"{a.A}/{arm}/pair-meta.json"
    if not os.path.exists(p): sys.exit(f"{arm}: no {p} (asym27.sh STEPS=pair)")
    d = json.load(open(p)); meta = json.load(open(m)) if os.path.exists(m) else {}
    r = d["kld_b"] / d["kld_a"]; rs.append(r)
    print(f"{arm:10s} control {os.path.basename(meta.get('control', '?')):22s} {d['kld_a']:.6f} -> asym {d['kld_b']:.6f}: "
          f"r {r:.4f} (z {d['z']:+.1f}, {d['docs']} docs, {d['positions']} positions; noop gate {meta.get('noop', '?')})")
g = math.exp(sum(math.log(r) for r in rs) / len(rs))
v = "PASS" if g <= 0.97 and all(r < 1 for r in rs) else "FAIL" if g >= 1.0 else "INCONCLUSIVE"
if a.paired and a.golden and len(a.arms) >= 2:
    x, y = (f"{a.A}/{arm}a.json.rows" for arm in a.arms[:2])
    if os.path.exists(x) and os.path.exists(y):
        out = subprocess.run([sys.executable, a.paired, a.golden, x, y], capture_output=True, text=True).stdout.strip().splitlines()
        if out:
            d = json.loads(out[-1]); print(f"asym seed floor {a.arms[0]}a -> {a.arms[1]}a: r {d['kld_b'] / d['kld_a']:.4f} (z {d['z']:+.1f})")
print(f"ASYM27 VERDICT {v}: geometric mean {g:.4f} over {len(rs)} seed pairs ({', '.join(f'{r:.4f}' for r in rs)}); "
      f"rule: PASS g <= 0.97 and every r < 1, FAIL g >= 1.00")
