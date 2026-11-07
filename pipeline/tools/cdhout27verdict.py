#!/usr/bin/env python3
"""cdhout27verdict.py A ARM [ARM ...] -- CDHOUT27.md's rule over the seed pairs cdhout27.sh wrote (A/<ARM>/pair.json:
control -> <ARM>cd on golden-search27, tools/paired.py: kld_a = control, kld_b = tile CD + H_out arm). The CD27.md rule
(v3/e3/settle/CD27.md), unchanged:
  r_i = kld_b / kld_a per pair; g = exp(mean log r_i)
  PASS  g <= 1.000 and every r_i <= 1.01 (two pairs)  -> tile CD stays on at K >= 4 levels (with H_out)
  FAIL  otherwise                                     -> tile CD off where H_out applies (quant.tile_cd: K < 4 only)"""
import json, math, os, sys

A, arms = sys.argv[1], sys.argv[2:]
rs = []
for arm in arms:
    p, m = f"{A}/{arm}/pair.json", f"{A}/{arm}/pair-meta.json"
    if not os.path.exists(p): sys.exit(f"{arm}: no {p} (cdhout27.sh STEPS=pair)")
    d = json.load(open(p)); meta = json.load(open(m)) if os.path.exists(m) else {}
    r = d["kld_b"] / d["kld_a"]; rs.append(r)
    print(f"{arm:6s} control {os.path.basename(meta.get('control', '?')):20s} {d['kld_a']:.6f} -> cd {d['kld_b']:.6f}: r {r:.4f} "
          f"(z {d['z']:+.2f}, se {d['se'] / d['kld_a']:.4f} rel, {d['docs']} docs, {d['positions']} positions; noop gate "
          f"{meta.get('noop', '?')}, hygiene {meta.get('hygiene', '?')})")
g = math.exp(sum(math.log(r) for r in rs) / len(rs))
v = "PASS" if len(rs) >= 2 and g <= 1.0 and all(r <= 1.01 for r in rs) else "FAIL"
print(f"CDHOUT27 VERDICT {v}: geometric mean {g:.4f} over {len(rs)} seed pairs ({', '.join(f'{r:.4f}' for r in rs)}); "
      f"rule: PASS g <= 1.000 and every r <= 1.01 (two pairs)")
