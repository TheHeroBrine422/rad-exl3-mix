#!/usr/bin/env python3
"""cd27verdict.py A ARM [ARM ...] -- the CD27.md rule over the seed pairs cd27.sh wrote (A/<ARM>/pair.json: control ->
<ARM>cd on golden-search27, tools/paired.py: kld_a = control, kld_b = tile-CD arm, bootstrap SE over documents).
  r_i = kld_b / kld_a per pair; g = exp(mean log r_i)
  PASS  g <= 1.000 and every r_i <= 1.01   -> "does no harm": cd3 on attn_qg is adopted for the ladder
  FAIL  otherwise                          -> not adopted (BACKLOG, with the pair data)"""
import json, math, os, sys

A, arms = sys.argv[1], sys.argv[2:]
rs = []
for arm in arms:
    p, m = f"{A}/{arm}/pair.json", f"{A}/{arm}/pair-meta.json"
    if not os.path.exists(p): sys.exit(f"{arm}: no {p} (cd27.sh STEPS=pair)")
    d = json.load(open(p)); meta = json.load(open(m)) if os.path.exists(m) else {}
    r = d["kld_b"] / d["kld_a"]; rs.append(r)
    print(f"{arm:6s} control {os.path.basename(meta.get('control', '?')):18s} {d['kld_a']:.6f} -> cd {d['kld_b']:.6f}: r {r:.4f} "
          f"(z {d['z']:+.2f}, se {d['se'] / d['kld_a']:.4f} rel, {d['docs']} docs, {d['positions']} positions; noop gate {meta.get('noop', '?')})")
g = math.exp(sum(math.log(r) for r in rs) / len(rs))
v = "PASS" if len(rs) >= 2 and g <= 1.0 and all(r <= 1.01 for r in rs) else "FAIL"
print(f"CD27 VERDICT {v}: geometric mean {g:.4f} over {len(rs)} seed pairs ({', '.join(f'{r:.4f}' for r in rs)}); "
      f"rule: PASS g <= 1.000 and every r <= 1.01 (two pairs)")
