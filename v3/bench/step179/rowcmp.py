#!/usr/bin/env python3
"""rowcmp.py A.json B.json -- two radiance --kld-out reports of one model on one reference: byte-identical rows or not;
else per position (the .rows file: 4 f32 a position, KL first) how many differ, mean KL of each and the paired delta,
max |dKL|. Standard library only. (w152, #179)"""
import json, struct, sys
a, b = sys.argv[1], sys.argv[2]
ra, rb = open(a + ".rows", "rb").read(), open(b + ".rows", "rb").read()
ma, mb = json.load(open(a))["all"]["kld"]["mean"], json.load(open(b))["all"]["kld"]["mean"]
if ra == rb:
    print(f"IDENTICAL rows ({len(ra) // 16} positions), mean KLD {ma:.6f}"); sys.exit(0)
n = min(len(ra), len(rb)) // 16
fa = struct.unpack(f"<{4 * n}f", ra[:16 * n]); fb = struct.unpack(f"<{4 * n}f", rb[:16 * n])
d = [fb[4 * i] - fa[4 * i] for i in range(n)]
nd = sum(1 for i in range(n) if ra[16 * i:16 * i + 16] != rb[16 * i:16 * i + 16])
print(f"DIFFER: {nd} / {n} positions; mean KLD {ma:.6f} -> {mb:.6f} (delta {mb - ma:+.3g}); "
      f"paired mean dKL {sum(d) / n:+.3g}, max |dKL| {max(abs(x) for x in d):.3g}")
