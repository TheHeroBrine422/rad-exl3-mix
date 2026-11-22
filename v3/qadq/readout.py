#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout.py RESULTS.tsv [RESULTS.tsv ...] [--size S] [--arm REGEX]   -- the table across both machines' results.tsv (box2 /data/q27/queue/results.tsv, iceman /data/queue/results.tsv; the rows carry
their machine). Markdown to stdout, sorted by size, arm, checkpoint rows. Columns: ratio vs the unhealed release on j217 (paired per document, 95 % CI), ratio without w217's blow-up documents
(1893, 3820), docs better / worse, paired vs the arm's control (@ctl) read, s / step.
       readout.py --gate A.jsonl B.jsonl [N]   -- bit comparison of the first N (2) main-step records of two train.jsonl files (train_kl, gnorm, rows, n): the iceman / box2 gate."""
import json, re, sys


def ck_rows(ck):
    m = re.match(r"ckpt-([0-9.]+e[+-]\d+)", ck); return float(m.group(1)) if m else 0.0


def gate(a, b, n=2):
    ra = [r for r in map(json.loads, open(a)) if r.get("what") == "main"][:n]; rb = [r for r in map(json.loads, open(b)) if r.get("what") == "main"][:n]
    ok = True
    for x, y in zip(ra, rb):
        same = all(x.get(k) == y.get(k) for k in ("step", "rows", "train_kl", "gnorm", "by", "n"))
        ok &= same; print(f"step {x['step']}: train_kl {x['train_kl']!r} vs {y['train_kl']!r}, gnorm {x['gnorm']} vs {y['gnorm']}, rows {x['rows']} vs {y['rows']}: {'IDENTICAL' if same else 'DIFFERENT'}")
    print("GATE:", "PASS (bit-identical)" if ok and len(ra) == len(rb) == n else "differences")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--gate": return gate(a[1], a[2], int(a[3]) if len(a) > 3 else 2)
    size = a[a.index("--size") + 1] if "--size" in a else None; arm = re.compile(a[a.index("--arm") + 1]) if "--arm" in a else None
    files = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--size", "--arm"))]
    rows = []
    for f in files:
        for l in open(f):
            c = l.rstrip("\n").split("\t")
            if c and c[0] != "when" and len(c) >= 19: rows.append(c)
    rows = [r for r in rows if (size is None or r[5] == size) and (arm is None or arm.search(r[3]))]
    rows.sort(key=lambda r: (r[5], r[3], ck_rows(r[4])))
    print("| size | arm | ckpt | machine | ratio vs unhealed [95 % CI] | w/o blow-ups | docs b/w | vs control | s / step |\n|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        vc = f"{r[14]} [{r[15]}, {r[16]}] ({r[13].rsplit('-ckpt-', 1)[0]})" if r[14] != "-" else "-"
        print(f"| {r[5]} | {r[3]} | {r[4]} | {r[1]}:{r[2]} | {r[7]} [{r[8]}, {r[9]}] | {r[10]} | {r[11]}/{r[12]} | {vc} | {r[17]} |")


if __name__ == "__main__": main()
