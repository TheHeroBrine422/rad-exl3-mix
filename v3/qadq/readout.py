#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout.py RESULTS.tsv [RESULTS.tsv ...] [--size S] [--arm REGEX]   -- the table across both machines' results.tsv (box2 /data/q27/queue/results.tsv, iceman /data/queue/results.tsv; the rows carry
their machine). Markdown to stdout, sorted by size, arm, checkpoint rows. Columns (w222 additions: median per-document ratio, share of documents improved, ratio without the documents whose KL / row rose > 10x vs unhealed, ratio by token-frequency bin of the target token): ratio vs the unhealed release on j217 (paired per document, 95 % CI), ratio without w217's blow-up documents
(1893, 3820), docs better / worse, paired vs the arm's control (@ctl) read, s / step.
       readout.py --gate A.jsonl B.jsonl [N]   -- bit comparison of the first N (2) main-step records of two train.jsonl files (train_kl, gnorm, rows, n): the iceman / box2 gate."""
import json, re, sys


def ck_rows(ck):
    m = re.match(r"ckpt-(?:sd-)?([0-9.]+e[+-]\d+)", ck); return float(m.group(1)) if m else 0.0


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
            if c and c[0] != "when" and len(c) >= 19: rows.append(c + ["-"] * (26 - len(c)))
    rows = [r for r in rows if (size is None or r[5] == size) and (arm is None or arm.search(r[3]))]
    rows.sort(key=lambda r: (r[5], r[3], ck_rows(r[4])))
    print("| size | arm | ckpt | machine | ratio vs unhealed [95 % CI] | w/o 1893, 3820 | w/o own > 10x docs (ids) | median doc ratio | docs improved (b/w) | vs control | s / step | rare tokens | mid tokens | common tokens |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        vc = f"{r[14]} [{r[15]}, {r[16]}] ({r[13].rsplit('-ckpt-', 1)[0]})" if r[14] != "-" else "-"
        sh = f"{float(r[20]) * 100:.0f} %" if r[20] != "-" else "-"
        ckl = r[4].replace("ckpt-sd-", "short-decay ckpt-") if "ckpt-sd-" in r[4] else r[4]
        print(f"| {r[5]} | {r[3]} | {ckl} | {r[1]}:{r[2]} | {r[7]} [{r[8]}, {r[9]}] | {r[10]} | {r[21]} ({r[22]}) | {r[19]} | {sh} ({r[11]}/{r[12]}) | {vc} | {r[17]} | {r[23]} | {r[24]} | {r[25]} |")


if __name__ == "__main__": main()
