#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout223.py (w223) -- gates (ii) and (iii) from the queue's B runs (box2 QROOT/runs/A8{I,F}-sp{,32,48}/train.jsonl):
  (ii)  steps 1-2 of A8I-sp vs A8F-sp (and vs A8-25's own first steps if given): train KL, gnorm, rows (relative differences)
  (iii) s / step = median wall of main steps 3..6 per run; x of --a8-int8 vs --a8 per trainable range; peak GiB
  python readout223.py RUNS_DIR [--a8-25 PATH/train.jsonl] > readout223.md"""
import argparse, json, os, statistics


def recs(p):
    return [r for r in (json.loads(l) for l in open(p) if l.strip()) if r.get("what") == "main"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--a8-25"); a = ap.parse_args()
    R = {}
    for tl in ("", "32", "48"):
        for arm in ("A8I", "A8F"):
            p = os.path.join(a.runs, f"{arm}-sp{tl}", "train.jsonl")
            if os.path.exists(p): R[f"{arm}-sp{tl}"] = recs(p)
    print("## (ii) steps 1-2: --a8-int8 (A8I-sp) vs --a8 (A8F-sp)\n")
    print("| run | step | rows | train KL | gnorm | KL rel diff vs A8F-sp | gnorm rel diff vs A8F-sp |"); print("|---|---|---|---|---|---|---|")
    ref = R.get("A8F-sp", [])
    others = [("A8F-sp", ref), ("A8I-sp", R.get("A8I-sp", []))] + ([("A8-25", recs(a.a8_25))] if a.a8_25 else [])
    for name, rr in others:
        for r in rr[:2]:
            f = next((x for x in ref if x["step"] == r["step"]), None)
            dk = f"{r['train_kl'] / f['train_kl'] - 1:+.2e}" if f else "-"; dg = f"{r['gnorm'] / f['gnorm'] - 1:+.2e}" if f else "-"
            print(f"| {name} | {r['step']} | {r['rows']} | {r['train_kl']:.8f} | {r['gnorm']:.6f} | {dk} | {dg} |")
    print("\n## (iii) s / step (median wall of main steps 3-6; box2, other cards training)\n")
    print("| trainable | --a8 s/step | --a8-int8 s/step | x | peak GiB a8 / int8 | steps |"); print("|---|---|---|---|---|---|")
    for tl, lab in (("", "full model"), ("32", "layers 32-63"), ("48", "layers 48-63")):
        f, i = R.get(f"A8F-sp{tl}"), R.get(f"A8I-sp{tl}")
        def med(rr): w = [r["wall"] for r in rr if 3 <= r["step"] <= 6]; return statistics.median(w) if w else float("nan")
        if f is None or i is None: print(f"| {lab} | {'-' if f is None else f'{med(f):.1f}'} | {'-' if i is None else f'{med(i):.1f}'} | - | - | - |"); continue
        print(f"| {lab} | {med(f):.1f} | {med(i):.1f} | {med(f) / med(i):.3f} | {max(r['peak_gib'] for r in f):.2f} / {max(r['peak_gib'] for r in i):.2f} | {len(f)} / {len(i)} |")


if __name__ == "__main__":
    main()
