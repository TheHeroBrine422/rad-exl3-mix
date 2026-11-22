#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qrow.py -- results.tsv of the global QAD queue (w222): one row per served read on j217.
  qrow.py add ROOT --machine M --card C --name NAME --arm ARM --ckpt CK --size S --golden G --ctldir DIR [--ctl CTLNAME] [--run RUNDIR] [--rows FILE]
  qrow.py fill ROOT --golden G --ctldir DIR     (recompute the paired-vs-control columns of rows whose control read has landed since)
NAME's rows file = ROOT/scores/NAME.json.rows (or --rows). Columns: ratio of the arm's KL / gen row vs the unhealed release (DIR/ctl-<size>.json.rows) with the paired per-document
bootstrap CI (cmp214's arithmetic, 2,000 resamples, seed 0), the ratio without w217's two blow-up documents (calib ids 1893, 3820), docs better / worse, the paired ratio vs the
control read CTLNAME (same golden) and s / step (median wall of the run's main steps >= 2, RUNDIR/train.jsonl)."""
import argparse, datetime, fcntl, json, os, sys, time, zoneinfo
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp214  # noqa: E402

BLOW = {1893, 3820}
COLS = ["when", "machine", "card", "arm", "ckpt", "size", "kl_gen_row", "ratio", "ci_lo", "ci_hi", "ratio_no_blowup", "docs_better", "docs_worse", "ctl", "vs_ctl", "vs_ctl_lo", "vs_ctl_hi", "s_per_step", "name"]


def s_per_step(run):
    f = f"{run}/train.jsonl" if run else None
    if not f or not os.path.exists(f): return None
    w = [r["wall"] for r in map(json.loads, open(f)) if r.get("what") == "main" and r.get("step", 0) >= 2 and "wall" in r]
    return float(np.median(w)) if w else None


def stats(golden, ctldir, size, rows, ctlrows):
    m = json.load(open(f"{golden}/kld.json")); orig = np.array([int(d["orig"]) for d in m["docs"]]); keep = ~np.isin(orig, list(BLOW))
    xc, n = cmp214.doc_sums(golden, f"{ctldir}/ctl-{size}.json.rows"); x, nb = cmp214.doc_sums(golden, rows); assert (nb == n).all(), rows
    r = cmp214.paired(xc, x, n); r["kld"] = float(x.sum() / n.sum()); r["nobl"] = float(x[keep].sum() / xc[keep].sum())
    if ctlrows and os.path.exists(ctlrows):
        c, nc = cmp214.doc_sums(golden, ctlrows); assert (nc == n).all(); r["vs_ctl"] = cmp214.paired(c, x, n)
    return r


def fmt(v, p=4): return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def row_of(a_when, machine, card, arm, ck, size, name, ctl, r, sps):
    v = r.get("vs_ctl")
    return [a_when, machine, str(card), arm, ck, size, fmt(r["kld"], 6), fmt(r["ratio"]), fmt(r["ci95"][0]), fmt(r["ci95"][1]), fmt(r["nobl"]), str(r["docs_better"]), str(r["docs_worse"]),
            ctl if ctl != "-" else "-", fmt(v["ratio"]) if v else "-", fmt(v["ci95"][0]) if v else "-", fmt(v["ci95"][1]) if v else "-", fmt(sps, 1), name]


def load(path):
    if not os.path.exists(path): return []
    L = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    return L[1:] if L and L[0][0] == "when" else L


def save(path, rows):
    with open(path + ".part", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows: f.write("\t".join(r) + "\n")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("add", "fill")); ap.add_argument("root")
    ap.add_argument("--machine"); ap.add_argument("--card", default="-"); ap.add_argument("--name"); ap.add_argument("--arm"); ap.add_argument("--ckpt"); ap.add_argument("--size")
    ap.add_argument("--golden", required=True); ap.add_argument("--ctldir", required=True); ap.add_argument("--ctl", default="-"); ap.add_argument("--run"); ap.add_argument("--rows")
    a = ap.parse_args(); tsv = f"{a.root}/results.tsv"
    lk = open(f"{a.root}/results.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)
    rows = load(tsv)
    if a.cmd == "add":
        rows = [r for r in rows if r[-1] != a.name]
        rf = a.rows or f"{a.root}/scores/{a.name}.json.rows"
        r = stats(a.golden, a.ctldir, a.size, rf, f"{a.root}/scores/{a.ctl}.json.rows" if a.ctl != "-" else None)
        rows.append(row_of(datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago")).strftime("%m-%d %-I:%M %p CT"), a.machine, a.card, a.arm, a.ckpt, a.size, a.name, a.ctl, r, s_per_step(a.run)))
        save(tsv, rows); print("row:", "\t".join(rows[-1]))
    else:
        for i, r in enumerate(rows):
            if r[13] != "-" and r[14] == "-":
                cf = f"{a.root}/scores/{r[13]}.json.rows"
                if os.path.exists(cf):
                    st = stats(a.golden, a.ctldir, r[5], f"{a.root}/scores/{r[-1]}.json.rows", cf); v = st["vs_ctl"]
                    r[14:17] = [fmt(v["ratio"]), fmt(v["ci95"][0]), fmt(v["ci95"][1])]; print("filled", r[-1])
        save(tsv, rows)


if __name__ == "__main__": main()
