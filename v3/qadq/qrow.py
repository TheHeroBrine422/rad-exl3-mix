#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qrow.py -- results.tsv of the global QAD queue (w222): one row per served read on j217.
  qrow.py add ROOT --machine M --card C --name NAME --arm ARM --ckpt CK --size S --golden G --ctldir DIR [--ctl CTLNAME] [--run RUNDIR] [--rows FILE]
  qrow.py fill|refresh ROOT --golden G --ctldir DIR  (refresh: recompute every row from the scores present, incl. the w222 columns)     (recompute the paired-vs-control columns of rows whose control read has landed since)
NAME's rows file = ROOT/scores/NAME.json.rows (or --rows). Columns: ratio of the arm's KL / gen row vs the unhealed release (DIR/ctl-<size>.json.rows) with the paired per-document
bootstrap CI (cmp214's arithmetic, 2,000 resamples, seed 0), the ratio without w217's two blow-up documents (calib ids 1893, 3820), docs better / worse, the paired ratio vs the
control read CTLNAME (same golden) and s / step (median wall of the run's main steps >= 2, RUNDIR/train.jsonl)."""
import argparse, datetime, fcntl, json, os, sys, time, zoneinfo
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp214  # noqa: E402

BLOW = {1893, 3820}
COLS = ["when", "machine", "card", "arm", "ckpt", "size", "kl_gen_row", "ratio", "ci_lo", "ci_hi", "ratio_no_blowup", "docs_better", "docs_worse", "ctl", "vs_ctl", "vs_ctl_lo", "vs_ctl_hi", "s_per_step", "name",
        "med_doc_ratio", "share_improved", "ratio_wo_blow10", "blow10_docs", "bin_rare", "bin_mid", "bin_common"]
BLOW10_X, BLOW10_MIN_ROWS, BLOW10_FLOOR = 10.0, 8, 0.01


def s_per_step(run):
    f = f"{run}/train.jsonl" if run else None
    if not f or not os.path.exists(f): return None
    w = [r["wall"] for r in map(json.loads, open(f)) if r.get("what") == "main" and r.get("step", 0) >= 2 and "wall" in r]
    return float(np.median(w)) if w else None


def extra(golden, rows, ctlrows, xc, x, n, orig):
    """w222 / D1: median per-document ratio, share of documents improved, the ratio without the documents whose KL / gen row rises > 10x vs unhealed (>= 8 gen rows, arm >= 0.01 / row),
    and the ratio by token-frequency bin (tokfreq222.py: rare / mid / common target tokens) with the paired per-document bootstrap CI"""
    ok = n > 0; nn = np.maximum(n, 1); a, u = x / nn, xc / nn
    blow = ok & (n >= BLOW10_MIN_ROWS) & (a >= BLOW10_X * u) & (a >= BLOW10_FLOOR)
    e = {"med": float(np.median(x[ok] / xc[ok])), "improved": float((x[ok] < xc[ok]).mean()), "blow10": [int(orig[i]) for i in np.where(blow)[0]],
         "wo_blow10": float(x[~blow].sum() / xc[~blow].sum()) if (~blow).any() else float("nan"), "bins": None}
    tb = os.path.join(os.path.dirname(os.path.dirname(rows)), "tokbins.npz")
    if os.path.exists(tb):
        m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool); rb = np.load(tb)["rowbin"]
        K = np.fromfile(rows, np.float32).reshape(-1, 4)[:, 0].astype(np.float64); C = np.fromfile(ctlrows, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
        sa = np.zeros((3, len(m["docs"]))); sc = np.zeros((3, len(m["docs"])))
        for i, d in enumerate(m["docs"]):
            sl = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[sl]; b = rb[sl]
            for k in range(3): w = g & (b == k); sa[k, i] = K[sl][w].sum(); sc[k, i] = C[sl][w].sum()
        e["bins"] = [cmp214.paired(sc[k], sa[k], n) for k in range(3)]
    return e


def stats(golden, ctldir, size, rows, ctlrows):
    m = json.load(open(f"{golden}/kld.json")); orig = np.array([int(d["orig"]) for d in m["docs"]]); keep = ~np.isin(orig, list(BLOW))
    xc, n = cmp214.doc_sums(golden, f"{ctldir}/ctl-{size}.json.rows"); x, nb = cmp214.doc_sums(golden, rows); assert (nb == n).all(), rows
    r = cmp214.paired(xc, x, n); r.update(extra(golden, rows, f"{ctldir}/ctl-{size}.json.rows", xc, x, n, orig)); r["kld"] = float(x.sum() / n.sum()); r["nobl"] = float(x[keep].sum() / xc[keep].sum())
    if ctlrows and os.path.exists(ctlrows):
        c, nc = cmp214.doc_sums(golden, ctlrows); assert (nc == n).all(); r["vs_ctl"] = cmp214.paired(c, x, n)
    return r


def fmt(v, p=4): return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def row_of(a_when, machine, card, arm, ck, size, name, ctl, r, sps):
    v = r.get("vs_ctl"); bn = r.get("bins")
    cell = lambda d: f"{d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    return [a_when, machine, str(card), arm, ck, size, fmt(r["kld"], 6), fmt(r["ratio"]), fmt(r["ci95"][0]), fmt(r["ci95"][1]), fmt(r["nobl"]), str(r["docs_better"]), str(r["docs_worse"]),
            ctl if ctl != "-" else "-", fmt(v["ratio"]) if v else "-", fmt(v["ci95"][0]) if v else "-", fmt(v["ci95"][1]) if v else "-", fmt(sps, 1), name,
            fmt(r["med"]), fmt(r["improved"]), fmt(r["wo_blow10"]), ",".join(map(str, r["blow10"])) or "-"] + ([cell(d) for d in bn] if bn else ["-"] * 3)


def load(path):
    if not os.path.exists(path): return []
    L = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    L = L[1:] if L and L[0][0] == "when" else L
    return [r + ["-"] * (len(COLS) - len(r)) for r in L]


def save(path, rows):
    with open(path + ".part", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows: f.write("\t".join(r) + "\n")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("add", "fill", "refresh")); ap.add_argument("root")
    ap.add_argument("--machine"); ap.add_argument("--card", default="-"); ap.add_argument("--name"); ap.add_argument("--arm"); ap.add_argument("--ckpt"); ap.add_argument("--size")
    ap.add_argument("--golden", required=True); ap.add_argument("--ctldir", required=True); ap.add_argument("--ctl", default="-"); ap.add_argument("--run"); ap.add_argument("--rows")
    a = ap.parse_args(); tsv = f"{a.root}/results.tsv"
    lk = open(f"{a.root}/results.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)
    rows = load(tsv)
    if a.cmd == "add":
        rows = [r for r in rows if r[18] != a.name]
        rf = a.rows or f"{a.root}/scores/{a.name}.json.rows"
        r = stats(a.golden, a.ctldir, a.size, rf, f"{a.root}/scores/{a.ctl}.json.rows" if a.ctl != "-" else None)
        rows.append(row_of(datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago")).strftime("%m-%d %-I:%M %p CT"), a.machine, a.card, a.arm, a.ckpt, a.size, a.name, a.ctl, r, s_per_step(a.run)))
        save(tsv, rows); print("row:", "\t".join(rows[-1]))
    elif a.cmd == "refresh":      # recompute every row's w222 columns (and the paired columns) from the scores present; rows whose rows file is missing keep what they have
        for i, r in enumerate(rows):
            rf = f"{a.root}/scores/{r[18]}.json.rows"
            if not os.path.exists(rf): continue
            cf = f"{a.root}/scores/{r[13]}.json.rows" if r[13] != "-" else None
            st = stats(a.golden, a.ctldir, r[5], rf, cf); new = row_of(r[0], r[1], r[2], r[3], r[4], r[5], r[18], r[13], st, float(r[17]) if r[17] != "-" else None)
            rows[i] = new
        save(tsv, rows); print("refreshed", len(rows))
    else:
        for i, r in enumerate(rows):
            while len(r) < len(COLS): r.append("-")
            if r[13] != "-" and r[14] == "-":
                cf = f"{a.root}/scores/{r[13]}.json.rows"
                if os.path.exists(cf):
                    st = stats(a.golden, a.ctldir, r[5], f"{a.root}/scores/{r[18]}.json.rows", cf); v = st["vs_ctl"]
                    r[14:17] = [fmt(v["ratio"]), fmt(v["ci95"][0]), fmt(v["ci95"][1])]; print("filled", r[18])
        save(tsv, rows)


if __name__ == "__main__": main()
