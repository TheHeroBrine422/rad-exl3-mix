#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""curve181.py EVALDIR CTL TAG [TAG ...] [--run RUNDIR] -- #181's readout for one mix: every TAG vs the control CTL and
each TAG vs the previous one (paired per document: ratio, 95% document-bootstrap CI (2,000 resamples, seed 0), docs
better / worse; healeval.cmp's arithmetic) per slice, the rule's verdict (c CI below 1 and x below 1), KL by position
bin (0-2k / 2-4k / 4-8k) vs the control, and with --run the training curve summary (running training KL EMA per corpus
at each checkpoint, monitor reads). JSON on stdout."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad"))
from healeval import cmp  # noqa: E402


def bins(ev, a, b, sl):
    A = json.load(open(os.path.join(ev, a, "result.json")))["slices"][sl]; B = json.load(open(os.path.join(ev, b, "result.json")))["slices"][sl]
    g = A["golden"]; m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    ka = np.fromfile(A["rows"], np.float32).reshape(-1, 4)[:, 0].astype(np.float64); kb = np.fromfile(B["rows"], np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    pos = np.concatenate([np.arange(d["score_from"], d["tokens"] - 1) for d in m["docs"]])
    out = {}
    for lo, hi in ((0, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30)):
        s = gen & (pos >= lo) & (pos < hi)
        if s.sum() >= 100: out[f"{lo}-{hi if hi < 1 << 30 else 'inf'}"] = {"rows": int(s.sum()), "ratio": float(kb[s].sum() / ka[s].sum())}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ev"); ap.add_argument("ctl"); ap.add_argument("tags", nargs="+"); ap.add_argument("--run")
    a = ap.parse_args()
    import contextlib, io
    def q(x, y):
        with contextlib.redirect_stdout(io.StringIO()): return cmp(a.ev, x, y)
    rep = {"ctl": a.ctl, "vs_ctl": {}, "adjacent": {}, "bins": {}}
    prev = None
    for t in a.tags:
        if not os.path.exists(os.path.join(a.ev, t, "result.json")): continue
        r = q(a.ctl, t)["slices"]; rep["vs_ctl"][t] = r
        rep["vs_ctl"][t]["gains"] = bool("c" in r and "x" in r and r["c"]["ci95"][1] < 1 and r["x"]["ratio"] < 1)
        rep["bins"][t] = {sl: bins(a.ev, a.ctl, t, sl) for sl in r if sl in ("c", "x")}
        if prev:
            r2 = q(prev, t)["slices"]; r2["gains"] = bool(r2["c"]["ci95"][1] < 1 and r2["x"]["ratio"] < 1); rep["adjacent"][f"{t}/{prev}"] = r2
        prev = t
    if a.run:
        tr = [json.loads(l) for l in open(os.path.join(a.run, "train.jsonl"))]
        main_ = [r for r in tr if r["what"] == "main"]; last = {}
        for r in main_: last[r["step"]] = r
        main_ = [last[k] for k in sorted(last)]
        rep["train"] = {"steps": len(main_), "rows": main_[-1]["rows"] if main_ else 0,
                        "ema_at": {str(r["rows"]): r["ema"] for r in main_[::max(1, len(main_) // 20)]},
                        "pos_s_median": float(np.median([r["pos_s"] for r in main_])) if main_ else None}
        rep["monitor"] = [json.loads(l) for l in open(os.path.join(a.run, "mon.jsonl"))]
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
