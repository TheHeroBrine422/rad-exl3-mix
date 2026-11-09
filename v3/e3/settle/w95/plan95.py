#!/usr/bin/env python3
"""plan95.py BOX > jobs-BOX.txt -- q87.py job lists for #126 (../w95.md), w87's 27 tensors x 2 samples. One job =
(tensor, sample): i4 and i8 at K0.125 / 0.5 / 1 at today's sigma (settle95.py). Box dirs are w87's (BOX/w87: code,
hb, scales, cache, hout-bn); outputs under w95/. Split as w87: Thor every ffn_down pair plus half the rest, colo the
other half (<= 8 GB of ours); "all" = every pair, smallest first (if Thor is gone: copy Thor's finished w95/ files to
colo, then run "all" there; finished files are skipped)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w87"))
from plan87 import TENSORS, EST, pairs  # noqa: E402


def main():
    box = sys.argv[1]   # colo | thor | all
    C = "code/v3/e3/settle"
    ENV = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache "
           "W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn")
    PY = "python3 -W ignore"
    big = lambda p: p[0] == "ffn_down"
    small = [p for p in pairs() if not big(p)]
    mine = {"colo": small[::2], "thor": [p for p in pairs() if big(p) or p not in small[::2]], "all": pairs()}[box]
    order = sorted(mine, key=lambda p: (not big(p), -EST.get(p[0], 1)) if box == "thor" else (big(p), -EST.get(p[0], 1)))
    for t, l, s in order:
        d = f"w95/out/{l}-{t}-s{s}.DONE"
        print(f"w95-{l}-{t}-s{s}|-|{EST.get(t, 1.2)}|{d}|{ENV} {PY} {C}/w95/settle95.py job {l} {t} {s} w95/out {d}")


if __name__ == "__main__":
    main()
