# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""S1 level pass: the diagonal output Hessian H_out = sum dy^2 (#33-27 / #33-28; adopted at K >= 4, 11-07) and its
block-normalized estimator. Two steps:

1. collection: rad-exl3-mix v3/yaqa/collect.py, run IN PLACE (not ported): the bf16 27B forward + backward on the qad
   pool's 2048-token windows with labels sampled from the model, per trunk linear sum dy^2 over every position
   (YAQA_DIAG=1 with YAQA_DIAG_EXTRA=1: the "-dy" sums), fp32 activations, snapshots every 128 sequences. It is a
   two-card model-parallel backward written against CUDA / ROCm devices (accelerate dispatch, pinned host buffers, a
   Triton fp32 path): it cannot run on a CPU, so no CPU bit-identity check of a port is possible, and the 27B
   collection it made (1,920 sequences, gpubox /data/yaqa/hdiag) is the one V3 uses. collect() builds its production
   environment from v3/yaqa/hdiag27.sh (the run that made it) and calls it; port it when a second model needs one.
2. estimator: rxq.stats.hout_blocks (ported, bit-identical to v3/yaqa/hconv/mkhalf.py): the snapshots' 128-sequence
   blocks, each scaled to the same per-tensor total, averaged -> OUT/all/<key>.safetensors "hout_diag" (the
   inputs.hout.dir a quant job reads).

python -m rxq.hout collect OUTDIR NSEQ [--snap 128] [--rxm RXM]   (both cards; the caller holds the card locks)
python -m rxq.hout estimate OUTDIR DST [--est dy]                  (= rxq.stats hout OUTDIR DST dy)"""
import argparse, os, subprocess, sys

from . import stats


def collect_env(nseq, snap=128, seed=0):
    """hdiag27.sh's environment for collect.py (the 27B bn collection, #33-13 / H27)"""
    snaps = ",".join(str(n) for n in range(snap, nseq, snap)) + f",{nseq}"
    return {"YAQA_GENMASK": "0", "YAQA_SHORT": "0", "YAQA_FP32": "1", "YAQA_DIAG": "1", "YAQA_DIAG_EXTRA": "1",
            "YAQA_SEED": str(seed), "YAQA_PREC": "fp32", "TRITON_CACHE_AUTOTUNING": "1", "YAQA_SNAP": snaps,
            "YAQA_SAVE_EVERY": "25"}


def collect(out, nseq, rxm=None, snap=128, seqlen=2048, python=None):
    from .engine import rxm_root
    rxm = rxm or rxm_root()
    env = dict(os.environ, **collect_env(nseq, snap))
    return subprocess.call([python or sys.executable, "-I", f"{rxm}/v3/yaqa/collect.py", out, str(nseq), str(seqlen)],
                           env=env, cwd=f"{rxm}/v3/yaqa")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.hout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("out"); c.add_argument("nseq", type=int); c.add_argument("--snap", type=int, default=128)
    c.add_argument("--rxm")
    e = sub.add_parser("estimate"); e.add_argument("out"); e.add_argument("dst"); e.add_argument("--est", default="dy")
    a = ap.parse_args(argv)
    if a.cmd == "collect": return collect(a.out, a.nseq, a.rxm, a.snap)
    ns, ne, _ = stats.hout_blocks(a.out, a.dst, a.est, norm=True, split=False)
    print(f"H_out (bn, {a.est}): snapshots {ns}, {ne} sequences -> {a.dst}/all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
