#!/usr/bin/env python3
"""plan93.py BOX [EVERY] > jobs-BOX.txt -- q93.py (= w87's q87.py) job lists for settle93 (../w93.md item 2).
One job = (tensor, sample): every gain x level, one file per level (restartable), DONE = out/LAYER-TENSOR-sS.DONE.
colo (<= 8 GB of ours) takes every EVERY-th non-ffn_down pair; Thor the rest, ffn_down first (H_A + B1 / B2 of a
17408-wide group: ~14 GB). "all" = every pair (if Thor is gone: copy its finished out/ files to colo, then run "all"
there with --maxbig 0 for the small ones; ffn_down needs a box with ~14 GB free)."""
import sys

TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
EST = {"ffn_down": 14, "ssm_out": 1.5, "attn_output": 1.5}


def main():
    box = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ENV = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache "
           "W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn")
    PY = "python3 -W ignore code/v3/e3/settle/w93/settle93.py"
    pairs = [(t, l, s) for t, ls in TENSORS.items() for l in ls for s in (0, 1)]
    big = lambda p: p[0] == "ffn_down"
    small = [p for p in pairs if not big(p)]
    mine = {"colo": small[::every], "thor": [p for p in pairs if big(p) or p not in small[::every]], "all": pairs}[box]
    order = sorted(mine, key=lambda p: (not big(p)) if box == "thor" else big(p))
    for t, l, s in order:
        print(f"w93-{l}-{t}-s{s}|selftest93.json|{EST.get(t, 1.2)}|out/{l}-{t}-s{s}.DONE|{ENV} {PY} job {l} {t} {s} out out/{l}-{t}-s{s}.DONE")


if __name__ == "__main__":
    main()
