#!/usr/bin/env python3
"""plan87.py BOX > jobs-BOX.txt -- q87.py job lists (../w87.md). One job = (tensor, sample): every sigma x level,
one output file per sigma (restartable), DONE = out/LAYER-TENSOR-sS.DONE. Both boxes get every job (finished jobs are
skipped by DONE). The boxes do not share a disk, so the work is split: colo (<= 8 GB in total) takes every COLO_EVERY-th
non-ffn_down pair, Thor the rest (ffn_down first). "all" = every pair, smallest first (if Thor is gone: copy Thor's
finished out/ files to colo first, then run "all" there; finished sigma files are skipped).
The DONE path is repeated as the job's last argument, so a restarted q87.py adopts a running job (q77's alive()).
Before the sweep: the gates (G1 / G3: gate87.json; G2: selftest-*.json) on colo; every sweep job depends on them."""
import sys

TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
EST = {"ffn_down": 14, "ssm_out": 1.5, "attn_output": 1.5}  # GB peak (measured 11-08: 5120-wide 0.94, 6144-wide 1.22; ffn_down: H_A fp64 + metric build + B1 / B2 fp32)


def pairs():
    return [(t, l, s) for t, ls in TENSORS.items() for l in ls for s in (0, 1)]


def main():
    box = sys.argv[1]   # colo | thor | all [COLO_EVERY]
    C = "code/v3/e3/settle/w87"
    ENV = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache "
           "W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn")
    PY = "python3 -W ignore"
    gates = ["gate87.json", "selftest-i8-2.json", "selftest-i4-0.5.json"]
    out = []
    if box == "colo":
        out.append(f"gate87|-|2|gate87.json|{ENV} {PY} {C}/gate87.py fork/quantize.py gate87.json real")
        for cb, K in (("i8", "2"), ("i4", "0.5")):
            out.append(f"selftest-{cb}-{K}|-|2|selftest-{cb}-{K}.json|{ENV} {PY} {C}/settle87.py selftest 11 attn_k 0 {cb} {K} 0.005 selftest-{cb}-{K}.json")
    big = lambda p: p[0] == "ffn_down"
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    small = [p for p in pairs() if not big(p)]
    mine = {"colo": small[::every], "thor": [p for p in pairs() if big(p) or p not in small[::every]], "all": pairs()}[box]
    order = sorted(mine, key=lambda p: (big(p), -EST.get(p[0], 1)) if box != "thor" else (not big(p), -EST.get(p[0], 1)))
    dep = ",".join(f"gates/{g}" for g in gates)
    for t, l, s in order:
        out.append(f"w87-{l}-{t}-s{s}|{dep}|{EST.get(t, 1.2)}|out/{l}-{t}-s{s}.DONE|{ENV} {PY} {C}/settle87.py job {l} {t} {s} out out/{l}-{t}-s{s}.DONE")
    print("\n".join(out))


if __name__ == "__main__":
    main()
