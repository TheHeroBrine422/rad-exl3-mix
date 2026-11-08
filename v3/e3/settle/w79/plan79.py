#!/usr/bin/env python3
"""plan79.py > jobs79.txt -- w79 (#114 K0.125 edge, ../w79.md): w77's sweep jobs at i4 K0.125 only, sigma 0.3 / 0.5 / 1.0,
for q79.py (w77's q77.py, renamed) on colo /data/work/w79. Inputs as w77: its code snapshot, H_A / H_B1 / H_B2, the
K0.125 scale searches and the transformed-row cache are read from /data/work/w77 (symlinked into /data/work/w79)."""
import sys

C = "code/v3/e3/settle/w77"
ENV = ("HB_META=/data/work/w77/meta E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib "
       "SETTLE_CACHE=cache W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 "
       "W77_LEVELS=i4:0.125")
TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
SIG = sys.argv[1:] or ["0.3", "0.5", "1.0"]
SIZE = {"ffn_down": 3, "ssm_out": 1, "attn_output": 1}
pairs = sorted([(t, l, s) for t, ls in TENSORS.items() for l in ls for s in (0, 1)], key=lambda p: -SIZE.get(p[0], 0))
for t, l, s in pairs:
    dep = f"hb/hA/FINISHED,scales/{l}-{t}-s{s}-i4-K0.125.json"
    for sg in SIG:
        o = f"out/s{s}/{l}-{t}-d{sg}.jsonl"
        print(f"sweep-{l}-{t}-s{s}-d{sg}|{dep}|{14 if t == 'ffn_down' else 3}|{o}|mkdir -p out/s{s} && {ENV} python3 -W ignore {C}/settle77.py {l} {t} {s} {sg} {o}")
