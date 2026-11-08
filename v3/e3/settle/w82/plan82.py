#!/usr/bin/env python3
"""plan82.py OUTDIR -- #117 1(b) job lists (../w82.md): OUTDIR/jobs-colo.txt and jobs-thor.txt, lines
"MODE CB LAYER TENSOR K SAMPLE" for q82.py. Tensors are pinned to one box (row cache); K2.5 before K4, sample 0 first.
If Thor drops out, append its lines to colo's list and rerun (finished outputs are skipped by file)."""
import os, sys

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
T = {"colo": "2:ssm_qkv 50:ssm_qkv 10:ssm_z 34:ssm_z 3:attn_qg 43:attn_qg 11:attn_k 63:attn_k 43:attn_v "
             "1:ffn_gate 26:ffn_gate 62:ffn_gate 17:ffn_up 43:ffn_up".split(),
     "thor": "30:ssm_qkv 61:ssm_qkv 21:ssm_z 62:ssm_z 19:attn_qg 35:attn_k 7:attn_v 59:attn_v "
             "10:ffn_gate 43:ffn_gate 10:ffn_up 38:ffn_up 62:ffn_up".split()}
assert len(set(T["colo"] + T["thor"])) == 27
for box, ts in T.items():
    lines = [f"rperm mul1x_i8 {t.split(':')[0]} {t.split(':')[1]} {K} {s}" for K in ("2.5", "4") for s in (0, 1) for t in ts]
    open(f"{out}/jobs-{box}.txt", "w").write("\n".join(lines) + "\n")
    print(box, len(lines))
