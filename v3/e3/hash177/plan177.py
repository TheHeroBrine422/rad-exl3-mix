#!/usr/bin/env python3
"""plan177.py SEEDS KS GROUP ARMS CB [--down-last] -- q177.py job lines over w43's 27 tensors (#113's set) for one codebook
and arm list, K-major. (w139, #177)"""
import sys
T = [(2, "ssm_qkv"), (30, "ssm_qkv"), (50, "ssm_qkv"), (10, "ssm_z"), (34, "ssm_z"), (62, "ssm_z"), (10, "ssm_out"),
     (30, "ssm_out"), (50, "ssm_out"), (3, "attn_qg"), (19, "attn_qg"), (43, "attn_qg"), (11, "attn_k"), (35, "attn_k"),
     (63, "attn_k"), (7, "attn_v"), (43, "attn_v"), (59, "attn_v"), (19, "attn_output"), (43, "attn_output"),
     (63, "attn_output"), (10, "ffn_gate_up"), (43, "ffn_gate_up"), (62, "ffn_gate_up"), (2, "ffn_down"), (30, "ffn_down"),
     (63, "ffn_down")]
seeds, ks, grp, arms, cb = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3], sys.argv[4], sys.argv[5]
for s in seeds:
    for k in ks:
        for l, w in sorted(T, key=lambda t: t[1] == "ffn_down"):
            print(l, w, k, s, grp, arms, cb)
