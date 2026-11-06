#!/usr/bin/env python3
"""r94bplan.py > jobs.txt -- w45's #94-rerun job list (r94bq.py lines "CB LAYER TENSOR K SEED ROWS"), in priority order:
(1) settle.py's codebooks mul1 / mul1_i8 on sample 0 at K 6.5 .. 8 (the #94 rerun proper), (2) the same on sample 1
(replication), (3) the exact exllamav3 codebooks mul1x / mul1x_i8 (cbx.py) on sample 0 (codebook check); (2) and (3)
only at K7 (the i8 ladder's top today) and K7.25 (the candidate closest to the bar), trimmed 2026-11-06 ~5:12 AM CST
once block 1's first tensors sat at x1.67 at K7.25 with a spread of 0.002 (the full blocks would cost ~10 Thor-hours of
ffn_down jobs for a foregone verdict). ffn_down first in each block (its jobs are the longest: they would set the
tail). #94's 10 tensors (#90-02's), 128 rows."""
T = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
     (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v")]
KS = ["6.5", "7", "7.25", "7.5", "7.75", "8"]
for cbs, seed, ks in ((("mul1", "mul1_i8"), 0, KS), (("mul1", "mul1_i8"), 1, ["7", "7.25"]),
                      (("mul1x", "mul1x_i8"), 0, ["7", "7.25"])):
    for l, w in sorted(T, key=lambda t: t[1] != "ffn_down"):
        for K in ks:
            for cb in cbs: print(cb, l, w, K, seed, 128)
