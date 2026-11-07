#!/usr/bin/env python3
"""plan64.py OUTDIR -- w64's CPU job lists (../w64.md): OUTDIR/jobs-cpu1.txt (colo) and jobs-cpu2.txt (Thor), lines
"ITEM MODE CB LAYER TENSOR K SEED M" for q64.py, in priority order. Tensors are pinned to one box (prep cache).
  item0  cd, mul1x_i8, attn_qg 3 / 19 / 43, K2 / 3 / 4, samples 0 / 1 (the order check vs w50's GPU table)
  item2  gain at m = 1, codebooks mul1x, mul1x_i8, mul1x_i8c{1.15,1.33,1.5,1.75,2}, r94b's 10 tensors,
         K 7 / 6.5 first, then 7.25 / 7.5 / 8; samples 0 / 1
  item0b cd, mul1_i8 (w43's codebook), attn_qg 3 / 19 / 43, K2, samples 0 / 1: rev vs w43's fwd LDLQ error
  item5  order (#106 / #103 on the order-fixed proxy), first in each list
  item4  rs, attn_output 3 / 19 / 43 / 63, ssm_out 2 / 10 / 30 / 50 / 62, K5 / 6 first, then 4 / 7; samples 0 / 1;
         codebook as the ladder (i8 to K5.75, mul1 from K6)"""
import os, sys

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
CBS = ["mul1x", "mul1x_i8"] + [f"mul1x_i8c{a}" for a in ("1.15", "1.33", "1.5", "1.75", "2")]
I2 = {"cpu1": "10:ffn_down 19:attn_output 19:attn_qg 43:attn_v".split(),
      "cpu2": "30:ffn_down 10:ffn_gate_up 10:ssm_out 10:ssm_qkv 50:ffn_gate_up 50:ssm_qkv".split()}
I4 = {"cpu1": [f"{l}:attn_output" for l in (3, 19, 43, 63)], "cpu2": [f"{l}:ssm_out" for l in (2, 10, 30, 50, 62)]}
jobs = {"cpu1": [], "cpu2": []}
for l in (3, 19, 43):
    for K in (2, 3, 4):
        for s in (0, 1): jobs["cpu1"].append(f"item0 cd mul1x_i8 {l} attn_qg {K} {s} -")
for box in jobs:
    def i2(Ks, seeds):
        for s in seeds:
            for K in Ks:
                for t in I2[box]:
                    l, w = t.split(":")
                    for cb in CBS: jobs[box].append(f"item2 gain {cb} {l} {w} {K} {s} 1.0")
    def i4(Ks):
        for s in (0, 1):
            for K in Ks:
                for t in I4[box]:
                    l, w = t.split(":"); cb = "mul1x_i8" if K < 6 else "mul1x"
                    jobs[box].append(f"item4 rs {cb} {l} {w} {K} {s} -")
    i2((7, 6.5), (0, 1)); i4((5, 6)); i2((7.25, 7.5, 8), (0, 1)); i4((4, 7))
# item 5 (lead note 11-07 ~1:00 AM CST, high priority; put first at 07:3x UTC): #106 rule tensors and #103 down tensors,
# mode order (natural / head / free / rev in stored order), mul1_i8 as #106, K2 / 3 / 4, samples 0 / 1
I5 = {"cpu1": [(19, "attn_output"), (43, "attn_output"), (63, "attn_output"), (10, "ffn_down"), (0, "ffn_down")],
      "cpu2": [(10, "ssm_out"), (30, "ssm_out"), (50, "ssm_out"), (30, "ffn_down"), (43, "ffn_down"), (63, "ffn_down")]}
for box in jobs:
    first = [f"item5 order mul1_i8 {l} {w} {K} {s} -" for s in (0, 1) for K in (2, 3, 4) for l, w in I5[box] if w != "ffn_down"]
    first += [f"item5 order mul1_i8 {l} {w} {K} {s} -" for s in (0, 1) for K in (2, 3, 4) for l, w in I5[box] if w == "ffn_down"]
    jobs[box] = first + jobs[box]
# item 5 extension (07:5x UTC, reported only, #106's extension set, sample 0), right after item 5's rule jobs
X5 = {"cpu1": [(l, "attn_output") for l in (3, 7, 11, 15, 23, 27, 31, 35, 39, 47, 51, 55, 59)],
      "cpu2": [(l, "ssm_out") for l in (0, 2, 17, 26, 42, 46, 49, 52, 61, 62)]}
for box in jobs:
    n5 = sum(j.startswith("item5") for j in jobs[box])
    jobs[box] = jobs[box][:n5] + [f"item5x order mul1_i8 {l} {w} {K} 0 -" for K in (2, 3, 4) for l, w in X5[box]] + jobs[box][n5:]
# appended 2026-11-07 06:19 UTC: item 0 at K2 with w43's codebook (mul1_i8), for rev vs w43's fwd LDLQ error
jobs["cpu1"] += [f"item0b cd mul1_i8 {l} attn_qg 2 {s} -" for l in (3, 19, 43) for s in (0, 1)]
for box, j in jobs.items():
    with open(f"{out}/jobs-{box}.txt", "w") as f: f.write("\n".join(j) + "\n")
    print(box, len(j), "jobs")
