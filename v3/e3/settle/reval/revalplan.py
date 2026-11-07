#!/usr/bin/env python3
"""revalplan.py OUTDIR [ASSIGN_DIR] -- w46's job lists (reval.md): OUTDIR/jobs-cpu1.txt (colo) and jobs-cpu2.txt (Thor), lines
"ITEM MODE CB LAYER TENSOR K SEED M" for revalq.py, in priority order (block by block; ffn_down first inside a
block: the longest jobs). A job's output is out/MODE/CB/sSEED/LAYER-TENSOR-KK.jsonl, so a job listed twice (e.g.
mul1_i8 at K2 in items 1 and 2) runs once, at its first (higher-priority) listing; later duplicates are dropped here.
Every tensor runs on one box (its prep cache, cache/<layer>-<tensor>-s<seed>-...pt, is per box): tensors are dealt
greedily by estimated cost, Thor counted 1.4x colo's speed x its slots (ASSIGN_DIR: keep the tensor -> box map of
the job files there, e.g. when the plan is revised while the queues run). Prints the estimated core-hours per block."""
import os, sys
from collections import defaultdict

T20 = [(10, "ssm_qkv"), (10, "ssm_out"), (10, "ffn_gate_up"), (10, "ffn_down"), (30, "ffn_down"), (50, "ssm_qkv"),
       (50, "ffn_gate_up"), (19, "attn_qg"), (19, "attn_output"), (43, "attn_v"),          # #90-02's 10
       (26, "ffn_down"), (39, "ffn_down"), (6, "ssm_qkv"), (46, "ssm_z"), (39, "attn_output"), (6, "ffn_gate_up"),
       (26, "ssm_out"), (46, "ffn_gate_up"), (58, "ssm_qkv"), (39, "attn_qg")]                 # #90-03's 10 (r100.sh b)
T90 = [(l, w) for l in (2, 10, 30, 50, 62) for w in ("ssm_qkv", "ssm_z", "ssm_out", "ffn_gate_up", "ffn_down")] + \
      [(l, w) for l in (3, 19, 43, 63) for w in ("attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down")]
K9002 = ["0.125", "0.25", "0.5", "0.75", "1", "1.5", "2", "2.25", "2.5", "2.75", "3", "3.5"]
K100 = ["0.25", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6", "7"]
K90 = ["0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1", "1.25", "1.5", "2"]
M3, M1 = "0.9,1.0,1.1", "1.0"
R106 = [(19, "attn_output"), (43, "attn_output"), (63, "attn_output"), (10, "ssm_out"), (30, "ssm_out"), (50, "ssm_out")]
X106 = [(l, "attn_output") for l in range(3, 64, 4) if l not in (19, 43, 63)] + \
       [(l, "ssm_out") for l in (0, 2, 17, 26, 42, 46, 49, 52, 61, 62)]
TA = "iqrh:0xd0474b:0x2545f491"

blocks = []                                                 # (name, [(item, mode, cb, layer, w, K, seed, M)])
def blk(name, item, mode, cbs, T, Ks, seeds, M):
    blocks.append((name, [(item, mode, cb, l, w, K, s, M) for s in seeds for (l, w) in T for K in Ks for cb in cbs]))

blk("1 #90-02 i4 vs i8", "i1", "gain", ["int4grid", "mul1_i8"], T20, K9002, [0, 1], M3)
blk("2 #100 iqrh vs mul1_i8", "i2", "gain", ["iqrh", "mul1_i8"], T20, K100, [0, 1], M3)
blk("4 #106 rule", "i4", "order", ["mul1_i8"], R106, ["2", "3", "4"], [0, 1], "-")
blk("2s #100 side: mul1, tuned A", "i2s", "gain", ["mul1"], T20, K100, [0], M3)
blocks[-1][1].extend(("i2s", "gain", TA, l, w, K, 0, M3) for (l, w) in T20 for K in ["2", "3", "4", "5", "6"])
blk("4x #106 extension", "i4x", "order", ["mul1_i8"], X106, ["2", "3", "4"], [0], "-")
blk("3 #90 sub-1 curves (i4)", "i3", "gain", ["int4grid"], T90, K90, [0, 1], M1)
blk("3c #90 sub-1 curves (old codebook i8c)", "i3c", "gain", ["cubic0.02_i8"], T90, K90, [0], M1)
blk("1x exact fork codebooks", "i1x", "gain", ["int4x", "mul1x_i8"], T20[:10], ["0.5", "1", "1.5", "2", "2.5", "3"], [0], M3)
blk("1f #90-02 at the ladder's 1/8 steps K1.125-1.875 (addendum)", "i1f", "gain", ["int4grid", "mul1_i8"], T20,
    ["1.125", "1.25", "1.375", "1.625", "1.75", "1.875"], [0, 1], M3)
blk("1s third sample (items 1, 2)", "i1s", "gain", ["int4grid", "mul1_i8"], T20, K9002, [2], M3)
blocks[-1][1].extend(("i1s", "gain", cb, l, w, K, 2, M3) for (l, w) in T20 for K in K100 for cb in ("iqrh", "mul1_i8"))

blk("2s1 #100 side arms, seed 1", "i2s", "gain", ["mul1"], T20, K100, [1], M3)
blocks[-1][1].extend(("i2s", "gain", TA, l, w, K, 1, M3) for (l, w) in T20 for K in ["2", "3", "4", "5", "6"])

# added 2026-11-06 ~5:10 PM CST (after #106's o_proj pass; reported, no rule): o_proj's head order at K1.5 / 5 / 6
blk("4k #106 o_proj at K1.5 / 5 / 6 (pricing)", "i4k", "order", ["mul1_i8"], [(l, "attn_output") for l in range(3, 64, 4)],
    ["1.5", "5", "6"], [0], "-")
blocks[-1][1].extend(("i4k", "order", "mul1_i8", l, "attn_output", K, 1, "-") for l in (19, 43, 63) for K in ("1.5", "5", "6"))


def cost(j):                                                # Thor-core-seconds (w43's per-job times)
    item, mode, cb, l, w, K, s, M = j
    down, gu = w == "ffn_down", w == "ffn_gate_up"
    search = 180 if (down or gu) else 130
    per = 105 if down else 30
    n = 4 if mode == "order" else len(M.split(","))
    return search + per * n


# Revision 2026-11-06 ~12:25 PM CST, after ~2 h of running (no data looked at for this): memory allows only 5-8
# ffn_down jobs a box (~12 GB each, 12-18 min a job), so ffn_down would have run ~20 h past everything else. ffn_down
# is dropped from the optional blocks (no verdict rests on them) and #90's sub-1 curves run it on sample 0 only.
# Which box a tensor runs on is unchanged (prep cache) except 39:ffn_down's #100 jobs, moved to colo (balance).
OPTIONAL = {"i2s", "i1f", "i1s", "i3c", "i1x"}
drop = lambda j: j[4] == "ffn_down" and (j[0] in OPTIONAL or (j[0] == "i3" and j[6] != 0))
MOVE = {("i2", 39, "ffn_down"): "cpu1",
        ("i3", 3, "ffn_down"): "cpu1", ("i3", 43, "ffn_down"): "cpu1"}    # 5:10 PM CST: colo's ffn_down slots were free

seen, jobs = set(), []
for name, js in blocks:
    for j in sorted(js, key=lambda j: j[4] != "ffn_down"):
        if drop(j): continue
        key = (j[1], j[2], j[6], j[3], j[4], j[5])
        if key in seen: continue
        seen.add(key); jobs.append((name, j))
tc = defaultdict(float)
for _, j in jobs: tc[(j[3], j[4])] += cost(j)
cap = {"cpu1": 30 * 1.0, "cpu2": 38 * 1.4}                  # colo 30 slots; Thor 40 (a couple to w43's tail), faster
load = {b: 0.0 for b in cap}; box = {}
if len(sys.argv) > 2:                                       # ASSIGN_DIR: keep the tensor -> box map of earlier job files
    for b in cap:
        for line in open(os.path.join(sys.argv[2], f"jobs-{b}.txt")):
            f = line.split(); t = (int(f[3]), f[4])
            if t in tc and t not in box: box[t] = b; load[b] += tc[t]
for t in sorted((t for t in tc if t not in box), key=lambda t: -tc[t]):
    b = min(cap, key=lambda b: (load[b] + tc[t]) / cap[b]); box[t] = b; load[b] += tc[t]
out = sys.argv[1] if len(sys.argv) > 1 else "."
fh = {b: open(os.path.join(out, f"jobs-{b}.txt"), "w") for b in cap}
per = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int)
for name, j in jobs:
    b = "cpu1" if j[0] == "i4k" else MOVE.get((j[0], j[3], j[4]), box[(j[3], j[4])])   # i4k: all on colo (it had room)
    fh[b].write(" ".join(map(str, j)) + "\n"); per[name][b] += cost(j); cnt[name] += 1
for name, _ in blocks:
    print(f"{name:42s} {cnt[name]:5d} jobs  " + "  ".join(f"{b} {per[name][b] / 3600:6.1f} core-h" for b in cap))
print("total", len(jobs), "jobs;", "  ".join(f"{b} {load[b] / 3600:.0f} core-h = {load[b] / 3600 / cap[b]:.1f} h wall" for b in cap))
