# w43 · Settling open V3 trunk-quantizer settings on the CPU proxy: gain per (family, K), damping at fractional K, tile CD price

Worker w43 (2026-11-06). Three questions that can change the frozen trunk quantizer settings before the ~200 card-hour
ladder: **#39** (global scale gain per K and family; "best factor rises with K", GPU result never logged), **#38**
re-checked at fractional K on the i8 codebook (it was settled at integer K before i8), and **#77** tile coordinate
descent priced per K (the user decides adoption from the table). Rules below were written 2026-11-06 ~1:05 AM CST,
before any data.

## Tool: `settle.py` proxy v2 (exllamav3's own transforms, scale choice and refit)

- Base as `trellis2.py`: block LDLQ + bitshift trellis (fractional K via the shift pattern, free start state: absolute
  errors 4–6% optimistic, #57; ratios unaffected), codebook **mul1_i8** (the V3 trunk codebook; i8c is likely
  dropped, v3.md §1c-bis), damping **adopted ladder values** (sigma_reg × mean diag H: 0.005 for the layer-input
  group ssm_qkv / ssm_z / attn_qg / attn_k / attn_v, 0.025 for the rest), proxy error rel = tr(dW H dW^T) /
  tr(W H W^T) under the undamped calib H_in, in the original basis.
- **Transforms = exllamav3's `regularize`** (fork eb5a2139), on one aligned 128-row block per sample (128 rows =
  one output-Hadamard block): per-output-channel RMS scales with random signs, the 128-Hadamard over output channels,
  per-input-channel RMS scales su (over the whole tensor) with random signs, the input 128-Hadamard. LDLQ runs
  under P diag(su) (H + δI) diag(su) P, as the ladder's `--scaled-hessian` (damping δ added in the original basis,
  as `finalize_capture_H`). `ffn_gate_up`: gate_proj on even seeds, up_proj on odd (exllamav3 quantizes them apart).
- **Scale = exllamav3's own choice**: its tile sample (width-3 wrapped diagonal + the extreme-RMS tiles) over the
  whole transformed tensor, capped at 2,304 tiles by a uniform subsample, × `LDLQ_DRIFT`; the `g_scale_search_batch`
  grids in its own units (its weights have column RMS `codebook_scale` = 1.2437 against a codebook of sd ~1: coarse
  0.1–1.9 step 0.2 on every 3rd tile, fine ±2 × 0.075, parabola). A multiplier m on it is exactly
  `ourq.py --gain` (and `quantize/gaintest.py`'s MULT). The old #39 sweep (`gainsweep.sh`) searched a factor on
  the per-row RMS instead, which is not what the quantizer does.
- **Per-output-channel scale refit emulated** (`refit_scales`' sv step, closed form under the damped H, per original
  row): `rel` = after it (primary), `rel_norefit` = before. The input-channel (su) refit step is not emulated (5k+
  free scales fitted on 128 rows would be in-sample).
- Prep (transformed rows, su, tile sample) is cached per (tensor, seed) in `cache/` on the box; a lock makes one job
  compute it. `settle.py selftest`: the transforms invert exactly (max error 4e-16, rel 1e-31).
- Note found while reading: `ldlq_drift(K)` is a dict lookup on {1, 1.5, 2, 2.5, 3}, so the ladder's quarter steps
  K1.25 / 1.75 / 2.25 / 2.75 get drift 1.0 (their neighbours 1.08–1.004). Item 1b measures what that costs.

**Why v2 (2026-11-06 ~1:25 AM CST, after the first ~40 jobs of the v1 tool; the rules below were not changed).**
v1 was trellis2.py's proxy (random rows, input Hadamard only, unit-RMS rows) plus the emulated g_scale and refit.
Its first data contradicted itself on the same tensor: layer-30 ssm_out wanted m = 0.85 (−24% at K6) on seed 0
and m = 1.0 on seed 1. The likely cause is outlier rows, which v1 leaves concentrated (no output Hadamard, no su)
while the quantizer spreads them, and the scale question is exactly about these tails. Under v2, the same tensor
gives m = 1.0 at K4 and K6 on all three samples (seeds 0 / 1 / 2: rel(0.85) / rel(1) = 1.07–1.13, rel(1.15) /
rel(1) = 1.07–1.17), and s_exl agrees across seeds (0.89–0.90 at K4). v1's 87 finished jobs (11-value grid 0.7–1.2) are kept
(`data/v1/`, boxes `out-v1/`) for a v1-vs-v2 comparison, and are not used for any decision.
**v2a (2026-11-06 1:41–2:19 AM CST):** the first v2 run searched g_scale on a finer grid than exllamav3's (coarse 0.1,
fine 0.025, in our units). Fixed to exllamav3's grids in its units (s moves ≤ 0.025: layer-30 ssm_out K2 0.950 →
0.961, K4 0.898 → 0.894, K7 0.850 → 0.825) and restarted clean. Its 136 jobs are kept (`data/v2a/`, boxes
`out-v2a/`), not used.

Tensors (27, three per family, spread over depth; all nine trunk families):
ssm_qkv 2 / 30 / 50 · ssm_z 10 / 34 / 62 · ssm_out 10 / 30 / 50 · attn_qg 3 / 19 / 43 · attn_k 11 / 35 / 63 ·
attn_v 7 / 43 / 59 · attn_output 19 / 43 / 63 · ffn_gate_up 10 / 43 / 62 · ffn_down 2 / 30 / 63.
Two independent samples per (tensor, K): **seed 0** (sign seeds 0, row block 0) and **seed 1** (sign seeds 1, another
row block). Per-tensor ratios are paired (same rows, signs and codes path); "gmean" = geometric mean over tensors.

Size: "≤ ~6 CPU-h per item" is read as ≤ ~6 hours of the boxes per item (the brief also asks for ~12 h of work
on ~46 single-threaded slots, which a literal 6 CPU-hours per item could not fill). Estimated cost of each item
(Thor-core-hours): item 1 ~75, item 2 ~24, item 3 ~16, item 1b ~37, item 1c ~37.

## Item 1 · #39 gain per (family, K)

Grid m ∈ {0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3} on exllamav3's g_scale, K ∈ {1.5, 2,
2.5, 3, 4, 5, 6, 7}, seeds 0 and 1. r(t, K, s, m) = rel(m) / rel(1.0).

**Rule (written before data):**
1. Pick on seed 0: for each (family, K), m* = the grid value minimising the family's gmean of r on seed 0.
2. Judge on seed 1 (held out): gain g = 1 − gmean of r(t, K, 1, m*) over the family's three tensors.
3. An entry (family, K) **qualifies** if g ≥ 1% and every tensor of the family has r(m*) ≤ 1.00 on both seeds.
4. A family gets a factor table if it qualifies at ≥ 2 of the 8 K (the lead's rule: ≥ 1% at ≥ 2 K, every tensor
   ≤ 1.00); its non-qualifying K keep 1.0. No family qualifies → **"flat: keep 1.0"**.
5. Between table K the factor is interpolated linearly in K (1.0 where an entry did not qualify); below K1.5 the
   K1.5 entry (if 1b does not set K1.25), above K7 the K7 entry; sub-1 K: 1.0 (not measured).
6. Output: `gain-table.json` ({family: {K: m}}) and, per ladder K, the `ourq.py --gain` string (suffixes: ssm_qkv
   `in_proj_qkv`, ssm_z `in_proj_z`, ssm_out `linear_attn.out_proj`, attn_qg `q_proj`, attn_k `k_proj`, attn_v
   `v_proj`, attn_output `o_proj`, ffn_gate_up `gate_proj` + `up_proj`, ffn_down `down_proj`).
7. Edge check: if m* sits on a grid end (0.6 or 1.3) for any family and K, it is reported and the grid is extended
   before the table is used.
Reported too: pooled gmeans over both seeds, `rel_norefit` (how much the refit absorbs), s_exl per K.
Runtime cost: none (quantizer only: same format and kernels). Adoption path: the 27B confirmation queue
(`gaintest.py` / a swap pair) — nothing ships on proxy evidence alone.

### Item 1b · the quarter steps K1.25 / 1.75 / 2.25 / 2.75 (drift-lookup miss)
Same grid, tensors and seeds. Same rule (3), entries added to the table if it adopts (they count towards rule 4's
"≥ 2 K"). Also reported: m* against the drift ratio of the neighbours (the miss would show as m* ≈ the missing drift).

### Item 1c · third sample (seed 2), K ∈ {1.5, 2, 2.5, 3, 4, 5, 6, 7}
Lowest priority, confirmation only: rule 3's "every tensor ≤ 1.00" is also required on seed 2 for table entries that
are adopted; it can only remove entries, never add.

## Item 2 · #38 damping at fractional K (i8)

K ∈ {1.5, 2.5, 3.5, 5.5}, seeds 0 and 1. Input group (ssm_qkv, ssm_z, attn_qg, attn_k, attn_v): 0.005 (adopted) vs
0.0025 / 0.01; rest (ssm_out, attn_output, ffn_gate_up, ffn_down): 0.025 (adopted) vs 0.0125 / 0.05. Each at m ∈
{0.9, 1.0, 1.1}; primary m = 1.0 (the quantizer's own g_scale). r = rel(d) / rel(adopted), per tensor and seed.

**Rule (written before data):** per group and alternative d, at each K: gain = 1 − gmean of r over the group's
tensors and both seeds. d replaces the adopted value if at ≥ 2 of the 4 K the gain is ≥ 1% and every tensor (both
seeds) has r ≤ 1.00. If both alternatives pass, the larger mean gain over the 4 K wins. The change applies from the
lowest to the highest passing K, extended to the menu end (K1 or K7) when that end K passes.
Caveat fixed now: H quantizes and scores (in-sample); damp3 (#38) measured that this favours *less* damping. A pass
toward **less** damping (0.0025, 0.0125) therefore gets a held-out check (split-half H on the GPU, as damp3) before
adoption; a pass toward **more** damping counts as is. Also reported: the ratios at the best of m ∈ {0.9, 1, 1.1}.
Runtime cost: none.

## Item 3 · #77 tile CD priced per K

K ∈ {1.5, 2, 3, 4, 6}, seeds 0 and 1, m = 1, adopted damping, LDLQ then 3 CD sweeps (`trellis2.ldlq_cd`).
No adoption rule (the user decides from the table). Reported per K: cd1 / cd2 / cd3 ÷ ldlq (mean, parameter-weighted
mean, tensors ≤ 1, per family; both after and before the refit), the gain in equivalent bits (from item 1's rel(K)
slope at m = 1 per family), accepted-tile shares and the CPU time ratio (ldlq + sweeps) / ldlq.
Runtime cost: none (codes only; 2–3× quant time on the card).

## Files and how to run

- `settle.py` the proxy (`SETTLE_PROXY=1`: v1) · `plan.py` job lists per box + ETA · `settleq.py` the queue runner (on a box: skips finished
  outputs, caps own jobs / ffn_down jobs / cgroup anon memory, `nice 10`, one thread a job) · `report.py` applies the
  rules above (`report.py OUTDIR...` → markdown + `gain-table.json`).
- Boxes: `/data/work/w43/` on cpu1 (colo) and cpu2 (Thor); outputs `out/<mode>/s<seed>/<layer>-<tensor>-K<k>.jsonl`,
  queue log `logs/queue.log` ("SETTLE QUEUE DONE" at the end), per-job logs in `logs/`. Launch (from that dir):
  `setsid nohup python3 /data/work/w43/settleq.py jobs-cpu2.txt --max 30 --maxdown 7 --anoncap 118` (Thor) /
  `... jobs-cpu1.txt --max 16 --maxdown 2 --anoncap 110` (colo); `touch logs/STOP` pauses (restart = same line).
- Report: copy `out/` from both boxes into `data/v2/<box>/` and run `report.py --table gain-table.json data/v2`.
- Running since 2026-11-06 2:22 AM CST (08:22 UTC): Thor 863 jobs (115 core-h est.), colo 487 (58). ETA (planner
  simulation + 15%, ffn_down runs ~20% over the model): item 1 ~6:00 AM CST, item 2 ~7:10 AM, item 3 ~7:45 AM,
  1b ~9:30 AM, 1c ~11:15 AM. Items finish in that order on both boxes (each box runs its list in priority order;
  ffn_down jobs set the tail of each item).
- Watchdog rules: `local/watch.conf` (`settle1` / `settle2`).

## Results

(none yet.) Interim observation from the superseded v2a run (finer search grid, 136 jobs, seeds incomplete; **not a
result**, recorded so the final read can be checked against it): most families pick m* = 1.0 within ±0.5%
(exllamav3's own scale is near the proxy optimum); attn_qg wanted m ≈ 1.15 at K6–7 (−6% to −17% on single tensors)
and layer-34 ssm_z m ≈ 1.1 at K6–7 (−2% to −7%), i.e. a *larger* input scale than exllamav3 picks — the opposite
direction to the old #39 RMS-factor sweep.
