# w140 rules: extra R16 head donors at the top (mul1 K6 / 6.5 / 7.5 / 8), price-table rows, i8 vs mul1, plugin + decode

Written 2026-11-13 ~10:35 AM CST, before any quant, score, pack or timing of this work. Committed alone before launch.
Task: `local/workers/instr3/w140.md` (#162-02). Parent: `v3/yaqa/w122/RULES.md` (#162); nothing there is changed.
Machine: ds8 only (`/data/q27/heads/`, locks `/data/q27/locks/card<N>.lock`, `ram.lock`, `conv.lock`).

## Build (no decision; w122's recipe, scripts unchanged)

- `run122.sh worker CARD QUEUE` (ds8 copy `/data/q27/heads/code/v3/yaqa/w122/run122.sh`, unchanged) on the queue
  `R16s003m1 {6, 6.5, 7.5, 8} 0.03 mul1`: same ourq flags as #162 (`--only-head --save-wq --head-perm R16.npy --qtf2
  --frac-ext SRC --scaled-hessian --sigma-reg 0.025 --codebook mul1 --hout hd-R16 --hout-blocks group --sigma-reg-out
  0.03`), same seed, f16 activations, w119's tree at `/data/q27/w119` (same code stamp as w122's donors; check after build:
  `stamp.json` "stamp" equal to `R16s003m1/K7`'s). Out: `/data/q27/heads/q/R16s003m1/K{6,6.5,7.5,8}`.
- Scores: `run122.sh score CARD fit` and `... search` (score119 unchanged; scores only donors without a score file, so
  the 17 existing rows are not touched).

## Readouts (report; no gate)

- **Price rows** (fit-gen = w119 fit-gen dump, 200 fit-pool docs; as `rep122.py`): calib-weighted head-only KL,
  unweighted KL, top-1, rare-token guard KL (rows), rare-script KL, per new donor, under new keys `u6m`, `u6.5m`,
  `u7.5m`, `u8m` (`codebook: mul1`, `avg_bits` = K) in `v3/yaqa/data/w140/rep/hcurve140.json` = hcurve122.json's 17
  plans unchanged + the 4 new ones (`u7` stays the K7 mul1 row).
- **i8 vs mul1 at equal bytes (K6, K6.5; same trellis bits, same scales):** ratio mul1 / i8 of the calib-weighted
  head KL, 2,000-resample document bootstrap (seed 0, `report119.Set.ratio`), on fit-gen and on search-gen; unweighted
  and the two guards beside. Reading: **mul1 better** at that K iff the ratio's 95% upper bound < 1 on both dumps;
  **i8 better** iff the lower bound > 1 on both; otherwise **tie**.
- **Curve to u8:** the u-curve K6 → K8 (mul1) beside i8 to K6.5; per-0.5-bit KL factor. **Flag** a new donor whose
  KL or either guard is above that of the next-lower-K mul1 donor on both dumps (as E5's monotonicity flag).

## Plugin + decode (report; one check)

ds8, `rad10-perf` (5c9956c, w136's staged build), `pipeline/hosts/rad10perf/r16check.sh`'s procedure: pack = the
ladder K4 trunk (`q27v3-144dd6e1-k4-i8/final.recipe`) + one head line (`output.weight exl3i src=DONOR codebook=mul1|mul1i8
rate=K rows=r16`), `home-pack`, `--no-error`; packs `i6, m6, i6.5, m6.5, m7.5, m8` (i = R16s003 i8, m = R16s003m1).
- **Serves:** each pack loads, `ask.sh` answers sensibly (greedy text compared i8 vs mul1 at K6 / 6.5), head stored
  bytes as expected (trellis K·V·D/8 + small overhead).
- **Decode:** `dstep.sh` (C 1, short prompt and 8k prompt), the six packs interleaved per round, 3 rounds, one card;
  statistic = median over rounds. **mul1 not worse** at K iff its median ≤ i8's × 1.01 (both prompts) — the 1% is
  the round-to-round noise allowance; differences inside it are reported as equal.
- A whole-model KLD of the packs is not part of this (w136 showed pack = bf16 splice to ~1e-6).

**Runtime class:** price rows: none (same format). mul1 vs i8 head: **decode** (codebook ALU in the head GEMV; prefill
head GEMM measured only if the decode result is mixed); measured above.
