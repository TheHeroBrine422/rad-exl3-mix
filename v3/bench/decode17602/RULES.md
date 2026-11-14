# #176-02 rules (w159) — decode speed table from e2e on the mad kernels: pre-registration

Written 2026-11-14 9:10 AM CST, before any held-out mix (m-*) is built or measured. Committed alone.

**Question.** Can the knapsack's decode term (λ_d) use per-(tensor, option) decode prices that add up to the
measured end-to-end decode step of any mix?

**Setup (fixed).** ds8 HIP 7 (R9700S, 210 W cap), radiance qc-exl3-v3 120132a (`/data/q27/rad10-e157b` install + home),
f16 decode activations (`QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0`, user default), bf16 KV, `--tp 1 --max-num-seqs 8
--max-model-len 40960`, w138's bench.py over the chat API, batch 1, greedy; ctx 0 (warm 1 + 3 × 256 tokens) and ctx 8k
(warm 1 + 2 × 128). Measure = server ms per decode step, median of a row's measured runs.

**Fit data (only these):** the 42 K ≥ 1 uniform mad packs (`packs-mad`): one stamped round (ctx 0,
`RADIANCE_AQL_STAMP=8`) for the per-tensor split, two unstamped rounds in opposite K order (ABBA) for the e2e totals;
head packs (62 R16 plans on the K1.125 pack), a/b T4 / T5 and 5 trellis-embedding packs on the K1 pack (stamped, ctx 0).
Model (`fit17602.py`): e2e_ctx = F_ctx + α_ctx · Σ stamped device time of the pack's menu tensors; prices = α_ctx ×
stamped time, per (tensor type, level) as the median over the type's layers.

**Held-out check (never used in the fit):** the 5 mixes of `mkheld.py` (seed 17602; m-h1..h5: random per-tensor levels in
2–3.5 / 3.5–5 / 5–7.5 / 1–7.5 and one per-type mix, five head plans incl. one mul1, T4 / T5, four trellis embeddings +
bf16, act a8g128 lines on four of them), two unstamped rounds in opposite order (a, b) + one stamped ctx-0 round (s, for
stamp additivity only). Predicted = F_ctx + Σ table prices of the mix's options.

**Pass rule.** |predicted − measured| / measured < 2 % for every held-out mix at ctx 0 and at ctx 8k (measured = mean
of rounds a and b). Reported beside it: each mix's error, the stamp additivity (table raw sum vs the mix's own stamps),
the uniform-fit residuals, and the a / b round spread (noise floor).

**If it fails:** the table is not handed over as passing. Any model change (e.g. per-type α, a term for act lines or the
head) is written as an amendment here first and judged on new held-out mixes (new seed), not on m-h1..h5.

**Not in scope:** sub-1 levels (w151 requant running; priced later with the same method, no new rule), int8 decode rows
(user default is f16), drafter on.

**Amendment 1 (2026-11-14 9:25 AM CST, before any held-out mix is built or measured).** ds8's host is saturated by
other workers' jobs (load ~26 / 32 threads, rad-convert / radmix / quant lanes): some runs of a row are 10–15 % slower
(spikes only add time; device-side stamps are unaffected). The e2e statistic per row is chosen on the **uniform fit
only** — the one with the smallest residual rms among (a) median of the row's measured runs, (b) minimum of the row's
runs, (c) median inter-chunk gap (bench.py field added at 9:20 AM; rows before that lack it) — and then applied
unchanged to the held-out mixes; all reported. Rows whose runs spread > 3 % may be rerun once (same list order).

**Amendment 2 (2026-11-14 11:05 AM CST, before any held-out row has run).** ds8 rebooted at ~10:08 AM CST (every
process lost). The e2e rounds were restarted on **HIP 6** (one card for all e2e, stamped, part and held-out rows; the
pre-reboot card-7 rows are kept in `res/c7/` as a cross-card check, not fitted). On HIP 6 a host-wide disturbance
10:47–11:00 AM CST made whole rows ~40 % slower with small in-row spread (card 7 was slow at the same time), which the
spread rule cannot see. Rule: a pack (uniform or held-out) whose round-a and round-b values differ by > 3 % gets a third
round (c, after round b, same card); its value is the median of the three rounds. Applied identically to uniform and
held-out packs.

**Decision under amendment 1 (2026-11-14 1:00 PM CST, uniform fit only; held-out lines were filtered out of the fit's
output and not read).** Card-6 stamps, rounds a / b / c (median of rounds): residual rms ctx 0 / 8k — median of runs
0.281 / 1.436 %, **minimum of runs 0.264 / 0.247 %**, median inter-chunk gap 0.264 / 0.254 %. The e2e statistic is the
**minimum of a row's measured runs**, for the uniform fit and the held-out check alike.

**Amendment 3 (2026-11-14 2:30 PM CST, lead note 2:11 PM: int8-decode column; written before any int8 held-out row).**
Same fit form for `QWEN35_EXL3_AQ_XB=1 EXL3I_MAD_GEMV=1` (int8 rows at decode; needs a8g128 weights, so every uniform level
is re-mixed with act lines on the 336 trunk linears): per level one unstamped e2e row (ctx 0 + 8k) and one stamped row
(ctx 0), levels in an order interleaved from both ends of K (one round: < ~2 h card budget; no ABBA), statistic = min of
runs (as decided). The `exl3i_mad_rows` pre-pass launched before a GEMV is that tensor's time. Heads, embedding and a/b
keep their f16-row prices (the head carries no act line, so its path does not change; a/b's int8 form not measured).
Held-out: the four act-line mixes h1 / h2 / h3 / h5 (same files), rounds a / b; pass rule unchanged (< 2 % each, ctx 0
and 8k). h4 has no act lines (int8 rows cannot run there) and is left out.
