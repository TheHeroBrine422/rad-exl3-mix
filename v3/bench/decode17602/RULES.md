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
