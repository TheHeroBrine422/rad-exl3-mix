# #173 calibration rows — proposal for the lead (w133, 2026-11-13; NOT run: needs the lead's approval)

The final re-solve (fin133.py, M1 on the full 49-level ladder + w135's embeddings + 17 R16 heads) is only as good as
the rows behind each option. Where M1 has rows: trunk K1.25–K7 (search 632 results + fit129's 20 uniform levels +
48 group swaps), heads u3–u5 (TH rows) + every head the search tried, embeddings base / e2 / e4 / e5 / e6 / bf16
(w135's 8 rows). Where it has none, by the re-solve's own checks:

1. **Sub-1 trunk levels (K0.125–K1.125).** M1's uniform-level curve is smooth and monotone K1.25–K7.5 (the eighth steps
   interpolate cleanly), but below K1.25 it breaks: uniform K1 ≈ K1.125 (non-monotone), K0.5 → 0 (compounding
   overflow; w120 saw the same overshoot with kfit2). Sizes 1.0 / 1.25 cannot be solved, and the full-ladder picks at
   1.5–2.0 put 5–15 % of the tensors at K0.5–K0.875 on these broken prices. → the trusted picks use K ≥ 1.25 only.
2. **A8.** Every row is act f16; the finals are A8. w132's 4 incumbent pairs (s171b): A8 − f16 log r within ~1 SE on all 8
   part-cases (mean +0.009), A8 level +1 % (2.5) … +6 % c (5–6 bpw): an additive floor that slightly compresses ratios at
   high sizes. n = 4: enough to say "transfers within noise", not enough to fit the floor per level.
3. **K7.5** (size 7.0): no row; M1 prices it ≈ K7 (−0.3 %), so the 7.0 pick barely uses it. Low priority.

## Design (A8 containers via w132's mix-writer patch; calib fit-pool slice fit129 = w129's, so the new rows share
## documents with the 90 existing f16 rows; never verify)

| block | rows | mixes | why |
|---|---|---|---|
| A1 sub-1 uniform | K0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25 (head u1.5, emb e2: the 1.0–1.25 picks' head / embedding) | 7 | the level curve below K1.25: refit the compounding (or switch on kfit3's saturation `sat`) |
| A2 sub-1 group swaps | base uniform K1; each of 6 groups × 2 depth halves → K0.625 | 12 | per-group sub-1 sensitivity (what the knapsack trades at 1.0–1.25) |
| A3 sub-1 checks | the flagged full-ladder picks at 1.0 / 1.25 / 1.5 / 2.0 after A1–A2 are refitted (second wave) | 4 | does the refit's pick beat the trusted one |
| C1 A8 levels | A8 rescore of w129's 20 uniform rows (K1.25–K7) | 20 | per-level A8 / f16 ratio on the same documents: fits the A8 mode term (additive floor) |
| C2 A8 swaps | A8 rescore of 24 of w129's 48 group swaps (donor K−1 side, both bases) | 24 | do group-swap ratios transfer to A8 (paired with their f16 rows) |
| B top | uniform K7.5 + K7 → K7.5 on ffn_gu / ffn_down (head u7, emb e5) | 3 | the 7.0 size's top option |
| **total** | | **70** | |

Cost: fit129 is 128k scored positions ≈ 3 min an eval on an R9700 (w129; A8 a little faster) → **≈ 3.5 card-h**,
≈ 30 min on ds8's 8 cards; radmix ≈ 1 min a mix under conv.lock (one at a time) → **≈ 70 min wall, mix-bound** (the
second wave A3 after a 2-min refit). Run through the coordinator (`rad-local/multi/v3`, HOLD lifted for these units
only, or an own coord.json), rows to obs JSONL (label `fit129-a8`, mode a8; A1–A3 / B also mode a8), refit M1 with
the `mode` term (m133 ext "mode": P(1 + a) + c per act mode), re-solve.

**Cheaper alternative (no new card time for the picks):** add the 13 trusted picks (`picksT/pickT133-<size>.json`) to
w132's search-value check (init vs current incumbent, A8, on a fresh two-part slice) as a third arm: 13 more evals
(≈ 13 × 5.5 min ≈ 1.2 card-h on a 224k-position slice) answer "does M1's re-solve beat the search's incumbents" on
data neither model saw. I recommend this first; blocks A / C only if the sizes 1.0 / 1.25 are wanted.
