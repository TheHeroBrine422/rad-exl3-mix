# RULES200 — QAD noise: pull-back (AdamW toward the start) and big batches (#200)

Written 2026-11-16 1:55 AM CST before any run (user: "do it"). Why: w183 (`rad-local/research/qad-ideas-2026-11-15.md`
ideas 1–2): trained log multipliers random-walk; Adam moves no-signal scales ~lr a step whatever the batch, so a bigger
batch fixes the direction of signal scales but not the walk of no-signal ones; a pull-back toward the start caps the walk.
Student, judge, served path as #191 / #196 / #198 (2.5 incumbent of plan191; x181g + fe38 pooled gen rows; per-document
bootstrap). Trainable = the existing log multipliers (start 0). Data: #191's order (#181 mix), extended past 1M rows with
the next rows of the same stream for the 4M arms (as #181's 4M run). lr 1e-3, #191's schedule shape stretched to the
step count. Pull-back = AdamW decoupled weight decay on the log multipliers (decay toward 0 = toward the stored scales):
wd 20 ≈ 2 % of the offset a step at lr 1e-3, wd 50 ≈ 5 %. Big batch = gradient accumulation (same micro-batch, same VRAM).

Arms, in this order (each ~1.5 card-h per 1M rows):
1. **A4M**: plain Adam, 4M rows, normal batch (196 steps) — reference for the 4M arms (#181's 4M was a different student).
2. **P20-4M**: AdamW wd 20, 4M rows, normal batch.
3. **B4**: plain Adam, 4× batch, 49 steps (4M rows).
4. **P50-4M**: AdamW wd 50, 4M rows, normal batch.
5. **B4P20**: AdamW wd 20, 4× batch, 49 steps.
6. **s2, s3**: #191 arm A (1M) with data-order seeds 2 and 3; with D0 and w186's A-s1 → **S4seed** = soup of 4 seeds
   (the same 4M rows of compute as B4).
Contrasts: P20-4M, P50-4M, B4 vs A4M; A4M vs D0 (1M; the plateau: does 4M help?); B4 vs S4seed; B4P20 vs B4; best of
these vs D0. Verdict: 95 % CI wholly below 0 = better. Runtime: none (scales fold into the format).
