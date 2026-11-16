# RULES198 — QAD soup: averaging independent scale-QAD runs (#198)

Written 2026-11-16 12:55 AM CST before any soup is built or scored (user: "lets try the soup idea first though since
its nearly free"). Idea and evidence: w183, `rad-local/research/qad-ideas-2026-11-15.md` idea 1 (trained scales move
like a random walk; two runs agree on a scale's sign ~52 %).
Student, trainer and judge exactly as #191 / #196 (2.5 incumbent of plan191, served path, judge sets x181g + fe38,
pooled gen rows, per-document bootstrap). All runs below exist on box2 (`/data/q27/w170/runs/`, `/data/q27/w182/`).

Soup = element-wise mean of the trained **log** multipliers of the member runs (scales folded the same way each run's
own served container folds them); everything else from the student.
Arms (eval only, no training):
- **S2**: soup of L0B0T0Q0D0 (#191 reference) + L0B0T0Q0D1 (other data mix).
- **S4**: soup of L0B0T0Q0D0, L0B0T0Q0D1, L0B0T0Q1D0, L0B0T1Q0D0 (all #191 runs without adapters / block stage;
  Q and T were null additions, so they act as alternative runs of the same recipe).
- **C2**: sign consensus of the S2 pair: the mean where both moved the same sign, 0 elsewhere (no line search).
- **N0**: #196's arm A vs #191's L0B0T0Q0D0 (same recipe and seed): the rerun noise floor, reported.
Training arm (one, ~1.5 card-h): **A-s1**: arm A with a different data-order seed (same mix, same budget); then
**S2s** = soup(L0B0T0Q0D0, A-s1), the clean same-recipe pair.
Contrasts: each soup vs its best member, and vs L0B0T0Q0D0. Verdict per contrast: 95 % CI wholly below 0 = soup better.
If S2s is better than its best member: every later V3.1 QAD arm is a soup. Runtime: none (scales fold into the format).
Data: judge sets are #191's (ledger rows exist); training data as #191 (qad pool + xcorp).

## Amendment 1 — 2026-11-16 2:15 AM CST (user: "yea add that"), before any agreement-shrinkage soup is built
Arm **SH** (eval only): agreement shrinkage over a set of member runs. Per log multiplier: mean m and unbiased variance
v across the N members; value = m · m² / (m² + v / N) (0 where m = 0). No fitted parameter.
Built on (a) the S4 members and (b) the S2s pair (D0 + A-s1) → **SH4**, **SH2s**; also on all available independent
seeds if w187's s2 / s3 exist by then (**SHseed**, optional). Contrasts: each SH vs its plain soup (S4 / S2s) and vs its
best member, as RULES198. Verdict rule unchanged.
