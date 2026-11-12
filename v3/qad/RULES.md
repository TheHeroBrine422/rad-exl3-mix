# #165 pilot rules: scale-only end-to-end QAD on a pass-1 uniform mix (w125, pre-registered)

Written 2026-11-12 ~3:50 PM CST, before any 27B training data. Committed alone; amendments are appended below with a
time and a reason, never edits of the text above them.

## Question
Does training only the continuous parameters of a finished V3 level (every trunk tensor's suh / svh; arm C also
q_norm / k_norm / final-norm gains) end to end against the bf16 model's stored top-512 distributions lower the mix's
KL on held-out documents, by how much, at what card cost, and does a hidden-state term (B) or trainable norms (C) add
to it? Runtime: **none** (same codes, same container layout and kernels; arm C's norms are bf16 vectors already in the
container).

## Start point
- Level: ds8 `/data/q27/ladder/q27v3-144dd6e1-k2.5-i8/final` (codes stamp 144dd6e1, rxq code 43e08d79; DONE
  `scales: null`): the K2.5 i8 trunk with C9's writer scales applied, **no swj** (the ladder had no swj scales at this
  point; #150 pending). This is "the final scales the ladder has then".
- Student: `v3/qad/qmodel.py`. Codes decoded once (int8, bit-exact to the i8 codebook's integers), trainable log
  multipliers r (suh) / c (svh), gauge as #89 heal.py (down_proj r and v_proj c fixed). Embedding, lm_head, norms and
  GDN small projections: the bf16 checkpoint's, frozen (the eval mix's head / embedding are V2 3.00's; see Judgement).
- GDN core: flash-linear-attention 0.6.0 (Triton) when `tests/gdncheck.py` passes vs transformers' torch path
  (gradient cosine ≥ 0.99 over all trainable vectors, per-document |ΔKL| ≤ max(1% of KL, 1e-5)), else the torch path.
  Checked before this commit on dad1 (4-layer truncated model, 6 qad docs; there KL is only ~2e-4 a row, so the
  relative-only criterion first drafted (1%) was replaced by the absolute floor): cosine 0.99982 (lowest vector
  0.9947), largest |ΔKL| 6.9e-6 → fla. fla is 2× faster on GDN blocks (fwd + bwd 107 vs 205 ms at 2,048 tokens).

## Data (pools-v3, never moved)
- Train: qad-pool documents, each document's first 2,048 tokens (only document-start windows match the stored golden
  rows), the published whole-calib golden (`challenge/golden/calib`, top-512 + tail, gen rows only). Seeded draw
  (seed 0) of **1,024 documents** with ≥ 16 scored rows, one epoch.
- Monitor (early stop / step pick): 32 search-pool documents (seed 1000, first 2,048 tokens, same golden). Monitor
  role in the ledger; never a pick between arms.
- No xcorp / session-trace training data in the pilot (allowed; left for the scale-up).

## Arms (same data order, same steps, same seed)
- **A** KL only: loss = scorer KL (top-512 + tail cell, `proxy2b/common.py score_rows`' formula) summed over a step's
  scored rows / rows.
- **B** A + λ · mean over the 64 blocks of ‖h_s − h_t‖² / ‖h_t‖² (student block output vs the bf16 teacher's on its
  own trajectory, teacher run alongside); λ fixed after step 1 to KL / MSE of step 1 (equal start).
- **C** A with q_norm / k_norm (16 full-attention layers) and final-norm log gains trainable as well.

- **D0** (lead note 11-12, before this commit) asym writers alone, per mix: #87's sibling-sequential target (λ 1,
  ρ 0) on this mix's own siblings (uniform mix = the level's): `rxq.sibmom` over the level (1M qad tokens, the moments
  window list), `rxq.asym denoise --lam 1 --rho 0`, the 128 writers (64 down_proj, 48 out_proj, 16 o_proj) requantized
  with the level's own job specs + hin / wpre, the rest of the level unchanged (`rxq assemble`), then C9 recomputed on
  the asym writers (`rxq.post c9 --dn`, target W F / metric G, the adopted alphas). Order per #87-08: H_out (in the
  level) → swj (none here) → moments → writers → C9.
- **D** = arm A's QAD (same data, order, steps, settings) on the D0 level.

## Fixed settings
Adam (0.9, 0.99), lr 1e-3, 8 warmup steps then cosine to the last step, 8 documents per step (128 steps), |log| ≤ 0.3,
monitor every 16 steps, early stop after 3 evaluations without a new best, the best monitor state is the result.
Cards: ds8 (or box2) by `/data/q27/locks/card*.lock`, ≤ 4 cards at once, only after the trunk ladder has finished:
A, C and D on 2 cards each (layers split 32 / 32), B on 4 (student + teacher, 16 layers a card); D0's moments and
writer jobs on 1 card. Lanes: (cards 1-2) A → C → D; (card 3) D0 moments → denoise → writers → C9, then the scoring;
B on 4 cards when the lanes are done.

## Judgement (once, after all arms)
- Container: each arm's scales applied to the level (`rxq.swj apply_level`: suh *= r, svh *= c, rounded to f16 as
  stored), packed with the level's own `final.recipe` (src replaced), mixed over V2 3.00 like a search eval (hostd's
  `pack_cmd` / `mix_cmd`); the unhealed control is the same level packed the same way. Arm C's norms need a container
  edit path; if none is ready, C is judged on its scales only and the norm part is reported from the torch monitor.
- Scored with radiance (bf16 KV) on (c) `golden-verify27-w114` (82 clean verify docs, its 2nd use; rebuilt on ds8 from
  the published calib golden with the same documents and caps) and (x) `golden-hd122-x` (130 xcorp pieces, never
  trained on). Paired per document (`mix/paired.py`), ratio healed / unhealed with a 95% document bootstrap CI.
- **A gains** if the c ratio's CI is below 1 and the x ratio is below 1 (same sign on both parts). **B or C beat A**
  only if their ratio vs A on c has a CI below 1 and the x ratio vs A is below 1; otherwise A (simpler) stands.
- **D0 pays** if D0 / unhealed has a c CI below 1 and x below 1; if not, the sequential idea is dropped for V3.1 (lead).
  **D beats A** with the B / C rule (D / A, c CI below 1, x below 1).
- swj-only: if the K2.5 level's swj scales exist when the judgement runs, the swj-only container joins the same read
  (report only); otherwise reported as not available.

- Convergence: if an arm's best monitor step is its last step, it is reported as not converged at this budget (the
  scale-up then uses more tokens); no extra steps are added to the pilot after its data.
- Ledger: the training docs (role train), the monitor docs (monitor) and the verify slice's second use (check) are
  added to `eval/pools/ledger.jsonl` with the run.

## Tooling checks done before this commit (dad1, 4-layer truncated 27B, fake golden from the bf16 4-layer model)
- Decode: QLinear in f32 vs `rxq.deq` W: max rel 2.2e-4 over 25 tensors (the f16 LUT's rounding of the i8 values;
  the student uses the exact int8 products).
- Loop: arms A / B / C, 24 steps each, monitor KL −6.4% / −6.2% / −6.3% (a 4-layer toy: says the loop works, nothing
  about the arms).
- Export: `rxq.swj apply_level` with the run's scales, stored f16 suh / svh bit-equal to the student's export
  (50 / 50), fresh student on the healed level within 4.9e-4 of the trained student's monitor KL.

## Reported
KL gain per arm (both parts, CIs), D0's card-hours (moments + denoise + writer jobs), monitor curve (steps / tokens to the best step), wall and card-hours per arm and the
projection per shipped size, per-document sign counts, max |log| reached.

## Amendments

### Amendment 1 — 2026-11-12 5:45 PM CST, before any training run (host and card layout only)
- Host: **box2** instead of ds8. box2's share of the trunk ladder ended at 5:29 PM (all 49 levels done or assigned to
  ds8's last unit); ds8 stays with the ladder and the search. The K2.5 level (final, level, specs, final.recipe) and
  both slices were copied from ds8 over the 10 GbE link (sha256 over all level shards equal) to box2 `/data/q27/w125`.
- Cards: the whole int8 student fits on one card (box2 gate: 25.35 GiB after the build, peak 28.76 of 31.86 GiB on a
  2,048-token window, 7.3 s forward + backward a window), so A, C and D run on **one card each** (A then D on card 0,
  C on card 1), B on 3 cards (0, 1, 3: student + bf16 teacher), D0 on card 2, the scoring on card 3; ≤ 4 cards. Where
  the layers sit changes no arithmetic (run-to-run GPU noise only).
- Student build: codes decoded in two phases (int8 to the host, then every layer placed at once) so the allocator does
  not fragment; same arithmetic.
