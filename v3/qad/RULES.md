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

### Amendment 2 — 2026-11-13 3:58 AM CST, before any H / W / L / long-slice data (lead notes 11-13: arms H, W, L, the long-context check)
Arms A, B, C, D0, D are finished and judged by the rules above (run165.log, 11-13 2:45 AM CST); nothing below changes
their rules or reruns them. Gates run before this text (tooling only, no training): head decode vs ourq's `wq`
3.7e-3 (bf16 activations, = the trunk tensors'), one 8,192-token window fwd + bwd 31 s on 2 cards (peak 19.4 / 21.8
GiB), r = 16 adapters 114.7M values (1-card peak 29.2 GiB before Adam state).

**R16 head (arm H and its control A16).** Donor: w122 / w119 `R16s003/K4` (u4, i8; box2 `/data/q27/w119/q/R16s003/K4`,
bit-identical to ds8's by w122's ident gate), the plan fit129 measured at this trunk size (K2.5 + R16 u4 0.04109 vs
V2's head 0.04234; LOG #169). Loaded as a QLinear with its own suh / svh, rows in the stored R16 order; the targets'
vocab ids are mapped to stored rows by `R16.npy` (stored row s = vocab R16[s]; equal to the lead's formula
((s>>4)&7)·31040 + (s>>7)·16 + (s&15) on all 248,320 rows). The KL needs no un-permute (its sums run over all rows).
- **A16** = arm A with this head in the student, frozen. **H** = A16 + the head's r (5,120) and c (248,320) log
  multipliers trainable, same lr / trust / schedule / data / seed.
- Container: level + the run's trunk scales; output.weight = the R16 head (H: with its healed suh / svh, f16 as stored)
  dequantized by rxq.deq (f16 rows), un-permuted to vocab order as bf16 by `v3/yaqa/headslab.py build` into a copy of the
  control's trunk pack, taken by radmix as `output.weight`; embedding etc. from V2 3.00 as before. The unhealed R16
  export of this path vs ourq's own `wq`: 67.9% bit-equal, rel 4.6e-4 (f16 rounding of the f32 W); control16 / A16 and
  H all use this path, so H vs A16 differs only in the scales. **control16** = the unhealed level + the unhealed R16 head.
- **H beats A16** by the B / C rule (H / A16: c CI below 1 and x below 1). If it does, the later arm L trains the head
  as well (lead: "every later arm includes the head"); B / D are finished and are not rerun. A16 / control16 and
  A16 / control (V2 head) are reported.
- Not run: H on V2 3.00's head (time). Embedding scales: not trained (the student embeds with the bf16 table and the
  judged mix carries V2 3.00's embedding; the V3 trellis embedding has one bf16 scale per row, trainable once the
  student embeds with the mix's own rows: scale-up).

**W (8,192-token windows).** Arm A with --len 8192: the same seeded qad order and ≥ 16-scored-rows rule (counted in the
first 8,192 tokens), documents taken in that order until A's 975,807 tokens are matched: the first **668** documents
(978,494 tokens; 863,334 scored gen rows, 306,210 of them at positions ≥ 2,048), **5 documents per step (133
steps)**, same lr / warmup / cosine / trust / patience; monitor = the same seeded 32-document search draw scored on
their first 8,192 tokens. 2 cards (32 / 32 layers). Judged as A (vs control, both parts) and **W beats A** by the
B / C rule (W / A).

**L (low-rank adapters), r = 8 and r = 16.** Base X = H if H beats A16, else A16 (decided by `lbase` from the H / A16
judgement before L starts). Every quantized trunk linear gets y += (x Aᵀ) Bᵀ on its input x: A [r, k] ~ N(0, 1/k)
(seeded per tensor), B [n, r] = 0, f32 parameters, bf16 GEMMs; trained jointly with X's scales on the same data /
order / steps; Adam (0.9, 0.99), adapter lr **1e-4** (same warmup + cosine; no trust clamp), the scales as A. L8 on 1
card, L16 on 2 (Adam state does not fit beside one card's 29.2 GiB peak).
- Scored on the **torch path** (radiance has no adapters): `qad.py --score` with the run's best state, every document
  of c and x whole, the scorer's KL on gen rows; X scored the same way (its radiance numbers stay its official ones;
  torch vs radiance for X reported as a calibration).
- Bytes: adapter values × 2 (bf16) and × 1 + 2 bytes per adapter row (int8 + an f16 row scale); extra bpw = bits /
  26,895,998,464 (the board's parameter count, `search/v3/menu.py`). **Size-adjusted ratio** = (L / X) × exp(b ·
  extra bpw), b = **1.48** (fit129h2's local slope at 2.5 bpw with priced heads, LOG #169), also shown with b = 1.6
  (V2 board). **L pays at a storage** if its size-adjusted c CI is below 1 and its size-adjusted x ratio is below 1.
  Raw ratios reported too, plus an analytic decode / prefill cost of the two extra GEMMs per linear. A format +
  kernel change: a V3.x / V4 candidate only, whatever the result.

**Long-context check (every judged arm).**
1. `posci.py` on c and x: KL by position bin 0–2k / 2–4k / 4–8k / 8k+ (c and x documents are capped at 8,192 tokens,
   so 8k+ is empty there; 4–8k has 8 / 9 documents), ratio healed / unhealed with a 95% document bootstrap CI, for
   A, B, C, D0, D vs control; A16, H vs control16; H vs A16; W vs control and vs A. L on the torch rows (L / X).
2. **agent-verify32** (verify: the 18 ≤ 32k chunks of rich + express, 275,375 gen rows; golden from gpubox
   `/data/agent32`, copied to box2; its first scored use) for **control, A and W** once (radiance, --max-model-len
   32768): pooled ratio with a chunk bootstrap (n = 2 traces: per-trace ratios reported; chunk resampling understates
   trace-level uncertainty) and the same bins.
3. **Flag:** a healed arm whose ratio CI lies wholly above 1 in any 4k+ bin (4–8k, 8k+) on any of these slices is
   flagged, whatever its pooled gain.

Ledger: W's train documents (positions up to 8,192), the monitor at 8,192, agent-verify32's check use. Cards: box2,
≤ 4 by the locks (A16 card 0, H card 1, W cards 2–3, then L8 / L16 / scoring).

### Amendment 3 — 2026-11-13 4:42 AM CST, arm W stalled at step 32, before resuming it (implementation only)
- Erratum to amendment 2's header: it was committed at 3:52 AM CST (51bef6e), not 3:58; the arms launched at 3:53 AM.
- W stalled after step 32 (state.pt at step 32, 10:13 UTC): ~28 min in layer 0's GDN causal conv (transformers'
  F.conv1d fallback -> MIOpen) on a 700-token document, its two cards near idle, MIOpen / CK compile output in the log:
  MIOpen's per-shape solver search on the many new window lengths. Killed at 4:40 AM CST.
- W resumes from its step-32 state (same data order; steps 33+ rerun) with the GDN causal conv computed by
  qmodel.shift_conv1d (K shifted multiply-adds in f32, rounded to bf16 like F.conv1d's output; `--conv shift`).
  Check (tests/convcheck.py, CPU, L 7 / 700 / 2,049 / 8,192): forward bit-identical to F.conv1d, input gradient within
  bf16 rounding (rel 3.5e-3), weight gradient rel ≤ 5e-5 (the conv weights are frozen). Arithmetic otherwise unchanged;
  the other arms keep F.conv1d. Nothing else about W changes.
