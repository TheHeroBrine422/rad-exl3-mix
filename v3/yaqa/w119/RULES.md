# w119 rules: head research E1 (#154), E2 (#155), E3 (#156), E9 precision arm (#157)

Written 2026-11-12 ~4:15 AM CST, before any E1 statistic, dump, quant or score of this work. Committed alone before
launch. Plan: `local/research/head-emb-plan-2026-11-12.md` (w118) §1 E1 / E2 / E3 / E9; w118's drafted rules are kept
or tightened, never loosened. User decisions 11-12: the 17 head donors are held until E2 / E3 decide row order and
layout; the head is not bound by int8-only (precision arms below); several head options at the top sizes are fine.

Fixed with this rule (sha256): `docs119.json` (from `mkdocs119.py`, ledger + pools-v3, no data read)
be8a3f2c12997633aa975c4f710350d6375c3cadd1cb06be3ce0d2b1645cf8a3. E3 (a)'s tiering (below) was chosen by
`tiers119.py` (model on the qad / fit Fishers of `headsens.npz` only; no dump, quant or held-out row) before this commit.

**Data pools.** Fisher / H_out: qad-gen (`headsens.npz` `fisher_qad_gen`, as today). Plan choice and pricing: fit-gen
dump. Accept / reject: fresh search-gen dump (`docs119.json` set `search`: 150 search-pool documents in no
golden-search27* slice and not in w94's dump). Final check: clean-verify-gen dump (100 documents in no golden), read
once at the very end. Public: never. Dumps: `headx94.py` unchanged (bf16 forward, final-norm output), one window of
≤ 256 consecutive golden rows per document inside the first 8,192 tokens, gen rows scored.

**Head-only KL** (all of E2 / E3 / E9): headsim94's arithmetic (reference = bf16 inputs × bf16 head in f32; an option
= f16 inputs × its f16 dequantized rows, f32 accumulation), per gen row, in **vocab order** (each donor's saved `wq`
is un-permuted by its own stored order). Per-block sums (score96's d / δ decomposition) are kept per stored order so
any per-block plan of donors sharing one stored order is scored exactly without matmuls.

**Weights (every E2 / E3 / E9 number is reported unweighted and calib-weighted; decisions use calib-weighted).**
Stratum = calib `source`, with `agent-trace/*` → `agent-trace`, `pg19-long` → `pg19`, `arxiv-long` → `arxiv`; strata
with < 2 documents in a dump set are merged into one stratum `other`. Document weight = (calib gen-token share of its
stratum, over the strata present) / (the dump's gen-row share of the stratum), capped at ×5, then renormalised.
Statistic = weighted ratio of sums (Σ w·KL over rows / Σ w over rows); CIs by a 2,000-resample document bootstrap
(seed 0) with the weights fixed.

**Strata for guards.** Rare-script = documents of the 12 non-Latin-script sources (`fineweb2-hq/{arb_Arab, rus_Cyrl,
ell_Grek, fas_Arab, jpn_Jpan, cmn_Hani}`, `wikipedia/{hi, he, ko, bn, uk, th}`), KL pooled over their gen rows,
unweighted. Rare-token = gen rows whose bf16 argmax has qad-gen Fisher rank ≥ 20,000.

**Runtime class.** E1: none. E2: decode only (a fixed logit un-permutation for R16; none for N). E3: decode only (per
128-row-block K / classes, a bf16 class). E9: decode (bf16 class, int8 G128 head kernel); a8 head activations: prefill
of eval / prompt-logprob rows only (serving runs the head at M = 1–16, f16 path).

## #154 · E1 — per-document re-check of the head's calib / xcorp split

Inputs: headwm90's stored per-position KL rows (gpubox `/data/w90/hw/<arm>-{x,c}.json.rows`) of v4, h4-bf16, h4-u5,
h4-s5-3, h4-u6, h4-s6-3, v8, h8-bf16 on golden-search27b-x / -c. Pairs: v4 → h4-bf16, h4-u5 → h4-s5-3, h4-u6 → h4-s6-3,
v8 → h8-bf16. Per pair and part: token-pooled absolute ΔKL per scored position (document-bootstrap CI), per-document
ΔKL per position (median; geo mean of per-document ratios), leave-one-document-out influence, per source, the top-1%
positions' share of |ΔKL|, and every document holding > 20% of a part's |ΔKL|. Report-only reuse of a decided slice:
it decides only how later head results are read.

**Rule.**
- (a) If calib / xcorp absolute ΔKL per position is in [0.5, 2] for ≥ 3 of the 4 pairs, the head cost is
  **corpus-general**: head experiments are read on absolute ΔKL per position (both parts same sign), knap5 keeps
  absolute head-only KL, and the 11-08 note "size head changes on the calib part" is corrected.
- (b) If the ratio is > 2 for ≥ 3 pairs, head prices come from calib-composition rows only (E4).
- (c) If one document carries > 30% of a part's |ΔKL| in a pair, that pair's ratio is recomputed without it and
  (a) / (b) are read on the recomputed ratios; per-document medians reported beside them.
- Neither (a) nor (b): inconclusive; head results are reported both ways until E4.

## #155 · E2 — row storage order for two-sided head LDLQ (uniform heads) + head activation precision

**Quants** (box2, one card each, repo `quantize/ourq.py` at the commit staged, exllamav3 fork b0052f7c = 0001–0009):
today's T0 donor flags `--only-head --save-wq --qtf2 --frac-ext SRC --scaled-hessian --sigma-reg 0.025 --codebook i8
--hout HD --hout-blocks group --sigma-reg-out σ` (σ 0.1 unless stated), no mean-row, same seed; the only change
between orders is `--head-perm P` and `hout_diag = fisher_qad_gen[P]`.
- **S** (`perm.npy`, Fisher-sorted, today): K3, K4, K5, K6, K7.
- **R16** (16-id runs strided across the 1,940 blocks: block j holds runs j, j+1940, …, j+7·1940 of
  `arange(248320).view(15520, 16)`): K3, K4, K5 (+ K6 for E3's U).
- **N** (identity): K3, K4.
- **R16 at σ_out 0.03 and 0.3:** K4.
- Check (environment): the box2 S K3 / K4 donors' `proxy_err` vs gpubox's w94 T0 donors (`/data/w94/q/two0/K3,4`,
  same flags and seed); a ratio outside [0.97, 1.03] stops E2 until explained.

**Rule (fresh search-gen, calib-weighted, X ∈ {R16, N} vs S at the same K).**
- Adopt R16 (or N) as the donor order iff X / S ≤ 0.90 at K3 and at K4 with the 95% upper bound < 0.97, and
  X / S ≤ 0.95 at K5 (point estimate; N has no K5: N needs only the K3 / K4 conditions plus N / R16 ≤ 1.05 at both).
- Guards at every tested K: rare-script KL ≤ 1.10× S; rare-token KL ≤ 1.30× S.
- Prefer N (no stored permutation) if both pass and N / R16 ≤ 1.05 at K3 and K4.
- σ_out stays 0.1 unless 0.03 or 0.3 is ≥ 3% better than R16 σ 0.1 at K4 (95% upper bound < 1) with the guards met.
- If nothing passes, S stays (today's recipe).
- Reported: fit-gen and qad-gen (w94's qad dump, in-sample Fisher) ratios (optimism check), unweighted ratios,
  top-1 agreement.

**Precision arm (head activations, user 11-12).** On the S K4, S K6 and R16 K4 donors: **a8** = the head input as
the EXL3 kernel's A8 path would see it: x̃ = had128(x ⊙ suh) per 128-input group (the donor's suh, exllamav3's
hadamard), int8 round-to-nearest with scale absmax / 127 per (row, 128 inputs), mapped back exactly; vs **f16** (today).
Reported: a8 / f16 head-KL ratio with CI, absolute ΔKL per position, and its bit-equivalent on the measured uniform
curve (Δbits = log2(KL_a8 / KL_f16) / slope, slope = log2 KL per bit between the neighbouring uniform donors).
**Rule:** the head keeps f16 activations (design §1b-5) unless a8 / f16 ≤ 1.02 (95% upper bound < 1.05) at both S K4
and S K6, in which case a8 is recorded as harmless for the head (the user may run it A8 in eval / prompt-logprob rows).

## #156 · E3 — head layout at equal bytes: slab (today) vs uniform stride vs fixed tiers, + bf16 tier

**(a) Tiering, chosen before any quant** (`tiers119.py`: w118's high-rate model, factor from qad-gen Fisher, error in
the fit-gen metric, K grid {3, 3.5, 4, 4.5, 5, 5.5, 6, 7 (×1.27 i8), bf16}, best per budget, geo mean over budgets
{3.25, 4, 5, 6}): candidates (2048 / 16384 / 65536) geo 1.986e-4, **(512 / 4096 / 32768) geo 1.974e-4 = chosen**,
(8192 / 65536) 2.067e-4 (relative to one-sided uniform; U R16 uniform 0.213). **T layout:** tier 1 = qad-gen Fisher
rank 0–511, tier 2 = 512–4,095, tier 3 = 4,096–32,767, tier 4 = the rest (215,552 rows); stored tier after tier,
inside each tier vocab-id order as 16-id runs strided across the tier's blocks (R16 inside the tier), hout_diag in
that order, σ_out as E2 decides.

**(b) T donors:** K3, 3.5, 4, 4.5, 5, 5.5, 6, 7 (8 quants, E2's flags and σ_out). bf16 rows: the bf16 weights (f16
cast) at 16 bits.

**(c) Plans and comparison.** Budgets {3, 4, 5, 6} (avg bits over the head's rows). Plans are chosen on **fit-gen**
(calib-weighted), scored on **fresh search-gen**:
- **T**: every per-tier K assignment over the 8 donors (K ≤ 7), the lowest fit-gen KL within the budget.
- **U**: uniform R16 at K = budget (E2's donors + K6), or the E2-adopted order if N wins.
- **S-slab**: today's BUILD: `headslab.py plan` (HS_TRAIN=qad, HS_FLOOR=3, corrected rowerr) s{B}-3 plans on the box2
  S donors K3–K7 at 4 / 5 / 6, u3 at 3.
- **Rule:** the layout with the lowest calib-weighted search-gen head KL, geo mean over the 4 budgets, wins; a layout
  needing per-block K (T, S-slab) wins over U only if it is ≥ 10% better than U (geo ratio ≤ 0.90, 95% upper bound
  < 1). Guards per budget vs S-slab: rare-script ≤ 1.10×, rare-token ≤ 1.30×; a layout failing a guard at any budget
  cannot win.
- Also reported: the slab / uniform and tier / uniform ratios per budget, qad-gen in-sample numbers, the fit-gen
  curve every budget 2.75–7 by ⅛ (pricing input for knap5 / E4; not a decision).

**(d) Whole-model confirmation** of the winner (if not S-slab) vs S-slab: heads spliced as bf16-dequantized rows
(`headslab.py build`-style, radmix `output.weight`) into one fixed trunk at the 4.0 pick (head budget 4.375) and the
2.5 pick (3.125); plans for those budgets chosen on fit-gen as in (c). A fresh two-part slice (fresh xcorp + the 64
`search_rest` documents of `docs119.json`), read in absolute ΔKL per position per part (or as E1 decides). Pass: ΔKL
< 0 on both parts and the pooled 95% upper bound of the ratio < 1. If U wins at a head-KL ratio within 0.95–1.05 of
S-slab, (d) is a no-harm check: |pooled| ≤ 0.3%. If box2 / goldens are not available, (d) is reported as not run.

**Final check (once, at the very end, never iterated on):** the winning layout's plans at the 4 budgets vs S-slab on
the clean-verify-gen dump (head-only KL, calib-weighted), reported.

## #157 · E9 — top-size head options (precision arm)

Options, scored as above (head-only KL; **fit-gen calib-weighted = pricing**, fresh search-gen reported):
- **bf16 tier:** T plans with tier 1 and / or tier 2 at bf16 (16 bits), budgets 4–7 by ¼ and 6.5.
- **int8 G128 FWHT128** (deadcode's radiance 2.0 recipe): per 128-input group a normalised FWHT128 of the inputs,
  int8 per (row, group) with an f16 scale = 8.125 bits; arms **RTN** (scale absmax / 127) and **GPTQ** (the rotated
  calib `lm_head.input` Hessian, σ 0.01 damping, act order off, same scales), each with **f16** and **a8** activations
  (int8 per (row, 128) in the FWHT domain).
- **Full bf16** head (16 bits, KL ≈ 0 by construction: reported as the endpoint).
- KL per byte: every option reported as (bits per head weight incl. scales / su / svh, head-only KL) on fit-gen.
**Rule.** An option is offered to knap5 iff it lies on the lower convex hull of (bits, calib-weighted fit-gen head
KL) over all measured head options (U, T, S-slab plans at every budget, the E9 options). int8 G128 is offered iff its
KL is within 5% of the hull at its bits **and** the user decides to ship its kernel (reuse radiance's vs our headdec).
The bf16 tier is offered only once E6 accepts a bf16 class (until then reported with its price).

## Logging

LOG + INDEX rows #154–#157 as results land; data into `v3/yaqa/data/w119/`; ledger rows for the three dumps
(`w119-headx-{search,fit,verify}`) and the E3 (d) slice. Machines: colo (CPU), box2 (GPU, per-card locks, ram.lock
shared), gpubox card 0 only for short checks.

## Amendment 1 (2026-11-12 ~4:45 AM CST, after the lead's notes 2–4 for w119; before any E2 / E3 / E9 data)

E1 has been read (#154, rule (a)); nothing of E2 / E3 / E9 has been dumped, quantized or scored.

**#157 int8 head arms, as deadcode ships them** (lead note 3: the head line has *no* fwht128; the kernel
`logits_gemm_i8` is W8A16):
- **GPTQ (as shipped):** `output.weight` of `StillDeadcode/qwen3.8-27b-iq8r-dflash` read as is (int8 codes −127…127,
  unrotated, one bf16 scale per (row, 128), GPTQ'd by rad-convert against the calib `lm_head.input` Hessian). Taken
  as-is: no tuning of ours on it.
- **RTN:** the same grid by round-to-nearest with radiance's `rule=search` scale (`lq_grid.cpp` `search_sym`: 21
  multiples 0.70–1.10 of the absmax scale plus each one's least-squares refit, scale rounded to bf16, no importance
  weights), re-implemented in python.
- **RTN FWHT128** (the plan's first description): report only.
- Activations: **bf16** (the kernel's) is the priced arm; **a8** (int8 per (row, 128 inputs), unrotated) is
  reported. Bits per head weight: 8.125.
- Same hull rule. Also reported: KL per byte against the trunk's marginal KL per byte at the 6.0 / 6.5 / 7.0 mixes
  (knap5's λ where available), and the lowest budget at which a trellis head still beats the int8 head on the hull
  (how far down the custom donor grid must go).

**#155 a8 arm, extended (lead note 4):** a8 vs f16 is computed for every E2 uniform donor (KL cost per K, report).
The rule is tightened, not loosened: a8 is recorded as acceptable for the head only if the original condition holds
**and** it buys measurable time; no A8 head kernel exists to time here, so unless a timing is supplied the head stays
f16 (said so in the report).
