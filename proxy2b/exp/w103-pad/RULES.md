# #137 · A-15 partial activation denoising, ρ sweep on the 0.8B (w103) — rule

Written 2026-11-09 (CST), **before any data** of this experiment; committed alone before the queue is registered.
Task: `local/workers/instr3/w103.md` (user go 11-09). Idea: `local/papers/runs/2026-11-08r2/ideas/ranked.md` §4 A-15;
prior: #88 (w07-denoise: level-matched full denoising with target transform F, K3 1.031 / K2 1.10, FAIL).

## Question

Does quantizing every trunk linear against **H′ = H + ρ·N** (N = second moment of the input error the layer sees in a
quantized model) beat today's metric H, and is any gain more than the extra damping that adding N's trace brings?
Metric-only (no target transform F, no C = E[xηᵀ] term): this is the partial version (ρ < 1 = λ) that #88 did not run.

## Setup

- **Model / stack:** Qwen3.5-0.8B, i8 codebook, today's ladder stack on the 0.8B: `q2b.sh` (qtf2, scaled Hessian,
  σ 0.025 / input groups 0.005) + H_out dy (`/data/yb/c18/hdiag-snap1024-dy`, σ_out 0.01, two-sided LDLQ) + swj
  (`w02-fixedscale/swiglufit.py`, fit on g08-fit1). Not included: tile CD on attn_qg (not in the 0.8B ourq; 0.5–1.2%
  on those tensors only) and writers asym (per-mix, not a ladder setting: #105 / #111).
- **N (per input group, all 96 trunk groups):** `moments.py --mode quant` (w07-denoise): η = the group's input when every
  trunk linear is the quantized stack's weight, minus the bf16 input, on **fit-pool** windows (pools-v3 `fit`, 2,048-token
  windows, all of the pool up to 2M tokens). Source quant: the same stack at **seed offset 3** (never an evaluated
  seed), one per K, after swj. N = E[ηηᵀ] (uncentred: the mean error is part of what the layer sees), rescaled to
  calib H's token count. Layer 0's input group has η = 0 (identical across arms).
- **Arms** (per K ∈ {3, 2}, seed offsets 0 and 1; `mkhp.py` builds the `--hin` dirs):
  - `r0`: ρ = 0 (calib H, today);
  - `n10 / n25 / n50`: H′ = H + ρN, ρ ∈ {0.1, 0.25, 0.5};
  - `d10 / d25 / d50` (control): H′ = H + c·diag(H) with c = ρ·tr N / tr H per group: the same added trace,
    structureless (extra damping ∝ diag H). ourq's own damping (σ × mean diag of H′) then grows identically in n and d.
- **Scoring:** **g08-search2** (170 search-pool documents, 2,048-token cap, never used for a decision: first use)
  decides; **g08-search3** (170 other search-pool documents, first use) is a same-sign confirmation only. Paired per
  seed with `pcmp.py` (per-document bootstrap SE); tails and top-1 reported (README guard).
- 28 arm quants + 2 noise-source quants, ~3 h of the 4070 (each GPU unit holds `/data/yb/gpu.lock`, < 15 min; VRAM ≤ 7.2 GB).

## Rule (decides on mean KLD)

Per K, with ratios KL(B)/KL(A) per seed and geo = geometric mean over seeds 0 and 1:

1. ρ*_K = the ρ with the lowest geo(r0 → nρ) on search2.
2. (a) geo(r0 → nρ*) ≤ **0.98** on search2;
   (b) geo(dρ* → nρ*) ≤ **0.99** on search2 (better than the equal-trace damping arm by ≥ 1%);
   (c) on search3, geo(r0 → nρ*) < 1 **and** geo(dρ* → nρ*) < 1.
- **PASS** if (a), (b) and (c) hold at **both** K3 and K2 → a 27B confirmation row (BACKLOG queue) with ρ*_K; the 27B
  needs an N collection per level (two-model layer-streamed forward), costed before any port.
- **FAIL, damping:** (a) holds at both K but (b) or (c) fails somewhere → the gain is damping (tuned in #114 / #114-03): drop.
- **FAIL:** otherwise → drop A-15.
- Not part of the rule, reported: every arm's ratios on both slices; d → r0 (if extra damping alone wins ≥ 2% at some K,
  flag it against #114, whose σ was tuned on a held-out proxy, not KL); N's trace share per group kind; per-seed z,
  KL99 / KL99.9 intervals, top-1.

## Gates

- **G0 (blocks):** `mkhp.py` at ρ = 0 (noise and damp) writes tensors equal to calib's H on every group.
- **Reproduction (information only):** r0 seed 0 on g08-search1 vs d1112's `k{K}ste_swj` (same recipe, 11-06 code);
  bit-identical expected unless the 0.8B quantizer code changed since; all arms share today's code either way.

**Runtime:** none (quantizer metric only; same format and kernels).
**Data use:** H_in calib (public, as every quant); N and swj on the fit pool; decisions on search2, confirmation on
search3; nothing from verify or public.

## Files

`mkhp.py` (H′ dirs + stats), `run.sh` (driver, idempotent; `/data/yb/w103/`), `report.py` (this rule; `--nstat`),
`queue.d-60-w103.sh` (registration), results copied to `data/` when done.
