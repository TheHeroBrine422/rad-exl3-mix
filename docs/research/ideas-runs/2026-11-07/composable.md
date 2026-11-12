# Composable lens: ideas for the V3 freeze week (2026-11-07)

I generated 86 ideas in 11 groups. Every one composes with the per-level library and knap4, or is a one-pass final step on 2–3 shipped mixes. All run on our hardware in under two days. One distinction matters more than usual this week:

- **Post-ladder compatible:** fixed-code fits (scale or gain changes on frozen codes). They do not have to precede the freeze. They can run per level after the ladder, the way swj does.
- **Must precede the freeze:** anything that changes the codes or the job spec.

Each idea is tagged with one of these.

**Exchange rate used for gains.** At 2.5 bpw, d ln KL / d bpw ≈ −1.6 to −1.8 (board 2.0 → 2.5 → 3.0: 0.164 → 0.0666 → 0.0306). So **0.01 bpw freed and re-spent by the knapsack ≈ −1.7 % KL at 2.5**, and 0.001 bpw ≈ −0.17 %.

**Shrink priors** follow the brief's hypothesis:
- coherent, per-channel fixes: 27B gain ≈ 1–2× the 0.8B gain at the matching K (swj grew from 0.8B K3 −3 % to 27B K2.5 −6.5 %);
- noise-shaping fixes: ⅓ of the 0.8B gain.

## Checks I ran (colo, read-only, results in `/data/work/ideas-composable/`)

These numbers reorder the list. They are diagnostics, not results.

| check | number | consequence |
|---|---|---|
| Layer-0 residual-stream gradient diagonal (Σdy², bn, at layer-0 `out_proj` / `down_proj` outputs). Proxy for an output metric on the embedding | ½log₂(AM/GM) = **0.037 bit** (whole vector), 0.036 bit inside 128-blocks | A *diagonal*-metric embedding LDLQ (previous lens G6 / E7) is capped at ≈0.0017 bpw. Killed (K1) |
| Same statistic at every writer output (128 writers) | mean **0.016 bit** per 128-block; L63 0.010 | Output weighting cannot help writers. Agrees with #82's 3 % writer share |
| GDN `in_proj_qkv` row importance split q / k / v | part-level allocation ceiling **0.043 bit** (L62 0.19) | Per-part K for GDN qkv is not worth a format change. Killed (K3) |
| Same, per 128-row head | **0.283 bit** for qkv, 0.122 bit for z | Matches #96's per-head remainder. Kept as D7, but only with a splice test first |
| Attention `q_proj` q vs gate halves | ≤ 0.125 bit, mean ≈ 0.05 | Killed (K4) |
| `gate_proj` vs `up_proj` importance | split gain **0.005 bit** mean. Exception: L63 0.127 bit | Separate gate / up K is dead (K2) |
| MLP intermediate-channel importance (gate + up Σdy²) by layer | L0 0.20, L62 0.33, **L63 1.00 bit**. At L63 the top 1 % of channels hold **35.5 %** of the importance; best 2-class split 0.62 bit | L63 is a spike tensor. See D5 |
| Embedding / head rows | 0 exact duplicates. 1,631 embedding rows (0.66 %) below 0.3× the median norm | No free "dead row" bytes (K5) |
| Bytes outside the trellis codes | GDN `in_proj_a` / `in_proj_b` are 48×5120 each × 48 layers = **23.6 M params in bf16 = 0.0140 bpw**. conv1d is 2.0 M (0.0012 bpw); norms are negligible | D1: about −1.2 to −1.8 % KL at 2.5 if they quantize for free |
| #112 C9 screen, read from `w64/report-4070.md` | α* is **+3 (grid edge) for `gdn_out`, −1 or 0 for `attn_o` and `down`**. Down α = +1…+3 is *worse* (K2 up to 1.035) | "LS shrink is the problem" holds only on the GDN branch. The likely source is attenuation through `silu(z)`, which motivates A1 / A2 |

---

## A. Coherent-error fits on frozen codes, bf16 inputs, per level (post-ladder compatible)

**A1 · swz: GDN z-gate refit through SiLU (the swj analogue), re-scoped to the 27B stack**
- **Mechanism.** The GDN output is `RMSNorm(core) · silu(z)`. The RMSNorm removes any gain error in the core (q and k are l2-normed, and v gain is cancelled per head). So the only coherent gain pathway left in a GDN layer is `in_proj_z`. LS refit shrinks z rows, so `silu(c·z)` is attenuated on every token.
- **Fit.** Per z-channel scalar `c_z` on frozen codes, minimizing Σ_t (silu(c_z ẑ_t) − silu(z_t))² on bf16 inputs. Use the pure form without the core term, so it does not depend on the sibling qkv level.
- **Source.** #108 arm `gz`. The screen passed: K3 0.9905, K2 0.9666. It faded only in the 0.8B full stack (H_out + swj + asym: K3 0.999, K2 ≈ 0.986). Swj supplies the method.
- **What's different.** The 27B stack at ≤ K3.5 has *no H_out* (not adopted). Per-level asym writers fail in mixes (#111). So the overlap that killed #108 is mostly absent from the stack we ship at 2.5.
- **Fit tag:** per level.
- **Test.** 0.8B fixed code in the 27B-relevant stack (trunk + swj + plain writers, ± C9), seeds 0–2, K2 / K3: ~1.5 h on the 4070. Then a 27B fixed-code pair at K2.5: the GDN staged forward (~1 card-h per two levels) plus one eval pair (~1 card-h). Pass: geo ≤ 0.985 at K2.5.
- **Expected:** −1 to −3 % at 2.5, P ≈ 0.5. Risk: it overlaps C9's `gdn_out` α (see A2).
- **Conflicts:** #108 (different stack); #112 (same junction, writer side).

**A2 · Source-vs-sink attribution: does swz absorb C9's `gdn_out` α = 3?**
- **Mechanism.** Hypothesis: C9's α = 3 inflation of `gdn_out` compensates attenuation upstream through `silu(z)`. A sink-side fix like that is tied to the z level it was fitted against. Arms: {C9 only, swz only, swz then C9 refit, neither} × (writer K = z K, writer K ≠ z K).
- **Source:** #112 numbers above; new.
- **Why.** If swz takes the gain, the writer α should fall toward 0–1 and become mix-robust. If not, C9 needs conditional variants (B5).
- **Fit tag:** per level, deciding the rollout form.
- **Test.** 0.8B fixed code, cells (Kw, Kz) ∈ {(3, 3), (3, 2.5), (3, 3.5), (2, 2.5)}, seeds 0–2, about 2 h on the 4070.
- **Gain:** protects the −1 to −4 % from #112 in real mixes. Risk: low.
- **Conflicts:** #111 (same mixed-K design, for asym).

**A3 · C9 α with depth knots**
- **Mechanism.** One α per (family, depth third) instead of per family. The C9 rule `c_n *= (1 − e_n)^(−α/2)` is unchanged.
- **Source:** #85 / #112; healing results (#89: all-layer healing ≫ last-quarter healing) suggest the depth dependence is real.
- **Why.** A single α at the grid edge (3) for `gdn_out` may average a large late-layer optimum with a small early one.
- **Fit tag:** per level.
- **Test.** 0.8B fixed-code screen, 3 families × 3 thirds × α ∈ {−1, 0, 1, 2, 3, 5}: minutes per arm, ~1 h total. Rides on w75's α ∈ {3…6} run.
- **Expected:** −0.2 to −0.8 % on top of C9, P 0.35.
- **Conflicts:** #112 (extension).

**A4 · Smooth α(K) for the 71-level ladder**
- **Mechanism.** Fit the C9 α* (and swz grid settings) at K2, K3, K4 and K5. Interpolate α over K for the fractional and i4 levels instead of picking α per level by KL.
- **Source:** new rollout rule.
- **Why.** Per-level KL picks over 71 levels cost about 70 eval pairs. Swj's gain is smooth in K (−9 % K2, −6.5 % K2.5, −1 % K4).
- **Fit tag:** per level.
- **Test.** At 0.8B, check that interpolated α at K2.5 matches the directly picked α within 0.3 % KL.
- **Gain:** enabler. Risk: low.
- **Conflicts:** none.

**A5 · C9 α on top of swj's gate_up (a clean redo of #85-02)**
- **Mechanism.** Swj's `c_u` is still an LS fit per channel through the SiLU product, so it keeps a shrink. Add a per-level α on the channel's own product-residual error.
- **Source:** #85-02 (its "α* on swj" arm was broken: K3 0.122 vs 0.099, mis-ordered stack).
- **Why.** It is the last untested coherent knob on the largest family (MLP ≈ 2/3 of parameters).
- **Fit tag:** per level.
- **Test.** 0.8B fixed code, α ∈ {−1, 0, 1, 2}, seed 0 screen then 3 seeds: ~1 h.
- **Expected:** −0.2 to −1 %, P 0.25.
- **Conflicts:** #85-02.

**A6 · Direction-aware writer gains (closed form)**
- **Mechanism.** After a writer, the stream `h' = h + y` is read through RMSNorms. Errors parallel to h′ are partly harmless, and 2609.11716 finds 89–99 % of the final error is angular.
- **Fit.** Per-row gains `c` minimizing Σ_t ‖P⊥(h′_t)(c ⊙ ŷ_t − y_t)‖² / ‖h′_t‖² on bf16 inputs and bf16 h. This is a 5120-dim linear system: A = Σ w_t (diag(ŷ)² − (u⊙ŷ)(u⊙ŷ)ᵀ), one GEMM per tensor.
- **Why.** It could reproduce C9's family pattern (inflate `gdn_out`, shrink `attn_o` / `down`) from data, per row, without a grid.
- **Source:** new, motivated by 2609.11716 and #112.
- **Fit tag:** per level. It needs h at each writer, so a staged bf16 forward; the swj27 infrastructure has one.
- **Test.** 0.8B fixed code vs C9 α*: about 1 dev-day plus 1 h on the 4070.
- **Expected:** −0.3 to −1.5 % over C9, P 0.25. Risk: ignores later writers' relative magnitudes.
- **Conflicts:** #112, #108 (`aw` / `dw` used a writer-output loss for *reader* gains; this is different).

**A7 · Distil 0.8B healing into closed-form gain rules**
- **Mechanism.** Regress the healed scales log(c_heal / c_LS) per row and channel (#89-03 checkpoints) on features computable per level from bf16 data and codes: family, depth, own error e_n, K, row norm, Σdy² rank, input mean. Apply the fitted rule per level with no training.
- **Source:** #89 (healing K2 0.721 / K3 0.886 vs stack; closed-form swj captures only about ⅕ of its log-gain); TWIG 2610.31931 and ThinkBit 2610.34886 (scales carry most of the healing gain; they correlate only 0.07 with weight-MSE-optimal scales).
- **Why.** Healing is deferred at 27B. A per-level rule keeps only the transferable, systematic part, which is also the library-safe part.
- **Fit tag:** per level.
- **Test.** CPU regression (colo, under an hour). Apply to a fresh 0.8B seed and level as fixed code. Pass: ≥ 30 % of healing's log-gain on a level it was not fitted on.
- **Expected:** −1 to −4 % at 27B if it passes, P 0.3.
- **Conflicts:** #89 (deferred healing; this is not training).

**A8 · Gen-masked fixed-code fits**
- **Mechanism.** Weight the swj / swz / C9 fit tokens toward scored (generated) positions instead of all qad tokens.
- **Source:** #83 (gen-masked H_out loss: K3 0.971, K2 0.950) and #86 (gen-weighted H_in inconclusive: K2 0.976).
- **Why.** Only gen rows are scored. Scale fits have few parameters, so they are less noise-prone than reweighting H_in.
- **Fit tag:** per level.
- **Test.** 0.8B swj refit with mask vs without, fixed code: ~1 h.
- **Expected:** −0.2 to −0.8 %, P 0.3.
- **Conflicts:** #86.

**A9 · Robust swj / swz fit (heavy-tail guard)**
- **Mechanism.** Fit the channel grids on two halves of the fit tokens. If the halves disagree beyond noise, switch to a block-normalized or Huber-weighted fit.
- **Source:** #33-27 (a few sequences dominate early-layer Σdy²); NatScale (2610.20353) on smoothing.
- **Fit tag:** per level.
- **Test.** Split-half agreement of swj scales on the 0.8B. It is nearly free if logged in the 27B swj fit.
- **Expected:** 0 to −0.5 %, P 0.15.
- **Conflicts:** #104.

**A10 · α on the input-side (su) LS step**
- **Mechanism.** `refit_scales` alternates su and sv LS steps. Apply a per-family shrink correction to the su step as well.
- **Source:** w02 README (B1 was inert because only the k-vector of su could respond).
- **Fit tag:** per level.
- **Test.** 0.8B fixed-code screen, 2 families.
- **Expected:** ≈ 0 to −0.3 %, P 0.1. Low priority.
- **Conflicts:** #85 B1.

**A11 · Head temperature and per-slab α by exact head-only KL** (shortlist "head KL-side gain refit": concrete form)
- **Mechanism.** On headsim (32,768 real head inputs, exact KL), fit one global logit temperature, then α per Fisher slab with the C9 parametrization. Both fold into head svh rows.
- **Source:** headsim (#34); temperature-scaling calibration; SoftWater 2608.12026 (Fisher-weighted head error).
- **Why.** LS shrink of head rows is a non-uniform temperature change. The KL-optimal one has never been measured.
- **Fit tag:** per head level.
- **Test.** R9700, about 0.5 card-h per head level. Pass: head-only KL −5 % at the 2.5 mix's head K, guard KL not worse.
- **Expected:** −0.2 to −1 % total (head ≈ 5–7 % of total KL), P 0.4.
- **Conflicts:** #34.

**A12 · Head input-channel scales (su) / final-norm refit by KL**
- **Mechanism.** The final norm is read only by the head, so its gain vector equals head su, which is free. Take one diagonal-Newton step on head su against exact head-only KL.
- **Source:** ProQuant 2610.30779 (KL-trained norm gains fold); new for the head.
- **Fit tag:** per head level.
- **Test.** headsim, minutes.
- **Expected:** −0.1 to −0.4 %, P 0.3.
- **Conflicts:** none.

**A13 · Embedding row-energy restoration**
- **Mechanism.** The trellis embedding picks one scale per row by MSE, and MSE-optimal codes shrink rows by ≈ (1 − rel. error); that is about 5–7 % energy at t2–t2.5. The embedding then under-weights itself in the residual stream relative to every writer: a coherent error.
- **Fix.** Per-row `c = (1 − e_row)^(−α/2)`, α ∈ {−1, 1, 2}. Variant: choose c to minimize direction error at layer 0.
- **Source:** new; SoloQ 2610.07121 (inner-product-preserving scale).
- **Why.** The embedding is still a large KL term at low bits (#32 on a 4.0 base: e2 +38 %, e3 +8 %).
- **Fit tag:** per embedding level.
- **Test.** **27B fixed code**: same mix, only the embedding scale vector changes, so there is no rounding noise. 2–3 eval arms ≈ 2–3 card-h.
- **Expected:** −0.3 to −1.5 %, P 0.35.
- **Conflicts:** #35 (menu only), #32.

**A14 · Fixed-code fits for the fast (A8) variant**
- **Mechanism.** Refit swj / swz / C9 with A8-g128 fake-quantized inputs. Each variant ships its own container, so the scale vectors are free.
- **Source:** J6 idea in the earlier lens (healing with A8 on); E1b (A8 costs +5 % at K6).
- **Fit tag:** per level × variant.
- **Test.** 0.8B fixed code with the A8 hook.
- **Expected:** −0.1 to −0.5 % on fast variants only, P 0.15.
- **Conflicts:** none.

**A15 · Scale variants as knapsack options**
- **Mechanism.** Where a fixed-code rule helps only some tensors (C9 per row, A13 per row), keep both the LS and the variant svh per tensor. Let knap4 pick, priced by family factors.
- **Fit tag:** per level. Storage is fp16 vectors only.
- **Test.** Pricing on existing 0.8B arms.
- **Gain:** enabler for A3, A5 and A13. Risk: overfitting if priced on the search slice.
- **Conflicts:** none.

**A16 · Multi-level fixed-code fit passes**
- **Mechanism.** Swj needs only bf16 activations plus each level's quantized gate / up outputs. Compute 6–8 levels per bf16 forward instead of 2. It also gives swz and A6 the same bf16 pass.
- **Source:** swj27 pipeline (~1.5 card-h per two levels).
- **Why.** At 71 levels × several fits, the per-level bf16 forward is the cost that blocks this whole theme (≈ 50 card-h for swj alone).
- **Fit tag:** enabler.
- **Test.** Bit-identity of the scales against single-level runs.
- **Gain:** about 3–4× fewer card-hours. Risk: VRAM for the level weights (stream them per layer).
- **Conflicts:** none.

## B. Library vs mix compatibility

**B1 · Seed- and level-transferable part of asym**
- **Mechanism.** Compute the writer's asym correction from sibling moments averaged over three sibling seeds, or over levels K ± 0.5. Test on a fourth seed and on off-level siblings.
- **Source:** #87-07 (no transfer across seeds; F − I cosine 0.02–0.10), #111.
- **Why.** If any of asym's −14 to −18 % (0.8B plain) comes from the sibling's *systematic* error (its gain shrink), that part is per level.
- **Fit tag:** per level.
- **Test.** 0.8B, about 3 h on the 4070. Pass: ≥ 40 % of the matched gain kept off-seed.
- **Expected:** at 27B, conditional on #105, −1 to −4 %, P 0.2.
- **Conflicts:** #87-07, #111, #88 (the denoise-type target failed on all tensors; this is writers only).

**B2 · Conditional writer library (only if #105 passes)**
- **Mechanism.** Quantize each writer level against siblings at {K_w − ½, K_w, K_w + ½}. The packer picks the variant that matches the mix's sibling K.
- **Why.** It keeps asym's mix-specific gain (the per-mix pass keeps 0.81–0.90 on the 0.8B) without a per-mix requant.
- **Cost.** Writers are ≈ 29 % of parameters, so roughly +15–20 % ladder time.
- **Fit tag:** per level (conditional).
- **Test.** 0.8B #111 cells using the matched variant: should equal per-mix (≈ 0.81–0.90).
- **Expected:** most of asym's 27B gain, if any, P 0.3.
- **Conflicts:** #105, #111.

**B3 · Composite knapsack item for tied-K asym blocks**
- **Mechanism.** Per layer, offer {(K, asym writer, siblings at K)} next to independent plain options. Price asym by a measured per-K factor.
- **Source:** brief option list; new solver detail.
- **Fit tag:** solver.
- **Test.** CPU: run knap4 with the composite items on the 0.8B factors and compare predicted vs measured on 3 mixes.
- **Gain:** realizes asym without B2's extra quant cost, at the price of less K freedom. Risk: low.
- **Conflicts:** #111.

**B4 · Mixed-K robustness of every adopted fixed-code fit**
- **Mechanism.** Run #111's cells for C9 (and A1 / A6). Writer K ≠ upstream K; compare the per-level fit with a per-mix refit.
- **Why.** C9's α was chosen in uniform stacks. If its gain inverts in mixes like asym's did, we need B5 or A1 before shipping.
- **Fit tag:** check.
- **Test.** 0.8B fixed code, ~2 h. Run together with A2.
- **Gain:** protective. P that a problem exists: 0.3.
- **Conflicts:** #111, #112.

**B5 · Conditional scale variants**
- **Mechanism.** If B4 finds α depends on the upstream K, store 2–3 svh vectors per writer level, keyed by the upstream K offset. The packer selects; codes are shared.
- **Fit tag:** per level (conditional on scales only).
- **Test.** As in B4.
- **Gain:** keeps C9 in mixes. Cost: fp16 vectors only.
- **Conflicts:** #111.

**B6 · Per-mix "asym-lite"** (final pass)
- **Mechanism.** Writers' output scales are refit by LS on the shipped mix's own quantized inputs against clean outputs. Codes are frozen; one forward pass.
- **Source:** asym target (#87) restricted to scales.
- **Why.** About 10× cheaper than the per-mix asym requant (4–5 card-h a mix).
- **Fit tag:** final pass.
- **Test.** 0.8B #111 cells: what fraction of per-mix asym does it keep?
- **Expected:** −0.3 to −1.5 %, P 0.25.
- **Conflicts:** #85 B1 (inert, but that refit used H_out, not quantized inputs).

## C. Quantizer settings needing a requant (must precede the freeze)

**C1 · H_out at all K via a block-diagonal `ldlq_2hess` fast path**
- **Mechanism.** The rotated diagonal H_out is block-diagonal in 128-row blocks, but `ldlq_2hess` treats it as dense (72 % of a gate_proj's time). A blockwise path could bring the cost from 5× toward about 1.5×, and the user could then reconsider H_out for levels ≤ K3.5.
- **Source:** PROFILE.md target T1; #33-28 (K2.5: 0.972 / 0.992 / 0.976, every pair < 1, geo 0.9801).
- **Why.** At K2.5 this is the best-measured 27B gain not yet adopted. It stacks with swj (0.924).
- **Fit tag:** per level.
- **Test.** Bit-exactness against the current `ldlq_2hess` on 3 tensors. Then a 4th K2.5 seed pair if the user wants it under amendment 5.
- **Expected:** −2 % at 2.5, P 0.75. Cost: 1–2 dev-days.
- **Conflicts:** #33-28 (not adopted ≤ K3.5 on rule and cost; this attacks the cost).

**C2 · Tempered H_out at low K**
- **Mechanism.** Use D^p, rescaled to the same trace, with p ∈ {0.25, 0.5}, or larger output damping, at K ≤ 3.
- **Source:** ShamAN-Q 2609.38521 (p = ½ beats p = 1 when error ‖E‖ is large); earlier lens B2 (open). The 0.8B gain is smaller at K2 than at K3, which fits curvature being over-trusted.
- **Fit tag:** per level.
- **Test.** 0.8B K2 / K3, 3 seeds, about 2 h on the 4070. Pass: ≥ 2 % over plain bn at K2.
- **Expected:** −0.3 to −1 % on top of C1, P 0.3.
- **Conflicts:** #33-28.

**C3 · Log per-row e_n, LS shrink factors and pre-refit su / sv in every ladder job**
- **Mechanism.** Write one float per row and column for e_n and the LS gain, plus the pre-refit scales, into each level's metadata.
- **Why.** Every post-hoc rule (C9, A3, A6, A11, A13, A15) needs these. Recomputing them for 71 levels × 26.9 B weights means decoding plus H products on CPU; in the job they cost nothing.
- **Fit tag:** job spec, so before the freeze.
- **Test.** Bit-identity of codes with and without the extra outputs.
- **Gain:** enabler. Risk: none.
- **Conflicts:** none.

**C4 · Mean-direction weighting in LDLQ for gated-input writers** (after diagnostic I4)
- **Mechanism.** The writer inputs `normed core · silu(z)` and `silu(g)·u` have large positive means μ. The error ΔW μ is a constant residual-stream offset added at every token in every layer, and no bias exists to absorb it. Run LDLQ with H′ = H + λ μμᵀ, λ chosen per family.
- **Source:** new; classic bias correction (Nagel 2019) in a no-bias form. Related to the shortlisted "writers' mean-shift bias", with a zero-runtime variant.
- **Fit tag:** per level.
- **Test.** 0.8B I4 census first. If mean error exceeds 15 % of a family's proxy error, run a λ sweep on settle64 and then 0.8B KL.
- **Expected:** 0 to −1.5 %, P 0.2.
- **Conflicts:** #88 (targets), #38 (damping).

**C5 · Best-of-16 Hadamard sign seeds for the massive-channel readers only** (shortlist "massive-channel handling", concrete form)
- **Mechanism.** For the ~6 early readers at 1.35–1.74× the ideal coder (L10 `ssm_qkv`, L11 `attn_qg`, plus L3 / L7), search su / sv sign seeds by the real order-fixed LDLQ proxy. Add arms for undamped and higher-σ_in.
- **Source:** faraway M5 / FA-C2; 2605.06014 (ρ₃ and ‖·‖∞² per-block diagnostics).
- **What's different.** #41's 0.5 % spread was an average over ordinary tensors. Heavy-tail blocks can spread more.
- **Fit tag:** per level.
- **Test.** settle64 on colo, 6 tensors × 16 seeds × 3 K ≈ 2 CPU-h. Pass: ≥ 5 % proxy error on those tensors.
- **Expected:** 0 to −0.5 %, P 0.2.
- **Conflicts:** #41, #38.

**C6 · Tile CD on `attn_k`, and on every family at the sub-2 research levels**
- **Mechanism.** Extend `cd3` beyond `attn_qg`. On the GPU, `attn_k` gave 0.991 / 0.994 at about 2 s per tensor.
- **Fit tag:** per level.
- **Test.** No-harm 27B check like CD27, folded into the next attn_qg run.
- **Expected.** ≈ 0 % at 2.5 (`attn_k` is 0.3 % of parameters). Possibly −0.3 to −1 % at 1.x bpw. P 0.3.
- **Conflicts:** #77.

**C7 · Soft trellis marginals (BCJR) as a CD initializer**
- **Mechanism.** Run forward–backward over the 2^16-state trellis at temperature T to get marginal-MAP tile choices, then the existing tile CD.
- **Source:** MPQ 2610.32127 (E8P 2-bit: LDLQ 15.01 → +CD 12.76 → MPQ 11.63 PPL); BCJR-QAT 2605.10655.
- **Fit tag:** per level.
- **Test.** settle64 implementation, 27 tensors at K2 / K2.5. Pass: ≥ 2 % proxy error over cd3. About a dev-day plus 4 CPU-h.
- **Expected:** −0.2 to −1 %, P 0.15. Encode time 2–3×.
- **Conflicts:** #77, #113.

**C8 · One JARQ round: refit, then re-LDLQ with the refitted scales**
- **Source:** JARQ 2609.38599 (≤ 1 % on strong GPTQ hosts); earlier lens C7.
- **Fit tag:** per level; 2× quant time.
- **Test.** settle64.
- **Expected:** −0.1 to −0.5 %, P 0.15.
- **Conflicts:** #95-02.

**C9 · LDLQ feedback decay**
- **Mechanism.** Scale the feedback by γ_i = exp(−λ·[H⁻¹]_ii / mean · i/B).
- **Source:** QTEA 2609.00224 (+0.46 points on Qwen3-14B ternary).
- **Fit tag:** per level.
- **Test.** settle64, λ sweep.
- **Expected:** ≈ 0, P 0.1.
- **Conflicts:** #38.

**C10 · Partial activation denoising (ρ < λ)**
- **Mechanism.** The 2610.07522 target with λ = 0.4, ρ = 0.3 and γ = 0.05, applied to writers only.
- **Source:** closes 18–38 % of the parallel → sequential gap there.
- **Fit tag:** per level.
- **Test.** 0.8B K2.
- **Expected:** 0 to −0.5 %, P 0.1.
- **Conflicts:** **#88** (failed at level-matched noise, ρ = λ); try only the partial form.

**C11 · Schur-complement scoring of scale candidates**
- **Mechanism.** Score g_scale candidates by δᵀ S_{G|P} δ, using the LDL block already computed, instead of raw tile MSE.
- **Source:** 2609.36654; 2609.37416 (1/U_qq² weighting).
- **Fit tag:** per level.
- **Test.** settle64.
- **Expected:** ≈ 0, P 0.1, given that #39 found gain flat.
- **Conflicts:** #39.

## D. Bytes outside the trunk codes, and format items

**D1 · Quantize GDN `in_proj_a` / `in_proj_b`** (shortlist "GDN a/b int8", priced)
- **Mechanism.** These 23.6 M parameters are bf16 today, costing 0.0140 bpw. Options: int8 per row saves 0.0070 bpw; exl3i K5–6 saves about 0.0095.
- **Source:** 2609.04098 on our model: NVFP4 a/b raise GDN-output error by only 2.1 % / 2.6 %, the least sensitive projections.
- **Why.** At −1.7 % KL per 0.01 bpw, this is worth about −1.2 to −1.6 % at 2.5 if their own cost is ≈ 0.
- **Fit tag:** per level (tiny tensors, minutes).
- **Test.** Dequantize a/b at int8 / K5 / K4 into bf16 via G5. One eval pair on the 2.5 mix (~1 card-h). Pass: ≤ +0.15 % KL. Real bytes need the a/b op to accept the encoding (plugin work).
- **Expected:** −1 to −1.6 % at 2.5, P 0.7.
- **Conflicts:** v3 §1 "GDN a/b f16" is a *compute*-mode decision, not storage.

**D2 · 8-bit log-scale su / sv**
- **Mechanism.** Per-tensor fp16 scale vectors total ≈ 7.3 M values (0.0043 bpw). An 8-bit log code per value with a per-tensor offset saves about 0.002 bpw ≈ −0.35 %. Only apply it at ≤ K4, where 0.3 % scale error is far below code noise.
- **Fit tag:** pack-time format plus kernel prep / epilogue.
- **Test.** CPU proxy error with the scales re-encoded.
- **Expected:** −0.3 %, P 0.6. Kernel work needed.
- **Conflicts:** none.

**D3 · conv1d and norm weights to int8**
- 0.0006 bpw. Noted for completeness; skip unless D1's plumbing makes it free.

**D4 · Two-class MLP channel split at L62–L63** (shortlist "low-rank / spike capture", concrete form)
- **Mechanism.** Permute the intermediate channels; the permutation is free because gate/up rows and down columns move together. Put the top 1 % (174 channels, 35.5 % of L63's gate+up Σdy² importance) in a small high-K or bf16 sub-MLP.
- **Bound.** Ceiling ≈ 0.28 bit on L63 gate_up (1 %/99 % split); a 50/50 split gives 0.62 bit.
- **Fit tag:** format (an extra small GEMM in the L63 MLP); per level for the main part.
- **Test.** G5 splice test at 27B, 2 arms, ~2 card-h.
- **Expected:** −0.2 to −0.6 % net of 0.0005 bpw for spike storage, P 0.4.
- **Conflicts:** #21 (the L63 spike), #96.

**D5 · Per-head K for GDN qkv / z**
- **Mechanism.** Taking whole 128-row blocks from different levels is consistent with the 128-point output Hadamard, but each level's su differs. Measure the value first with dequantized splices (G5). A real format needs su tied across the levels involved, or per-span input transforms.
- **Bound.** Ceiling from Σdy² weights is 0.283 bit on qkv and 0.122 bit on z ≈ **0.034 bpw**. Assume about ⅓ of that is realizable (H_out-style shrink).
- **Fit tag:** format change, then a requant with tied su.
- **Test.** 27B splice mix: high-importance heads at K + ½, low at K − ½, equal bytes. 3 card-h.
- **Expected:** −1 to −2 % if it passes, P 0.3. Must precede the freeze only if adopted.
- **Conflicts:** #96 ("only with other per-slab work"); this supplies the measurement.

## E. Head and embedding: other items

**E1 · Head H_in from generated rows only**
- **Mechanism.** The head matters only at scored positions. Recompute the head's H_in from dumped gen-row inputs, then requantize the head levels.
- **Source:** earlier lens G1 (partly); #83 analogue.
- **Fit tag:** per head level. Each head quant needs about 22 GB of host RAM.
- **Test.** headsim at K4 / K5.
- **Expected:** −0.1 to −0.4 %, P 0.25.
- **Conflicts:** #34.

**E2 · ε-smoothed Fisher for slab ordering (rare-token guard)**
- **Mechanism.** Rank rows by (1 − ε)·F + ε·mean, ε = 0.1, for ordering and the floor.
- **Source:** SoftWater 2608.12026; #32 (rare rows matter).
- **Fit tag:** per head level.
- **Test.** headsim guard KL vs mean KL.
- **Gain:** a multilingual safety margin; neutral on mean, P 0.2.
- **Conflicts:** #34.

**E3 · Softmax-shift t-grid beyond the mean row**
- **Source:** 2609.31291 (Qwen3.5 + GPTQ: no gain); earlier lens G3.
- **Test.** headsim, minutes.
- **Expected:** ≈ 0, P 0.1. Do it only because it is free.
- **Conflicts:** #34.

**E4 · Two-sided head LDLQ with the analytic softmax Fisher**
- **Caveat.** Once rows are Fisher-sorted into slabs, the spread inside a 128-row block is small, so the gain is expected to be small.
- **Test.** headsim, one head level.
- **Expected:** 0 to −0.3 %, P 0.15.
- **Conflicts:** #34.

## F. GDN specifics

**F1 · Decay-lifetime covariate in kfit2**
- **Mechanism.** Add the per-layer mean GDN gate lifetime Σ_j e^{2jℓ} as a depth covariate for qkv / z sensitivity.
- **Source:** STEPQuant 2609.38169 (half-life vs error persistence ρ ≈ 0.80 across 2,304 Qwen3.8 heads, for the state).
- **Fit tag:** pricing.
- **Test.** CPU refit; compare CV log-RMSE.
- **Expected:** 0 to −0.5 %, P 0.2.
- **Conflicts:** earlier lens L1 / F10 (open).

**F2 · Conv-aware H_in: test inertness before building** (shortlisted item)
- **Argument.** The per-row effective Hessian is H_r = Σ_{τ,τ′} c_τ c_τ′ C(τ − τ′). If the lagged covariances C(1..3) have the same *shape* as C(0), H_r is a scalar multiple of H_in and LDLQ is unchanged.
- **Test.** On the 0.8B, compare the principal angles of C(1) vs C(0) on 4 GDN layers (forward-only statistics, 15 min). Build the feature only if they differ (cos < 0.9).
- **Gain:** decides a shortlist item cheaply.
- **Conflicts:** shortlist.

## G. Sensitivity measurement and knapsack pricing

**G1 · kfit3 on the two-sided proxy**
- **Mechanism.** err₂ = Σ_r h_out,r · e_rᵀ H_in e_r from the existing bn Σdy² diagonal. No new collection; recompute for every (tensor, option) row and refit.
- **Source:** BACKLOG "Σdy² as the knapsack price" (queued); 2609.33923 on Qwen3.8-27B (the layer objective misallocates: oracle-Q_real allocation PPL 6.087 vs block-cosine 6.041).
- **Fit tag:** pricing.
- **Test.** CPU, run where the levels are stored. Pass: CV log-RMSE below kfit2's 0.11–0.13 and better pair ranking.
- **Expected:** −0.3 to −2 % via allocation, P 0.4.
- **Conflicts:** w08-hprice.

**G2 · Fixed-code effect factors per (family, K) in knap pricing**
- **Mechanism.** Extend knap5's swj "gu" factor to swz, C9, the head and the embedding.
- **Fit tag:** pricing.
- **Test.** Predicted vs measured on 3 mixes.
- **Gain:** allocation follows the healed sensitivities. Small, cheap, P 0.5.
- **Conflicts:** w58 / w63.

**G3 · Sibling pair features in kfit2 from existing data**
- **Mechanism.** Add within-layer pair terms (gate_up × down, GDN in × out, qkv × o) as features fitted on the ≈ 1,300 V2 plus V3 whole-mix measurements. No new evals.
- **Source:** TWLA 2606.13054 (pair-aware DP 11.92 → 8.31 PPL, activation bits); 2610.09227 (single-module sums are compressive, β ≈ 0.53).
- **Fit tag:** pricing.
- **Test.** CPU grouped-CV.
- **Expected:** 0 to −1 %, P 0.3.
- **Conflicts:** **F5 / #16** (Shapley dropped as too expensive; this needs no new measurements).

**G4 · Antithetic swaps**
- **Mechanism.** Measure a group swap with error Δ and with the mirrored tensor 2W − Ŵ (packed bf16 via G5). The average cancels the realization-specific linear term (interaction with the base mix's errors) and halves the variance of the quadratic estimate.
- **Source:** TWIG 2610.31931 (symmetric difference).
- **Fit tag:** measurement.
- **Test.** 10 paired swaps at 27B; compare the SE of the sensitivity estimate.
- **Gain:** less noisy sensitivities, P 0.2.
- **Conflicts:** #14 (single swaps too noisy; this targets the rounding part, not the document part).

**G5 · Splice harness: dequantized bf16 tensors inside 27B evals**
- **Mechanism.** Pack the effective weights of sub-tensor allocations as bf16 options: per-head K, channel splits, mirrored errors, a/b quantization. Bytes are accounted analytically. bf16 is already a menu option.
- **Fit tag:** measurement enabler (D1, D4, D5, G4).
- **Test.** Byte-identity sanity: splice a level's own tensor dequantized and check KL equals the packed result within the f16 floor.
- **Gain:** removes kernel work from decision tests.
- **Conflicts:** none.

**G6 · Proxy validity after swj**
- **Mechanism.** Re-fit gate_up's factor between proxy error and KL using swj'd levels.
- **Why.** Swj changes gate_up's effective error, so an H_in proxy computed pre-swj misprices it by K-dependent amounts (−9 % at K2 vs −1 % at K4).
- **Fit tag:** pricing.
- **Test.** CPU with existing data plus 4 swaps.
- **Gain:** correct K slope for the largest family, P 0.4.
- **Conflicts:** w63's corrected swj curve partly covers this.

**G7 · Re-anchor sensitivities at the 2.5 operating point**
- **Mechanism.** Fractional-factorial group swaps around the knap4 2.5 mix instead of the V2 4.0 base.
- **Source:** MixQuant 2607.23047; Q-Strata 2608.30564 (in-context marginals 12.14 vs 13.72 PPL at 1.75 b).
- **Fit tag:** pricing.
- **Test.** 16 swap evals on a search slice.
- **Expected:** −0.3 to −1 %, P 0.3.
- **Conflicts:** earlier lens F4 (open).

**G8 · Inversion-rate check near the optimum**
- **Mechanism.** Measure 10 pairs of near-equal-predicted mixes around the knapsack optimum and count rank inversions.
- **Source:** 2610.09227 (15.3 % inversions in the top quartile).
- **Fit tag:** diagnostic.
- **Test.** 20 evals.
- **Gain:** tells how far to trust knap4 against the search, P n/a.
- **Conflicts:** none.

**G9 · Sensitivity tempering knob**
- **Mechanism.** Shrink sensitivities toward the family mean, s → s̄^(1−β)·s^β, with β fit on the search pool.
- **Source:** NatScale 2610.20353 (whole-model β ≈ 0.2); earlier lens H8.
- **Fit tag:** solver.
- **Expected:** 0 to −0.5 %, P 0.15.
- **Conflicts:** none.

## H. Search statistics (measurement design only, no runner plumbing)

**H1 · Successive halving over candidate batches**
- **Mechanism.** Score N candidates on ¼ of the slice, keep the best half, extend, and reuse the scores.
- **Source:** Loopy 2610.05265 (3.5–4.2× calibration savings); earlier lens G3.
- **Gain:** ~2× evals per card-hour, P 0.5.
- **Conflicts:** none.

**H2 · Prescreen budget-neutral swaps by an adjacent-pair cross-term proxy**
- **Source:** CASA 2609.25916 (cross-term screening 28.15 → 22.41 PPL vs self-term screening at 2.25 bpw).
- **Test.** Replay V2 local-search logs offline.
- **Expected:** −0.2 to −0.8 %, P 0.25.
- **Conflicts:** #70.

**H3 · Cap the changes per accepted step and re-price after**
- **Source:** 2609.09867 (composition error is 26–42× the pricing error); earlier lens F7.
- **Gain:** fewer bad accepts, P 0.3.
- **Conflicts:** #70.

**H4 · Lazy measured descent as a stream kind**
- **Source:** Q-Strata 2608.30564 (~2.5 evals per move); earlier lens F6.
- **Gain:** −0.3 % at low bpw, P 0.2.
- **Conflicts:** #70.

**H5 · Fixed-code variant flips as a mutation kind**
- **Mechanism.** Mutations that only switch a tensor's scale variant (A15). There is no rounding noise, so they are cheap to accept or reject.
- **Gain:** small, P 0.3.
- **Conflicts:** none.

**H6 · Discriminative-document search slices**
- **Mechanism.** Weight documents by between-candidate variance divided by within-candidate noise, estimated from V2's 1,450 evals.
- **Source:** earlier lens I3.
- **Gain:** ~1.3× lower SE per GPU-second, P 0.3.
- **Conflicts:** #58 capstudy.

**H7 · Regression-adjusted (CUPED) paired estimator with incumbent per-document KL**
- **Source:** earlier lens I4.
- **Gain:** 10–30 % lower SE, P 0.3.
- **Conflicts:** none.

## I. Diagnostics that change priorities (cheap)

**I1 · Coherent vs random error decomposition (0.8B, then 2B)**
- **Mechanism.** Per row, split e_r = β_r·w_r + e⊥. Fake-quant models: (a) gain error only, (1 + β)⊙W; (b) noise only, W + e⊥. Measure their KL shares at K2 / K3, per family.
- **Why.** It tests the brief's hypothesis directly and sizes the headroom for all of theme A. The 2B run gives the size trend.
- **Test.** About 1.5 h on the 4070 (0.8B), about 4 h (2B).

**I2 · 27B residual-norm and cosine trajectory under a uniform level**
- **Mechanism.** Staged forward of K2.5 vs bf16 on 8 documents. Compare ‖h_l^q‖ / ‖h_l‖ and cos(h_l^q, h_l) by layer, before and after swj.
- **Why.** Coherent drift appears as a norm trend. A ratio near 1 means only direction fixes (A6) pay.
- **Test.** About 1 card-h.

**I3 · KL budget map at the 2.5 mix**
- **Mechanism.** First, kfit2's predicted per-family, embedding and head shares (CPU, free). Then verify the top 3 with K6 stand-in evals.
- **Why.** It decides whether A11–A13, E1–E4 and D4 matter.
- **Test.** CPU plus 3 card-h.

**I4 · Mean-error census**
- **Mechanism.** ‖ΔW μ‖² / tr(ΔW H ΔWᵀ) per family on the 0.8B, using μ from one forward pass.
- **Why.** It gates C4.
- **Test.** About 30 min.

**I5 · Swj split-half agreement**
- **Why.** It gates A9.
- **Test.** Nearly free at the next 27B swj fit.

**I6 · Proxy → 27B transfer table by mechanism class**
- **Mechanism.** Tabulate (0.8B gain, 27B gain) for swj, H_out, xcorp, i8c and CD.
- **Why.** It formalizes the shrink priors that every ranking here depends on.
- **Test.** Bookkeeping only.

**I7 · α\* by depth from the healed 0.8B checkpoints**
- **Why.** It feeds A3 and A7.
- **Test.** CPU analysis of existing files.

**I8 · GDN lag-covariance shape**
- **Why.** It gates F2.
- **Test.** About 15 min on the 0.8B.

## J. Final pass on 2–3 shipped mixes (one pass, per mix)

**J1 · Per-mix refit of swj / swz / C9 on the mix's quantized upstream**
- **Why.** It captures the counteraction effect (2609.11716: a block's response to its input error cancels 50–63 %).
- **Test.** 0.8B mixed-K cells.
- **Expected:** −0.3 to −1 %, P 0.3.
- **Conflicts:** #88-02 (that was a code target; this is scales only).

**J2 · Per-mix asym-lite**
- See B6.

**J3 · One diagonal-Newton step on all scales by end-to-end KL**
- **Mechanism.** One Fisher-diagonal step, not training. A bridge while healing is deferred.
- **Test.** 0.8B: what fraction of the 512-step healing gain does one step give? Pass: ≥ 25 %.
- **Expected:** −1 to −3 % per shipped mix, P 0.25.
- **Conflicts:** #89 (deferred). This is one pass; a user call.

## K. Killed or deprioritized by today's checks

- **K1 · Diagonal-metric embedding LDLQ.** The layer-0 residual-gradient diagonal is near-isotropic: ceiling 0.037 bit ≈ 0.0017 bpw. A full-matrix version needs a new gradient collection, which the user has ruled out.
- **K2 · Separate gate / up K.** 0.005 bit mean split gain (L63 alone 0.13).
- **K3 · Per-part K for GDN q / k / v.** 0.043 bit mean.
- **K4 · Separate K for attention q vs gate halves.** ≤ 0.125 bit, mean ≈ 0.05, on 3.7 % of parameters.
- **K5 · Dedupe or sub-1 storage of dead embedding / head rows.** 0 duplicates; 0.66 % low-norm rows, < 0.0005 bpw.
- **K6 · Output-side weighting for writers.** Writer-output Σdy² spread is 0.016 bit inside 128-blocks.
- **Also out of scope this week:** a stored start state per block (CurveTQ-style, 0.0625 bpw per tile to recover the 4–6 % tail-biting loss; it costs more bits than it saves) and weighted Viterbi under RHT (inert per CurveTQ; earlier lens D9).

---

## Ranked top 20 by expected gain per unit cost

| # | idea | expected KL at 2.5 | P | cost | freeze? |
|---|---|---|---|---|---|
| 1 | **A1** swz z-gate refit in the 27B stack | −1 to −3 % | 0.5 | 1.5 h on the 4070, ~2 card-h at 27B | no (post-ladder) |
| 2 | **D1** GDN a/b to int8 or K5 (0.007–0.010 bpw) | −1 to −1.6 % | 0.7 | minutes to quantize, 1 card-h eval, plugin op | no |
| 3 | **A2 + B4** swz vs C9 attribution and mixed-K robustness | protects −1 to −4 % | 0.3 (risk real) | 2 h on the 4070 | no |
| 4 | **C3** log per-row e_n, LS gains and pre-refit scales in ladder jobs | enabler | — | zero | **yes** |
| 5 | **A13** embedding row-energy α (27B fixed code) | −0.3 to −1.5 % | 0.35 | 2–3 card-h | no |
| 6 | **A11 + A12** head τ / per-slab α / su by exact head KL | −0.3 to −1 % | 0.4 | ~1 card-h on headsim | no |
| 7 | **A3** C9 α with depth knots | −0.2 to −0.8 % | 0.35 | 1 h on the 4070 | no |
| 8 | **C1** block-diagonal `ldlq_2hess` → H_out at ≤ K3.5 | −2 % | 0.75 | 1–2 dev-days, ladder ×1.5 at low K | **yes** |
| 9 | **A7** healing-distilled gain rules | −1 to −4 % | 0.3 | 1 CPU-day plus 2 h on the 4070 | no |
| 10 | **G1** kfit3 on err₂ (existing Σdy²) | −0.3 to −2 % | 0.4 | CPU | no |
| 11 | **A16** multi-level fixed-code fit passes | enabler (3–4× cheaper theme A) | — | dev-day | no |
| 12 | **I1** coherent vs random decomposition | decides theme A | — | 1.5 h on the 4070 | — |
| 13 | **A6** direction-aware writer gains | −0.3 to −1.5 % | 0.25 | dev-day plus 1 h on the 4070 | no |
| 14 | **A8** gen-masked fixed-code fits | −0.2 to −0.8 % | 0.3 | 1 h on the 4070, 1.5 card-h | no |
| 15 | **D5** per-head K for GDN qkv / z (splice test first) | −1 to −2 % | 0.3 | 3 card-h test; format later | if adopted |
| 16 | **A5** C9 α on swj'd gate_up | −0.2 to −1 % | 0.25 | 1 h on the 4070 | no |
| 17 | **G5** splice harness | enabler | — | half a day | — |
| 18 | **G3** sibling pair features in kfit2 | 0 to −1 % | 0.3 | CPU | no |
| 19 | **D4** L63 spike-channel split | −0.2 to −0.6 % | 0.4 | 2 card-h test plus plugin op | if adopted |
| 20 | **C2** tempered H_out at low K (with C1) | −0.3 to −1 % | 0.3 | 2 h on the 4070 | **yes** |

The ranking leans on three structural facts:
- Fixed-code tests have no rounding noise, so they are 3–5× cheaper to decide at 27B than requant tests.
- Bytes outside the trellis codes convert to KL at about 1.7 % per 0.01 bpw.
- Coherent fixes have so far grown from 0.8B to 27B, while noise-shaping fixes have shrunk.

## The 5 cheapest diagnostics that would most change this ranking

1. **I1, coherent vs random error decomposition (0.8B, ~1.5 h on the 4070).** If the gain-only model carries < 10 % of the KL at K2–K3, theme A (ranks 1, 3, 5–7, 9, 13, 14, 16) drops a tier. If it carries > 30 % and grows from 0.8B to 2B, A7 and A6 move to the top.
2. **A2 + B4 on the 0.8B (~2 h on the 4070).** This shows whether swz absorbs C9's `gdn_out` α = 3 and whether α survives mixed K. It decides between A1 as the fix, conditional scale variants (B5), or both.
3. **I3 first pass from kfit2 (CPU, free).** The predicted embedding, head and per-family KL shares at the 2.5 mix decide whether A11–A13 and the head ideas are worth 27B card time.
4. **G1, CV of kfit2 with err₂ (CPU on existing data).** If err₂ beats the one-sided proxy on pair ranking, the knapsack pricing change is the cheapest allocation gain available. If not, Σdy² stays a quantizer-only signal.
5. **D1 a/b splice eval (one 27B eval pair).** If quantizing a/b costs ≤ 0.15 % KL, about 0.007–0.010 bpw is free money at every size, and the plugin work is justified.
