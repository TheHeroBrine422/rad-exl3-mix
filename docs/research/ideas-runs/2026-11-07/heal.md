# Heal lens: ideas for post-quantization training (healing / QAD), 2026-11-07

I didn't run any experiments for this report. I read the earlier heal report (`docs/ideas/2026-11-02/heal.md`, cited as **h1102-X** below), the w06/d1104/d1105 results and logs, and the 27B config on colo. The ideas below build on what has been measured since 11-02 rather than repeating that list. Tags: **P08** = the 0.8B proxy on the 4070 (`heal.py`); **S27** = the 27B healing stack, not built yet.

## 0. What is now known, and the constants every idea uses

**0.8B results, all with codes fixed:**

| test | result | source |
|---|---|---|
| heal-all vs base | K3 0.822, K2 0.619 | w06 |
| oracle ceiling | K3 0.432, K2 0.377; only 31% of the K3 ceiling generalises | w06 |
| last quarter only vs all layers | gets 45% (K3) / 41% (K2) of the all-layer gain; all layers wins at z +21 | w06 H1b |
| bf16 backward | gradcos PASS at both K | w06 H1c |
| level → mix transfer | 0.79 / 0.85 of the mix-specific gain; ¼-budget polish 0.96 / 0.97 → LIBRARY | w06 H4 |
| healing on bdiag (H_out) | keeps 0.87 of the gain | w06 |
| healing on edy+swj+asym (the full stack) | K3 0.886 / 0.891, K2 0.721 / 0.708 | #89-03 |
| one pass over 2,048 fresh windows vs 2 × 1,024 | 0.995 | #89-04 |
| xcorp + own-traces data vs qad | 1.014 (worse) | #89-05 |
| K4 at lr 1e-3 / 3e-4 | no gain / 0.949 | #89-06 |
| norm gains alone | ≤ 0.2% | #89-02 |
| retention on a fresh verify slice | 0.90 / 0.74 per healing step; 75–77% of the gain survives on xcorp | #93 |
| healing on self-generated or raw own text | +10% worse on calib | #91 |

**Ranking evidence.** The family mix and the depth mix keep their order after healing: unhealed mixD 0.320 < mixF 0.330; healed mixDh 0.209 < mixFh 0.223. The gap grew instead of closing.

**27B healing size** (computed from `config.json`):

| layer type | count | log-scale params per layer |
|---|---|---|
| GDN (qkv 15,360, z 11,264, out 11,264, MLP 67,584) | 48 | 105,472 |
| full attention (q+gate 17,408, k 6,144, v 6,144, o 11,264, MLP 67,584) | 16 | 108,544 |

- That is 6.80M scale parameters, about 5.7M after the down-r ≡ up-c gauge ties, against 0.55M on the 0.8B.
- The qad pool has about 3.7M tokens (≈3M scored rows). That is about 2 parameters per scored row, against 0.25–0.55 on the 0.8B: **the 27B is 4–8× more parameter-dense per training row.** Regularization and data matter more there than the 0.8B shows.

**27B student building blocks already in the repo:**
- `v3/sib27/deq.py`: a pure-torch decoder, bit-exact with the plugin oracle, covering fractional K and i8.
- `v3/sib27`: layer-streamed boundaries on disk.
- The YAQA collector's two-card 32/32 layer split: 5 s per 2k sequence in bf16.
- The published calib golden: top-512 log p, tail mass and gen mask for every pool (h1102 §0).

**Estimated cost of one 27B heal** (K2.5 codes 8.4 GB, single R9700):
- fwd + recompute + dX ≈ 3 × 110 TFLOP per 2k window → est. 4–7 s per window.
- 2,048 windows ≈ 2.5–4 h; data-parallel on two cards ≈ 1.3–2 h.
- ¼-budget polish ≈ 40–60 min.

**Gain convention.** Gains are the extra KL reduction at 27B 2.5 bpw. The 0.8B K3 stacked model has about the same absolute KL (≈0.066) as the 27B at 2.5 bpw, so it is the reference. My prior for core healing on top of swj at 27B is **−5…−11% (central −7%)**, and −10…−20% at 2.0 bpw. Healing fixes coherent per-channel error, which the brief expects to transfer better than ⅓; swj transferred at about full size at K2.5 but shrank at K4.

---

## A. 27B healing stack (S27): design, memory, gates

**A1 · Cached teacher: top-k KL from the golden, with a tail cell, and a k dose.**
- **Mechanism:** Replace the online bf16 teacher with the golden's stored top-k log p plus the tail mass. Use the metric-exact row loss KL = Σ_top p log(p/q) + p_tail log(p_tail/q_tail), summed in fp64, with gradient through the student's log-softmax. Dose k ∈ {32, 128, 512}.
- **Source:** h1102-D1, FidelityQ 2610.14934 (top-512 cache). New here: the k dose and the student-tail cell as a gate for the 27B.
- **Why:** A 54 GB bf16 teacher cannot share a 32 GB card with the student, and TP2 is unreliable, so the 27B *must* use a cached teacher. Top-512 storage for the qad pool is ≈11 GB; k=64 is ≈1.4 GB.
- **Fit:** training infrastructure.
- **Test:** P08 heal-all K3/K2, cached teacher vs online. Pass if the ratio is within 1.005 at both K for k=512, and report the smallest k within 1.01. Cost ≈1.5 GPU-h.
- **Gain/risk:** 0% by itself; it is the enabler. Risk: if the tail cell is badly shaped, rare-token mass is under-trained (check golden-ml).
- **Conflicts:** none; #34 uses the same golden.

**A2 · Single-card student with on-the-fly decode.**
- **Mechanism:** Keep codes (8.4 GB at 2.5 bpw) on the card. Decode each layer to a transient bf16 weight in the forward, the recompute and the backward. Use the exl3i decode as a torch op; `deq.py` is the fallback. Multiply the healing parameters in fp32 around the GEMM.
- **Memory:**
  - layer checkpoints: 21 MB per layer per 2k window (1.3 GB at 2k, 5.4 GB at 8k, 10.7 GB at 16k);
  - one decoded layer: 0.84 GB;
  - logits in 128-row chunks: 127 MB;
  - totals ≈14 GB at 2k, ≈20 GB at 8k, ≈26 GB at 16k.
- **Source:** w06 heal.py design, sib27/deq.py; new is the 27B single-card memory plan.
- **Why:** It makes 27B healing independent of card 1, the user's TP2 constraint.
- **Fit:** training infrastructure.
- **Test:** gate = identity (scales 0) vs radiance KL on the same container (≤1e-4 relative), then 8 timed windows on gpubox card 0. Half a day of engineering beyond S27, about 1 card-h.
- **Gain/risk:** enabler. Risk: if decode is slow, see A3; fla's GDN backward on ROCm is unproven on gfx1201.
- **Conflicts:** none.

**A3 · Layer-major micro-batches to amortize a slow decode.**
- **Mechanism:** If decode costs more than ~0.1 s per layer (likely with pure-torch `deq.py`), run each optimizer step layer-major. Forward all n windows of the step through layer l before decoding l+1, keep the n boundaries on the card, and run the backward in reverse. Each layer is decoded twice per step instead of 3n times.
- **Memory:** n=8 at 2k means boundaries of 10.7 GB.
- **Source:** sib27 streaming; new as a training schedule.
- **Why:** Cuts decode overhead 12× at n=8.
- **Fit:** infrastructure.
- **Test:** time per window on card 0 at n ∈ {1, 4, 8}; adopt if ≥1.5× faster at n=8.
- **Gain/risk:** cost only. Risk: activation memory caps window length at about 4k.
- **Conflicts:** none.

**A4 · Elastic data parallel over the two R9700s with a host all-reduce.**
- **Mechanism:** Each card runs a full A2 student on different windows. The gradients of ~6.8M parameters (27 MB fp32) are averaged through host memory each step, with no P2P or RCCL. If card 1 is taken back, card 0 continues alone from the same state.
- **Source:** new; it avoids TP2 entirely.
- **Why:** About 2× throughput when card 1 is free, and it survives preemption.
- **Fit:** infrastructure.
- **Test:** two-card vs one-card run with an identical step count; the scales must match to fp32 summation order.
- **Gain/risk:** halves wall time. No quality risk.
- **Conflicts:** none.

**A5 · 27B gradient-precision check (gradcos at 27B).**
- **Mechanism:** On 16 windows, compare dL/d(log c, log r) per layer from the bf16 backward with an x2 split-precision backward. A full fp32 27B does not fit, so do it on layers 0–15 only, with the rest bf16.
- **Source:** w06 H1c PASS on the 0.8B; #33-03 (27B bf16 H_out rows wrong early, cos 0.36 at layers 0–15).
- **Why:** This is the most plausible explanation of the field report ("last 16 layers > all"). The 0.8B cannot test it.
- **Fit:** gate.
- **Test:** card 0, about 1 h once A2 exists. Pass: cos ≥ 0.95 per layer, or at least the fp32 split-half cosine.
- **Gain/risk:** if it fails, mixed-precision dX for early layers (C7 in h1102) is needed, at about +40% time.
- **Conflicts:** #33-03; this time the quantity measured is the healing gradient, not the H_out rows.

**A6 · Fold, pack and plane round trip on both kernel paths.**
- **Mechanism:** Write healed suh/svh into the exl3i container as fp16 and norms into the checkpoint. Score in radiance on the f16 path and on the A8 path; the torch score must match.
- **Source:** h1102-A7, w11 review.
- **Why:** fp16 rounding of a healed scale moves weights (the k3jit floor was 1.0044 on the 0.8B), and A8 prep reads suh. A healed scale outside f16 range would be silent.
- **Fit:** gate for every shipped heal.
- **Test:** 2 evals per healed container.
- **Gain/risk:** none; it protects the rest.
- **Conflicts:** none.

**A7 · Healing as a rad-quant-at-home job type.**
- **Mechanism:** Job = (level id, window-set hash, recipe hash) → healed scales. The input is pinned on HF. The server verifies by score: it recomputes held-out KL on 8 cached windows with the returned scales and accepts if within the expected band.
- **Source:** rqah's verify-by-score; new for healing.
- **Why:** A library of 25 pass-1 levels at ~3 h each is ≈75 card-h, too much for gpubox alone.
- **Fit:** per-level library.
- **Test:** one donor-style job on gpubox with a forged-scale rejection test.
- **Gain/risk:** throughput only. Risk: the "big-box" tier needs ≥16 GB VRAM and ~20 GB RAM.
- **Conflicts:** none.

**A8 · One healing-data cache for every level.**
- **Mechanism:** Build once: the fixed qad window set (token ids, gen mask, top-k teacher from A1), plus layer-boundary bf16 hidden states at 4 depths for D3 and F3, plus the held-out early-stop windows. Every level, mix and polish reads the same cache.
- **Source:** new.
- **Why:** Results are comparable across levels, and there is no teacher pass per heal.
- **Fit:** infrastructure.
- **Test:** none.
- **Gain/risk:** none. Ledger entry needed for the qad pool.
- **Conflicts:** none.

**A9 · Gauge ties and parameter census at 27B.**
- **Mechanism:** Port the 0.8B gauge fixes (down r ≡ up c, v c tied to o r over the GQA group, norm gains vs readers' suh), plus GDN-specific ties: out_proj r ↔ the gated-norm weight, and qkv rows shared by conv1d. That gives ≈5.7M free parameters. Log the census.
- **Source:** w06 selftest 3.
- **Why:** Flat directions drift under Adam and waste fp16 range. Fewer parameters also helps data starvation (see C6).
- **Fit:** infrastructure.
- **Test:** CPU self-test (equivalence to 5e-5).
- **Gain/risk:** none.
- **Conflicts:** none.

## B. What to train

**B1 · Granularity decomposition (diagnostic).**
- **Mechanism:** On P08, heal the same quant with (a) one scalar per tensor (150), (b) scalars per tensor × 4 depth bands, (c) svh only, (d) suh only, (e) both (the current recipe). Report each as a share of (e).
- **Source:** h1102-B1/B7, never run.
- **Why:** If (a) or (b) gets ≥50% of the gain, the 27B can heal forward-only (J2) before S27 exists, and there is little to overfit. If (c) ≫ (d), the 27B parameter count halves.
- **Fit:** diagnostic.
- **Test:** 4070, 5 arms × 2 K × 15 min ≈ 2.5 h. No pass rule: it decides J2 and C6.
- **Gain/risk:** none directly; it changes the ranking.
- **Conflicts:** #89 decomposed only norms (≤0.2%).

**B2 · Family decomposition (diagnostic).**
- **Mechanism:** Heal only the MLP / only GDN in-projections / only the full-attention projections / only the writers (down, out_proj, o_proj). Report shares and additivity (sum of singles vs all).
- **Source:** h1102-C6; 2609.04098 (GDN error sits in qkv / z / out: 10.4 / 9.9 / 12.7%; a/b 2.1 / 2.6%).
- **Why:** It tells us where the 27B should spend steps. It also tells us whether healing duplicates swj (MLP) and #112 C9 (writers).
- **Fit:** diagnostic.
- **Test:** 4070, 4 arms × 2 K ≈ 2 h.
- **Gain/risk:** none directly.
- **Conflicts:** #108 (local joint refits fail in the full stack); end-to-end healing may differ.

**B3 · GDN conv1d taps as compensators.**
- **Mechanism:** Train the 4-tap depthwise conv after in_proj_qkv (40,960 per GDN layer, bf16, not quantized), initialized at bf16. It is a superset of a per-row scale with a temporal filter.
- **Source:** h1102-B4, untested.
- **Why:** These are free, unquantized parameters right after the worst-error GDN projection.
- **Fit:** per level (library-compatible: they are layer weights stored with the level's scales).
- **Test:** P08, arm "heal-all + conv" vs heal-all, 2 training seeds. Pass: geo ≤ 0.99 at K3 and K2, both seeds below 1, and g08-verify1 agrees.
- **Gain/risk:** −0.5…−2%. Risk: the conv weights become level-specific, so a mix gets one conv per layer (take it from the qkv level).
- **Conflicts:** none.

**B4 · GDN decay parameters A_log / dt_bias (48 + 48 per layer).**
- **Mechanism:** Train them with a small lr. They retune each head's memory length to stop noise accumulating in the state.
- **Source:** h1102-B5; STEPQuant 2609.38169 (per-head gate half-life vs error persistence, Spearman 0.80).
- **Why:** State-error persistence is set by these parameters, so shortening noisy heads' memory can trade a little modelling for less accumulated error on long windows.
- **Fit:** per level.
- **Test:** P08 add-on arm with 4k windows (needs E3). Same pass rule as B3, plus KL-by-position must not rise past 2k.
- **Gain/risk:** 0…−1%. Risk: hurts long context. Guard with M3.
- **Conflicts:** none.

**B5 · Full-weight training of in_proj_a / in_proj_b (bf16, 2 × 48 × 5120 per layer, 23.6M total).**
- **Mechanism:** These set β_t and the decay gate. Full freedom lets the state dynamics compensate upstream q/k/v error at zero bit cost.
- **Source:** h1102-B6.
- **Why:** These are the least quantization-sensitive GDN tensors (2609.04098: 2.1 / 2.6% layer error), so they have spare capacity. They stay bf16 in V3 (design §1).
- **Fit:** per level.
- **Test:** P08 arm. Pass as B3, plus overfit check: train vs held-out KL gap no worse than heal-all's.
- **Gain/risk:** −0.5…−2%. Risk: 23.6M parameters on 3M rows overfit (C6).
- **Conflicts:** #108 (GDN local refits overlapped H_out / asym in the stack). This is end-to-end, not local.

**B6 · Depth-tied low-rank log-scales.**
- **Mechanism:** Parametrize log c_{l,i} = Σ_k a_{l,k} b_{k,i} for each family (rank 2–4 across the 64 layers), plus a free per-layer scalar. Same for log r. Parameters fall from 5.7M to ≈0.3–0.6M.
- **Source:** new.
- **Why:** If the correction is coherent (the brief's hypothesis), it should be low-rank across depth. That cuts 27B overfitting, and the shared components could be fit on the 0.8B or 9B and transferred.
- **Fit:** per level.
- **Test:** P08 at rank 2 / 4 vs free, at full data and at ¼ data (C6). Pass if rank 4 keeps ≥90% of the free gain at full data and beats free at ¼ data.
- **Gain/risk:** 0% at full data; +1…+3% of retained gain under starvation. Risk: low.
- **Conflicts:** none.

**B7 · Residual-writer channel biases (format change: epilogue add).**
- **Mechanism:** Add an fp16 bias on the down / out_proj / o_proj outputs: 64 × 5120 × 2 B per family ≈ 0.66 MB, 0.0002 bpw. Initialize at the closed-form mean shift 0.85·E[y* − y_q] on bf16 inputs, then train by KL.
- **Source:** h1102-B9; LoopQuant LCTC (per-module gain + bias by KL); the shortlist's "writers' mean-shift bias".
- **Why:** Scales cannot fix a mean shift in the residual stream, and RMSNorm amplifies DC offsets in later readers.
- **Fit:** format change. Decide the exl3i bias plane before the ladder if wanted, but values come later.
- **Test:** P08 fake-quant bias. Pass: geo ≤ 0.99 at both K. Then a kernel epilogue cost check (should be ≈0).
- **Gain/risk:** −0.5…−2%. Risk: needs a plugin change and a selftest-matrix entry.
- **Conflicts:** none measured.

**B8 · Gate pre-activation shift (bias on gate_proj outputs, format change).**
- **Mechanism:** A per-intermediate-channel bias on gate before SiLU (17,408 per layer, 2.2 MB in total). It moves SiLU's operating point, which scales cannot.
- **Source:** QuantForge HiRes 2609.34680: MLP gate shift + scale + up scale cut Fit-3 damage ≈50% (W4A4); h1102-B10.
- **Why:** swj fits gains through SiLU but not shifts. Quantization of gate rows shifts the pre-activation distribution's mean, and SiLU is curved near 0.
- **Fit:** format change.
- **Test:** P08 fake-quant. Pass as B7.
- **Gain/risk:** −0.3…−1.5%. Risk: kernel change.
- **Conflicts:** #85 C10 / swj (scales only), so it is complementary.

**B9 · Full-attention emphasis: q/k norm per-dim gains plus v/o scales only.**
- **Mechanism:** For the 16 attention layers train only q_norm / k_norm gains (512 each) and the v / o scales, with a separate higher lr.
- **Source:** w06 (q/k norms trained), LAQuant 2605.08755 (E2E KL starves K/V directions).
- **Why:** The attention logit temperature after q/k quantization is a coherent error that a per-dim gain fixes cheaply.
- **Fit:** per level.
- **Test:** part of the B2 grid.
- **Gain/risk:** −0.2…−1%.
- **Conflicts:** #108 (attention local refits ≥1.00; q/k norms absorbed per-head scales). This may null too.

**B10 · Do not train the layer norms (negative carry-over).**
- **Mechanism:** Keep input/post-attention norms fixed. They are in the span of the readers' r, and on the 0.8B they gave ≤0.2%.
- **Source:** #89-02.
- **Why:** It saves gauge trouble at 27B.
- **Fit:** all.
- **Test:** none.
- **Gain/risk:** 0. Risk: none.
- **Conflicts:** #89-02.

**B11 · Rank-2 multiplicative scale field (research ceiling, format change).**
- **Mechanism:** W ⊙ (c₁r₁ᵀ + c₂r₂ᵀ). This is not foldable; it needs a second weight-plane pass or per-(row, 128-block) scales.
- **Source:** h1102-B13, Scale-QLoRA 2609.04526.
- **Why:** Only if B1 shows per-row/col granularity saturates.
- **Fit:** format change (V4).
- **Test:** P08 fake-quant ceiling only.
- **Gain/risk:** −0…−2%. Risk: kernel cost (E2: group scales −12…−29% GEMM).
- **Conflicts:** #110 (bf16 row-128 scales fail at equal bits); KL-trained might differ.

**B12 · Sparse healing: only tensors with a high gradient SNR.**
- **Mechanism:** After 64 steps, rank tensors by |mean grad| / std over windows on log c and log r. Freeze the bottom 50% at their refit values and continue.
- **Source:** new (overfitting control).
- **Why:** At 2 parameters per row on the 27B, noise-fit parameters cost held-out KL.
- **Fit:** per level.
- **Test:** P08 at ¼ data (C6). Pass if held-out KL ≤ heal-all's.
- **Gain/risk:** 0…+1% of retained gain.
- **Conflicts:** none.

## C. Initialization, optimizer, regularization

**C1 · Warm-start from closed-form coherent refits.**
- **Mechanism:** Initialize healing at swj (gate/up), C9 α (writers, #112: gdn_out α ≈ 3–4.75) and a norm-restoring gain c_i = ‖w_i‖_H / ‖w_q,i‖_H for the other tensors, instead of the LS `refit_scales` values.
- **Source:** #104, #112, #85. New: these become the healing init.
- **Why:** A rough cross-experiment product says swj still adds ~3% after healing (K3: ≈0.70 without swj vs 0.674 with; K2 ≈0.52 vs 0.507; confounded by asym ρ). Healing does not reach swj's point on its own in 512 steps. A better start should cut steps and raise the end point.
- **Fit:** per level.
- **Test:** P08 on the edy stack: heal from LS vs from swj+C9 vs from norm-restoring, 2 seeds, plus steps to reach 95% of the plateau. Pass: final ≤ 0.99 × LS-init, or the same final at ≤ 50% of the steps.
- **Gain/risk:** −0.5…−3%, or 2× fewer 27B steps.
- **Conflicts:** none.

**C2 · Heal from un-refit codes (refit_scales off) vs the LS refit.**
- **Mechanism:** Heal level K where the quantizer skipped `refit_scales` (scales = the RMS ones) and compare with LS-refit and swj starts.
- **Source:** the brief's LS-shrink finding.
- **Why:** If LS shrink is the coherent error and healing undoes it anyway, the refit is just an init. If the un-refit start heals better, the freeze could pin "no refit" for healed-only levels (−4–8% quant time).
- **Fit:** quantizer setting (pre-freeze relevant).
- **Test:** P08, requires requant (rounding noise), so 3 quant seeds × K3/K2 ≈ 3 h. Pass: heal(no-refit) ≤ heal(LS) with all seeds < 1.
- **Gain/risk:** 0…−2%. Risk: unhealed level options get worse; only matters if every shipped tensor is healed.
- **Conflicts:** none.

**C3 · A K-dependent lr rule.**
- **Mechanism:** Set lr(K) = lr₃ · (relative error at K / relative error at K3)^½. Error falls ≈2.0 bits per bit, so lr ≈ 1e-3 · 2^{−(K−3)} and K4 ≈ 5e-4. Fit the exponent from a P08 sweep at K2 / 3 / 4 / 5.
- **Source:** #89-06 (K4: lr 1e-3 never beat step 0; 3e-4 gave 0.949).
- **Why:** One recipe has to cover 71 levels. Today's fixed lr fails at K≥4.
- **Fit:** per level.
- **Test:** P08 lr grid {3e-4, 1e-3, 3e-3} × K {2, 3, 4, 5}, one seed ≈ 3 h. Pass: the fitted rule is within 1% of the best lr at each K.
- **Gain/risk:** prevents zero-gain heals above K3.5. Risk: none.
- **Conflicts:** #89-06.

**C4 · Diagonal Gauss–Newton steps in place of 512 Adam steps.**
- **Mechanism:** From one pass over N windows, accumulate g and the per-parameter diagonal Fisher F = Σ_t (∂ℓ_t/∂θ)². These come from the same backward with per-token squaring of scale gradients, which is cheap because scale gradients are per-channel reductions. Step θ ← θ − g / (F + λ), with λ chosen by held-out line search. Repeat 2–4 times.
- **Source:** h1102-C3 (Fisher-preconditioned Adam); this is a closed-form variant.
- **Why:** The scale parameters are nearly decoupled with a well-conditioned loss (max |log| 0.07–0.12, far inside the trust region). A few Newton steps could replace hundreds of Adam steps: ~5× cheaper 27B heals and library polish.
- **Fit:** per level / polish.
- **Test:** P08 K3/K2: 3 GN steps × 256 windows vs Adam 512 × 4. Pass: ≥90% of Adam's gain at ≤25% of the windows.
- **Gain/risk:** cost only (5×). Risk: per-token Fisher needs token-level gradients of the scales (extra memory ≈ tokens × params per layer, so chunk it).
- **Conflicts:** none.

**C5 · Bigger accumulation, fewer steps (data-parallel friendly).**
- **Mechanism:** accum 16 instead of 4 at 2× lr, equal tokens.
- **Source:** QATFactory 2609.39223 (batch 16 at 8K).
- **Why:** The fewer the steps, the cheaper A4's synchronization and the fewer the evals.
- **Fit:** per level.
- **Test:** P08 two arms. Pass: within 1% of accum 4.
- **Gain/risk:** cost.
- **Conflicts:** none.

**C6 · 27B data-starvation emulation and a regularizer.**
- **Mechanism:**
  - On P08, heal with ¼ and ⅛ of the windows (256 / 128) to mimic the 27B's 4–8× parameters per row.
  - Compare regularizers: none / L2 toward the init (λ·Σ log²) / a Hessian-proxy anchor (h1102-B14) / B6 tying / B12 sparsity.
  - Rule: held-out early stop on qad windows.
- **Source:** new framing; h1102-B14.
- **Why:** It predicts whether the 27B heal is data-limited before the first 27B run.
- **Fit:** diagnostic + recipe.
- **Test:** P08 ≈ 2.5 h. Decision: if ¼ data keeps ≥80% of the full-data gain unregularized, ignore; otherwise adopt the best regularizer.
- **Gain/risk:** protects 1–3% at 27B.
- **Conflicts:** #89-04 (one pass vs two epochs was equal, so dose is not limiting on the 0.8B).

**C7 · Separate lr for svh vs suh, and per family.**
- **Mechanism:** A 3 × 3 lr grid (svh, suh, norm/q-k).
- **Source:** h1102-C10; SQuAT 2610.31412 (gain lr 1000× the shape lr).
- **Why:** Input-side scales of high-fan-in tensors (down: 17,408 inputs) move every output and may need a smaller lr.
- **Fit:** recipe.
- **Test:** P08 one seed, ≈2 h.
- **Gain/risk:** −0…−1%.
- **Conflicts:** none.

**C8 · EMA / Polyak averaging of the scales.**
- **Mechanism:** Ship the EMA (β 0.99) of log-scales.
- **Source:** h1102-C9.
- **Why:** The val curves are noisy step to step (train KL swings ±15% per eval).
- **Fit:** recipe.
- **Test:** free side output of any P08 run.
- **Gain/risk:** 0…−0.5%.
- **Conflicts:** none.

**C9 · fp32 master scales and optimizer state.**
- **Mechanism:** Keep log-scales and Adam state in fp32 even if the forward is bf16.
- **Source:** QAR 2609.21834 (bf16 shadow updates stall below the bf16 gap).
- **Why:** At K4+ updates are tiny (#89-06's flat lr-1e-3 run could partly be this).
- **Fit:** recipe.
- **Test:** check heal.py; one P08 K4 arm.
- **Gain/risk:** enables K≥4 heals.
- **Conflicts:** #89-06.

## D. Losses

**D1 · Prompt-row weighting against data starvation.**
- **Mechanism:** Loss = gen rows + w·prompt rows, w ∈ {0, 0.1, 0.3}. Tool outputs in agentic traces are non-gen rows with a valid teacher.
- **Source:** h1102-D2. New reason: the 27B's parameter density.
- **Why:** More rows means less noise per parameter, while the target is still the scored distribution.
- **Fit:** recipe.
- **Test:** P08 at ¼ data (C6) and full data. Pass: gen-row search KL ≤ 0.995 × w=0 at ¼ data and not worse at full.
- **Gain/risk:** 0…−1.5% at 27B. Risk: shifts capacity toward prompt-style text.
- **Conflicts:** #83 (gen-mask helped H_out; that was a statistic, this is a target).

**D2 · Hidden-state matching at 4 depths, auto-balanced.**
- **Mechanism:** Add Σ_d ‖h_d^q − h_d‖² / ‖h_d‖² at layers {15, 31, 47, 63}, weight sg(L_KL / L_aux). On the 27B, teacher states come from the A8 cache.
- **Source:** backlog item; CKA-QAD 2606.05682 (last-layer CKA 0.74 under KL-only QAD); NVIDIA Flash-Next.
- **Why:** It gives early layers a short-path target, which matters if the 27B early-layer gradients are noisy (A5).
- **Fit:** recipe.
- **Test:** P08, 2 seeds. Pass: verify KL ≤ 0.99 × KL-only at both K (KL is the metric, so the aux term must pay in KL).
- **Gain/risk:** −0…−2%. Risk: metric mismatch (BiSCo: local losses can hurt the whole model).
- **Conflicts:** #92 REAL-Q (block-output Fisher) failed; that was a rounding objective, this is a training auxiliary.

**D3 · K/V-fidelity auxiliary for the 16 full-attention layers.**
- **Mechanism:** Add a normalized MSE between the student's and teacher's k_proj / v_proj outputs, weight auto-balanced.
- **Source:** LAQuant 2605.08755 (E2E KL attenuates K/V directions: σ_max(H_KL) ≤ ½; E2E + K-cache RMSE +1.6 avg).
- **Why:** On long contexts the KV error of the 16 attention layers compounds across positions, but per-token KL rarely sees it.
- **Fit:** recipe.
- **Test:** P08 with 4k windows, KL by position. Pass: KL beyond 2k ≤ 0.98 × no-aux, and overall not worse.
- **Gain/risk:** 0…−1%, more on long traces.
- **Conflicts:** none.

**D4 · GDN output auxiliary (proxy research).**
- **Mechanism:** Add an MSE of each GDN layer's pre-out_proj output (o_t) vs the online 0.8B teacher, a proxy for state fidelity.
- **Source:** STEPQuant / LeapQuant (state error persists; long-generation collapse).
- **Why:** Hypothesis: state-sensitive error is under-weighted at the logits.
- **Fit:** recipe (27B only if P08 pays; then teacher states are needed at 48 layers, which is too big, so use 4 layers).
- **Test:** P08 arm. Same pass as D3.
- **Gain/risk:** 0…−1%. Risk: low prior.
- **Conflicts:** none.

**D5 · RMSE-normalized block losses for any block-wise stage.**
- **Mechanism:** If F3 is used, use per-block RMSE (√MSE) instead of MSE so every block gets an equal gradient magnitude.
- **Source:** 2610.00983 (Qwen3.5-27B OmniQuant W3A16 8.42 → 8.26; W2A16 37.5 → 22.2).
- **Why:** Block losses span orders of magnitude across depth, so early blocks starve.
- **Fit:** F3 recipe.
- **Test:** part of F3.
- **Gain/risk:** improves the F3 init.
- **Conflicts:** none.

**D6 · Dynamic forward/reverse KL mix.**
- **Mechanism:** α_v = sg[p_v / (p_v + q_v)] on top-k ∪ tail.
- **Source:** OnQ 2610.29378 (on-policy DKL 67.2 vs FKL 66.2 MATH); h1102-D3.
- **Why:** Possibly faster optimization of the two-sided errors.
- **Fit:** recipe.
- **Test:** P08 arm. Adopt only if *forward* KL improves (≤ 0.995).
- **Gain/risk:** 0…−1%. Risk: optimizes the wrong metric.
- **Conflicts:** none.

**D7 · Teacher-temperature warm-up.**
- **Mechanism:** T = 2 → 1 over the first 20% of steps.
- **Source:** h1102-D8.
- **Why:** It spreads early gradient over tail rows, where trunk quantization leaks mass.
- **Fit:** recipe.
- **Test:** P08 arm.
- **Gain/risk:** 0…−0.5%.
- **Conflicts:** none.

**D8 · Per-token robust clipping (tail guard).**
- **Mechanism:** Clip the per-token KL contribution at the batch p99.9 during training only.
- **Source:** h1102-D5.
- **Why:** A few spiky rows (the 27B Σdy² had heavy-tailed blocks, #33-27) dominate gradients of a small parameter set.
- **Fit:** recipe.
- **Test:** P08. Adopt only if mean KL improves.
- **Gain/risk:** 0…−0.5%.
- **Conflicts:** #33-27.

**D9 · Position-mass weighting.**
- **Mechanism:** Reweight tokens so the trained position histogram matches the scored corpus's.
- **Source:** h1102-D6.
- **Why:** Doc-start windows over-weight positions 0–2k.
- **Fit:** recipe (with E7).
- **Test:** P08 KL by position bin.
- **Gain/risk:** 0…−0.5%, more on long traces.
- **Conflicts:** none.

## E. Data and windows

**E1 · Data dose at 27B, plus the stopping signal.**
- **Mechanism:** One 27B K2.5 heal with checkpoints at 512 / 1024 / 2048 / 3,300 windows (the whole qad pool), held-out KL on qad holdout and search.
- **Source:** h1102-E4; #89-04.
- **Why:** It fixes the 27B per-level budget (cost ×1–3) and shows starvation directly.
- **Fit:** recipe.
- **Test:** gpubox, one run ≈ 4–5 h (needs A2).
- **Gain/risk:** decision.
- **Conflicts:** #89-04.

**E2 · Fit-pool reuse to double healing data.**
- **Mechanism:** Add fit-pool documents that are in no golden slice (the V2 kfit2 fit slices are spent for selection, not training) after repeating #55's reuse-bias test on a healed 0.8B.
- **Source:** h1102-E10, #55 (reuse bias ≤ 2%).
- **Why:** About 2× data for the 27B's starvation.
- **Fit:** data policy (user call).
- **Test:** P08: qad vs qad+fit, scored on search and verify. Pass: verify ≤ 0.99.
- **Gain/risk:** −0.5…−2% at 27B. Risk: pool hygiene; the user must approve.
- **Conflicts:** #55.

**E3 · Window length at equal tokens.**
- **Mechanism:** 2k / 4k / 8k on P08, and 2k vs 8k at 27B (memory in A2: 14 / 20 GB).
- **Source:** h1102-E1; QATFactory (32K vs 4K +1.9 acc at equal tokens).
- **Why:** The score includes positions up to 250k; the field report used 1k.
- **Fit:** recipe.
- **Test:** P08 3 arms ≈ 1.5 h. Pass: 4k or 8k ≤ 0.99 × 2k on search, and better KL beyond 2k.
- **Gain/risk:** −0…−3%.
- **Conflicts:** none.

**E4 · Truncated BPTT through the GDN state on the two long qad traces.**
- **Mechanism:** Chunk the attrs / click traces into 8k pieces. Carry the GDN state and attention KV detached across chunks so trained positions reach 100k+.
- **Source:** h1102-E2. The qad long traces are attrs and click (LOG 11-05).
- **Why:** Healing never sees deep-position statistics otherwise.
- **Fit:** recipe.
- **Test:** P08 then 27B. Pass: golden-long KL by position ≤ 0.98 beyond 32k, overall not worse.
- **Gain/risk:** 0…−1% mean; more on agentic traces.
- **Conflicts:** none.

**E5 · Prefix burn-in windows: loss deep in documents for forward-only cost.**
- **Mechanism:** Pick windows at random offsets inside long docs. Run the preceding up-to-8k tokens with no grad and no checkpoints (forward only) to build GDN state and KV, then train on the next 2k rows. The forward cost of the prefix is about ⅓ of a trained window per 2k.
- **Source:** new.
- **Why:** It matches the scored position distribution (agentic traces are ~25% of scored rows but 6% of qad rows) without paying backward memory.
- **Fit:** recipe.
- **Test:** P08 vs doc-start windows at equal trained rows. Pass: search ≤ 0.995 and long-position bins ≤ 0.98.
- **Gain/risk:** −0.5…−2%.
- **Conflicts:** none.

**E6 · Category- and agentic-share matching.**
- **Mechanism:** Sample qad windows to match the corpus's category token shares and the agentic share of scored rows.
- **Source:** h1102-E5; #33-24 counts (agent traces 18% of windows, 6% of scored rows).
- **Why:** Healing is a fit to the data distribution.
- **Fit:** recipe.
- **Test:** P08. Pass: search ≤ 0.995, no category > 1.01.
- **Gain/risk:** 0…−1%.
- **Conflicts:** #89-05 (xcorp mix worse; this keeps calib only).

**E7 · Tail-KL window selection (KIC).**
- **Mechanism:** Score qad windows by the unhealed model's top-25% token KL and oversample the top half in pass 1.
- **Source:** FidelityQ 2610.14934 (1.45× enrichment, ≈2% PPL on 8B); h1102-D4.
- **Why:** It puts more gradient where the damage is.
- **Fit:** recipe.
- **Test:** P08. Pass: ≤ 0.99.
- **Gain/risk:** 0…−1%.
- **Conflicts:** none.

**E8 · Never self-generated or own-session text (keep the negative).**
- **Mechanism:** Stay on calib qad.
- **Source:** #91 (+10% calib cost); #89-05.
- **Fit:** policy.
- **Gain/risk:** avoids a −10% mistake.
- **Conflicts:** #91, #89-05.

## F. Depth, curricula and cheap 27B gradient pilots

**F1 · First 27B gradient pilot: truncated backward on the last 8 layers.**
- **Mechanism:** On the K2.5 level with A1's cached teacher, run the no-grad forward through layers 0–55 (decode on the fly, no checkpoints) and backprop only through 56–63 plus the head-side scales. Memory ≈ codes + 8 checkpoints ≈ 10 GB, est. ≈2 s per window.
- **Source:** w06 H1b (last quarter = 41–45% of the gain); the field report.
- **Why:** It is the first real 27B healing number, and it can come before the full S27 backward is stable.
- **Fit:** per level.
- **Test:** gpubox card 0, 1,024 windows ≈ 1 h + 2 evals. Read: ratio vs unhealed on golden-search27, head-matched (the #104 lesson).
- **Gain/risk:** −1…−4% (≈⅓ of full). Risk: low.
- **Conflicts:** none.

**F2 · 27B depth replication: all / last 16 / first 16.**
- **Mechanism:** Three heals of the same K2.5 level at equal steps.
- **Source:** w06 H1b (0.8B: all layers), the field report (27B: last 16).
- **Why:** It settles whether the 0.8B depth result transfers. Combined with A5 it explains the discrepancy.
- **Fit:** recipe.
- **Test:** gpubox ≈ 3 × 3 h, or 3 × 1.5 h on two cards.
- **Gain/risk:** decision. If last-16 ≥ all at 27B, the 27B backward cost falls to ~40%.
- **Conflicts:** w06 H1b.

**F3 · Block-wise teacher-forced healing via sib27 streaming, then a short E2E pass.**
- **Mechanism:**
  - Per layer, read the bf16 boundary input and the bf16 teacher output from the A8 cache.
  - Train that layer's scales to minimize the residual-write error, weighted per channel by the 27B Σdy² diagonal of the writer outputs (already collected), with RMSE normalization (D5).
  - Every layer is an independent job (donor-friendly, ≈1–2 min of compute per layer).
  - Then run E2E KL for ¼ of the budget.
- **Source:** EfficientQAT / JustQuant 2609.33601 (local→global curriculum); h1102-G5/G6; #33-24 diagonal.
- **Why:** Most of the movement happens cheaply and in parallel; E2E only polishes.
- **Fit:** per level.
- **Test:** P08: block init + 128 E2E steps vs 512 E2E steps. Pass: ≥95% of the gain at ≤50% of the compute.
- **Gain/risk:** cost ÷2. Risk: block-local fits overlapped in #108.
- **Conflicts:** #108 (local junction refits fail in the full stack). Here they are only an init, and E2E decides the end point.

**F4 · Late-first, then all layers.**
- **Mechanism:** Steps 0–25% train layers ≥48 only (cheap truncated backward), then all layers.
- **Source:** new combination of w06 H1b and the field report.
- **Why:** If late-layer correction is what absorbs accumulated drift (2609.11716: blocks counteract inherited error), fixing the tail first lets early layers fit a cleaner downstream.
- **Fit:** recipe.
- **Test:** P08 vs plain all, equal compute. Pass: ≤ 0.99.
- **Gain/risk:** −0…−1.5%, plus ~15% compute saved.
- **Conflicts:** none.

**F5 · Adaptive backward truncation.**
- **Mechanism:** Every 64 steps, freeze the lowest layer group whose held-out gradient norm fell below 10% of its initial value, and stop the backward there.
- **Source:** new; Layer-wise curriculum 2609.19213 (progressive freezing).
- **Why:** It cuts late-step cost when early layers converge first.
- **Fit:** recipe.
- **Test:** P08: wall time to 98% of the gain.
- **Gain/risk:** cost −20…−40%.
- **Conflicts:** none.

**F6 · Depth-dependent lr.**
- **Mechanism:** lr_l ∝ 1/√(kfit2 sensitivity of layer l), or ∝ the measured Fisher of its scales.
- **Source:** h1102-C2.
- **Why:** Early-layer error compounds (kfit2 κ ≈ 38 at layer 0).
- **Fit:** recipe.
- **Test:** P08.
- **Gain/risk:** 0…−1%.
- **Conflicts:** none.

## G. Library, mixes and the knapsack

**G1 · Per-level library healing for the pass-1 levels, plus a per-mix polish (the default V3 plan).**
- **Mechanism:** Heal each of the ~25 pass-1 levels uniformly (all 64 layers at that K). radmix carries the healed suh/svh per tensor. Each shipped mix gets a ¼-budget polish.
- **Source:** w06 H4 (LIBRARY: transfer 0.79 / 0.85, polish 0.96 / 0.97).
- **Why:** The knapsack and the search then see healed options for free, and only 2–3 sizes × 2 variants pay polish.
- **Fit:** per level + final pass.
- **Cost:** 25 × 2.5–4 h ≈ 60–100 card-h (donors via A7), plus polish ≈1 h per shipped mix.
- **27B test:** heal the K2.5 and K3 levels, build a 2.75 mix from them, measure transfer share vs a mix-specific heal. Pass: share ≥ 0.7, transferred mix z ≤ −3.
- **Gain/risk:** −5…−11% at 2.5 bpw (central −7%). Risk: transfer may be lower at 64 layers.
- **Conflicts:** #89 (deferred by the user to the ladder phase; this is that plan).

**G2 · Two-level stochastic library healing (mix-randomized).**
- **Mechanism:** Train the scales of two adjacent levels (K_a, K_b) jointly. Each window samples a per-tensor K ∈ {K_a, K_b} (p = ½), so each (tensor, K) scale learns to work next to either neighbour.
- **Source:** h1102-I6 (supernet), simplified to two levels.
- **Why:** It raises the transfer share from 0.79–0.85 toward ~0.95 without per-mix polish, or makes the polish nearly unnecessary.
- **Fit:** per level.
- **Test:** P08 K2+K3 joint heal (≈2× one heal), then evaluate on mixF / mixD with no polish. Pass: share ≥ 0.92 in both mixes.
- **Gain/risk:** −0.5…−2% over G1 without polish. Risk: each level's uniform-context quality drops slightly.
- **Conflicts:** none.

**G3 · Context-mix library: heal level K inside its size's mix.**
- **Mechanism:** Heal each level inside a fixed "context mix" (the knapsack pick for the size where that level is used most) instead of uniform K. It is cheaper than G2 and closer to deployment.
- **Source:** new.
- **Why:** The scale corrections depend on upstream error levels; the uniform context is wrong for all but one K.
- **Fit:** per level.
- **Test:** P08 with a depth mix as context. Pass: share ≥ 0.9.
- **Gain/risk:** −0.5…−1.5% over G1.
- **Conflicts:** none.

**G4 · Interpolate healed log-scales across K.**
- **Mechanism:** Heal only integer and half-integer levels. For a level at K between them, use log c(K) interpolated linearly in 2^{−2K} between the neighbours' corrections, per tensor. The codes differ but the coherent correction should be smooth.
- **Source:** new.
- **Why:** It cuts the number of 27B healing jobs 3–4×.
- **Fit:** per level.
- **Test:** P08: heal K2 and K3 (have them) and a fresh K2.5. Interpolate to K2.5 and compare with direct K2.5 healing. Pass: ≥85% of the direct gain. Cost ≈1 h (one heal + 2 evals).
- **Gain/risk:** cost only.
- **Conflicts:** none.

**G5 · Parametric K-law for corrections.**
- **Mechanism:** Fit log c_i(K) = a_i + b_i · 2^{−2K} per tensor channel from 2–3 healed levels and predict every other level. It is G4 with extrapolation, which covers sub-2 and K5+ levels.
- **Source:** new.
- **Why:** The low-bpw research ladder and the K5–7 levels never get their own heal.
- **Fit:** per level.
- **Test:** P08 K2 / K3 / K4 → predict K5 and K1.5. Pass: ≥75% of the direct gain.
- **Gain/risk:** cost.
- **Conflicts:** none.

**G6 · Healed sensitivities in knap4.**
- **Mechanism:** With library healing, re-measure ~50 group-swap sensitivities on a healed base. Or fit a healability factor h(family, K) = healed / unhealed from the uniform-level heals and multiply the option costs. Re-solve and heal the new pick.
- **Source:** h1102-I3; the 0.8B healing gain is larger at K2 (0.72) than at K3 (0.886), so low-K options get relatively cheaper.
- **Why:** Bits move to spike tensors.
- **Fit:** knapsack.
- **Test:** 27B: two healed mixes at 2.5 (old pick vs re-solved pick) on golden-search27, head-matched. Pass: new ≤ 0.99. The 0.8B prior is mixed: the mixF/mixD order survived healing and the gap grew (3.2% → 6.1%).
- **Gain/risk:** −0.5…−2%.
- **Conflicts:** kfit2 / #45.

**G7 · Heal the top-3 finalists per size, pick on healed search KL.**
- **Mechanism:** Polish-heal the top 3 stream candidates per size, choose by healed search-slice KL, confirm on ver4.
- **Source:** h1102-I4/I5.
- **Why:** The healed ranking can differ from the unhealed one.
- **Fit:** final pass.
- **Test:** 27B, ≈3 polish runs per size (≈3 h).
- **Gain/risk:** 0…−2%.
- **Conflicts:** none.

**G8 · Polish budget sweep.**
- **Mechanism:** Polish with 32 / 64 / 128 steps from library scales.
- **Source:** w06 (128 steps → 0.96).
- **Why:** It sizes the per-mix cost for many finalists.
- **Fit:** final pass.
- **Test:** P08 on mixF / mixD, ≈30 min.
- **Gain/risk:** cost.
- **Conflicts:** none.

**G9 · Heal in order of absolute KL.**
- **Mechanism:** Run library heals in this order: levels used by 2.0–2.5 first, then 3.0, then 3.5–4.5. Skip K ≥ 5 unless G5 predicts them.
- **Source:** h1102-I8; 2609.32060 (QAT benefit fades as precision rises); #89-06 (K4 ≈ −5%).
- **Why:** Healing value falls steeply with K.
- **Fit:** scheduling.
- **Test:** none.
- **Gain/risk:** cost allocation.
- **Conflicts:** none.

**G10 · Healing-gradient features for local search.**
- **Mechanism:** At a healed mix, the per-tensor residual ‖∂KL/∂log c‖ ranks upgrade candidates.
- **Source:** h1102-I7.
- **Why:** It is a cheap proposal signal.
- **Fit:** search.
- **Test:** compare the beat rate of gradient-ranked vs random proposals over 50 search evals.
- **Gain/risk:** 0…−1%.
- **Conflicts:** none.

## H. Interplay with the adopted and running quantizer stack

**H1 · Clean swj-vs-healing overlap, and whether to freeze gate/up svh.**
- **Mechanism:** On P08 with identical asym ρ0, run (a) edy → asym → heal, (b) edy → swj → asym → heal (exists, d1104), (c) like (b) but with gate/up svh frozen during healing.
- **Source:** #85-03, #87-08, #89-03.
- **Why:** If (a) ≈ (b), swj only matters for unhealed options. If (c) ≥ (b), healing should not touch gate/up, which saves 2 × 17,408 × 64 parameters.
- **Fit:** recipe.
- **Test:** 4070 ≈ 2.5 h, 2 seeds.
- **Gain/risk:** −0…−1%, and it settles C1.
- **Conflicts:** #104.

**H2 · Asym-writer order relative to healing.**
- **Mechanism:** Compare (a) asym → heal-all (today) vs (b) heal readers → recompute sibling moments on the healed readers → asym writers → heal writers' scales only. #87-07 shows the asym correction is specific to the exact sibling state.
- **Source:** #87-07, #87-08.
- **Why:** In (a) healing changes the readers after the writers were fit to them.
- **Fit:** per mix (asym is per mix after #111).
- **Test:** P08 2 seeds (code change → 3 seeds) ≈ 3 h. Pass: (b) ≤ 0.98 × (a).
- **Gain/risk:** −0…−2% (only if asym ships).
- **Conflicts:** #87-07, #111.

**H3 · Plain writers + library heal + polish vs per-mix asym writers + heal.**
- **Mechanism:** On P08 mixes like #111's (3 on 2.5, 3 on 3.5): compare (i) plain writers + library heal + 128-step polish, (ii) per-mix writer pass + library heal + polish, (iii) per-level asym + library heal + polish.
- **Source:** #111 (per-level asym fails ×1.08–1.24 under mixed siblings; per-mix pass 0.81–0.90).
- **Why:** It informs the user's pending rollout decision. If (i) is within 2% of (ii), the 4–5 card-h per mix writer pass and its engineering can be skipped, because healing recovers it.
- **Fit:** per mix.
- **Test:** 4070 ≈ 3 h (3 mixes × 3 arms × 1 seed + 2nd seed on the best two). Pass for (i): ≤ 1.02 × (ii).
- **Gain/risk:** saves ~10–15 card-h per release.
- **Conflicts:** #111, #105.

**H4 · Use the #112 C9 writer α as healing input, not as a separate stage.**
- **Mechanism:** If the 27B C9 pair passes, put α into the C1 init. Healing then refines per-row.
- **Source:** #112.
- **Why:** gdn_out α ≈ 3–4.75 is far from LS, so a large coherent move should be the init, not something healing has to discover.
- **Fit:** per level.
- **Test:** part of C1.
- **Gain/risk:** fewer steps.
- **Conflicts:** #112.

**H5 · On-model Σdy² as a free by-product of healing.**
- **Mechanism:** During healing backward passes, accumulate Σ_t dy_{t,i}² at every linear of the deployed quantized model (no extra GEMM). Requant the 16 late layers of a shipped mix with this H_out and re-polish.
- **Source:** h1102-H2/H3; Q-Strata / MixQuant (sensitivities in the compressed context beat ones at the FP point).
- **Why:** H_out at bf16 was a hair under the adoption rule at K2.5 (0.980); H_out measured at the operating point may pay where the bf16 one did not.
- **Fit:** per mix.
- **Test:** 27B: collect during one polish (free), requant layers 48–63 with H_out (≈20 min at ~178 s per layer), polish, compare. Pass: ≤ 0.985.
- **Gain/risk:** −0…−2%.
- **Conflicts:** #33-28 (not adopted ≤ K3.5); different H_out source. No new bf16 Σdy² collection, per the user's 11-05 rule.

**H6 · Healing at H_out levels (K ≥ 4): small lr, low priority.**
- **Mechanism:** Use C3's lr. Expect −2…−5%; schedule after the low-K levels.
- **Source:** #89-06, #33-28 K4.
- **Fit:** per level.
- **Test:** part of G9.
- **Gain/risk:** −2…−5% at 4.0 bpw.
- **Conflicts:** #89-06.

## I. Code refresh (V3 optional, mostly V4)

**I1 · Scale-conditioned re-encode (one EM step).**
- **Mechanism:** After healing, re-run LDLQ + trellis per tensor with the healed suh/svh fixed (no refit), then polish.
- **Source:** h1102-F1.
- **Why:** Codes chosen for the healed input scaling see a different rotated target. Healed scales moved 7–12%, which is enough to change many codes.
- **Fit:** per level (re-quantize the level ≈37 min plus a re-heal).
- **Test:** P08, 3 quant seeds (codes change). Pass: ≤ 0.985 with all seeds < 1.
- **Gain/risk:** −0…−3%.
- **Conflicts:** none.

**I2 · PV-lite: re-Viterbi the worst tiles toward a KL-gradient target.**
- **Mechanism:** Accumulate G_W per tensor (one dW GEMM per tensor, accumulated per layer to host). Target W* = W_q − η·G scaled by the H_in diagonal. Re-encode only the 5–10% of tiles whose target moved most; accept by held-out KL.
- **Source:** h1102-F2; PV-Tuning, TVQ 2610.41822.
- **Why:** It moves codes where the loss wants them, which scales cannot do.
- **Fit:** per level.
- **Test:** P08, 3 seeds. Pass: ≤ 0.985.
- **Gain/risk:** −0…−3%. Risk: engineering; host RAM for G_W on the 27B (do it per layer).
- **Conflicts:** none.

**I3 · TVQ-lite STE latent healing on late layers (V4).**
- **Mechanism:** Keep a bf16 latent in the rotated domain for layers 56–63, re-Viterbi every 16 steps, identity STE, fp32 master.
- **Source:** TVQ 2610.41822 (≤0.2 PPL over QTIP at 7–8B, none at 70B, 8× cost); E8-QAT 2610.27699; BCJR-QAT (drift budget).
- **Why:** It is the only route to moving codes at scale.
- **Fit:** training (V4).
- **Test:** P08 last 4 layers on the healed stack. Pass: ≤ 0.98.
- **Gain/risk:** −0…−3% at high cost.
- **Conflicts:** none.

**I4 · Requant late layers with on-model H_in (GPTAQ-style) after healing.**
- **Mechanism:** Collect H_in for layers 48–63 on the healed mix's own inputs, with targets via the cross-covariance to bf16 inputs. Requant those layers and re-polish.
- **Source:** h1102-F5.
- **Why:** H_in measured in the deployed context.
- **Fit:** per mix.
- **Test:** P08 3 seeds.
- **Gain/risk:** −0…−2%.
- **Conflicts:** #88-02 (all-tensor asym 1.5–2.3× worse). Restrict to late layers and re-heal; the prior is weak.

**I5 · Drift-budget diagnostic before any code-refresh work.**
- **Mechanism:** Per tensor at K2 / K3 / K4, compare the trellis cell radius in the rotated domain with the expected latent drift lr·N·g_max, and estimate the fraction of codes that would flip in I3.
- **Source:** BCJR-QAT 2605.10655; h1102-F9.
- **Why:** If <0.01% would flip, I3 is dead before engineering.
- **Fit:** diagnostic.
- **Test:** CPU on colo, ≈20 min.
- **Gain/risk:** decision.
- **Conflicts:** none.

## J. Forward-only 27B healing, with no S27

**J1 · Global logit temperature τ by container edit.**
- **Mechanism:** Multiply the head svh of a K2.5 swj container by τ ∈ {0.97, 0.99, 1.01, 1.03} and score on golden-search27, head-matched.
- **Source:** h1102-A4, never run.
- **Why:** Temperature is the simplest coherent error. Trunk quantization often flattens or sharpens the output distribution.
- **Fit:** per mix (one scalar).
- **Test:** gpubox, 4 evals ≈ 20 min. Pass: τ* gives ≤ 0.995 and verify agrees.
- **Gain/risk:** 0…−1%.
- **Conflicts:** none.

**J2 · SPSA over about 128 coherent scalars.**
- **Mechanism:** Parameters: per layer one log-gain for its readers and one for its writers (128), applied by svh edits on the container. Use simultaneous-perturbation gradient estimates with radiance evals on a fixed 40k-row qad-holdout slice. Fixed codes mean the paired noise is tiny. 100 iterations × 2 evals ≈ 200 × 1.5 min ≈ 5 h.
- **Source:** new; motivated by B1.
- **Why:** If B1 shows coarse scalars carry ≥50% of healing, this gives a 27B healed number weeks before S27.
- **Fit:** per level / per mix.
- **Test:** gpubox ≈ 5 h. Pass: ≤ 0.98 on search, and verify agrees.
- **Gain/risk:** −1…−4% (gated by B1).
- **Conflicts:** none.

**J3 · Forward-only sequential output-matching scale refit (GPTAQ for scales).**
- **Mechanism:** Stream the mix layer by layer. For each linear, the inputs come from the quantized-and-refit prefix and the target is the bf16 output. Do a closed-form per-row and per-column ridge solve, Σdy²-weighted (writers' diagonals exist).
- **Source:** h1102-G3.
- **Why:** It needs no backward and is a sib27-style job.
- **Fit:** per mix.
- **Test:** P08 first. Pass: ≤ 0.98 vs the stack.
- **Gain/risk:** −0…−2%.
- **Conflicts:** #108 (local refits overlap H_out / asym in the full stack) and #87 (asym codes do this for writers). Low prior; test only if B1 shows large coarse-scale shares.

**J4 · Head-only KL healing on dumped head inputs.**
- **Mechanism:**
  - Dump 0.5–1M final-norm hidden states of a shipped mix on qad windows (one radiance pass with a hook).
  - On one card, train the head's row scales (248,320), τ and the head's input-side suh (5,120) by exact softmax KL against the cached top-k teacher.
  - The head decoded in bf16 is 2.5 GB; inputs stream from disk.
- **Source:** h1102-K1; SoftWater 2608.12026 (class-aware head rate ×6.5–8.3 KL at 2-bit heads); Softmax reparameterization 2609.31291 (Qwen near 0 gain with GPTQ heads).
- **Why:** The head is ~5–7% of total KLD (design §6b). It is mix-dependent but cheap.
- **Fit:** per mix.
- **Test:** gpubox ≈ 1–2 h + 2 evals. Pass: ≤ 0.99.
- **Gain/risk:** −0.5…−1.5%.
- **Conflicts:** #34 (slab head; complementary).

## K. Head and embedding

**K1 · Re-price the slab head after trunk healing.**
- **Mechanism:** After library healing, rerun headsim on the healed model's head inputs and re-solve the head's K and slab plan.
- **Source:** h1102-K4.
- **Why:** Healing changes the head inputs' error profile.
- **Fit:** knapsack.
- **Test:** CPU / GPU ≈ 1 h.
- **Gain/risk:** 0…−0.5%.
- **Conflicts:** #34.

**K2 · Embedding row-scale healing (9B pilot).**
- **Mechanism:** Train the trellis embedding's per-row scales from the layer-0 input gradient, for rows seen in qad only; unseen rows stay at their static values (#32's lesson).
- **Source:** backlog "embedding QAD"; h1102-K3.
- **Why:** The embedding counts in bpw and is low-bit at the small sizes.
- **Fit:** per level (embedding levels).
- **Test:** 9B (untied; the 0.8B / 2B are tied) with heal.py plus a cached teacher (A1) on an R9700 ≈ 1 h. Pass: ≤ 0.995, golden-ml ≤ 1.0.
- **Gain/risk:** −0…−1% at ≤ 2.5 bpw.
- **Conflicts:** #32.

**K3 · Logit bias (format change, 0.5 MB).**
- **Mechanism:** Add a 248k fp16 bias on the logits, learned in J4 with L2, seen rows only.
- **Source:** h1102-K2.
- **Why:** It corrects frequency-level shifts.
- **Fit:** format change.
- **Test:** within J4.
- **Gain/risk:** 0…−0.5%.
- **Conflicts:** none.

**K4 · Head shift search (softmax gauge).**
- **Mechanism:** Before quantizing the head, shift all rows by t·μ, with t chosen by post-quantization KL.
- **Source:** 2609.31291.
- **Why:** It is free if the logit path is linear.
- **Fit:** per level (head quant).
- **Test:** headsim, CPU ≈ 1 h.
- **Gain/risk:** ≈0 for Qwen with GPTQ-class heads (paper's own result); low priority.
- **Conflicts:** #34 mean-row (t=1 already).

## L. Speed variants and activations

**L1 · A8-aware healing for the fast variant.**
- **Mechanism:** Train with A8-g128 fake-quant (STE) on every tensor the fast variant runs at A8. Each variant ships its own healed scales; they are separate containers anyway.
- **Source:** h1102-J1; QATFactory (training through the deployed format).
- **Why:** The fast variant's extra KL from A8 is ≈+0.00005 at K6 (≈5% there) and more at low K. Healed scales can absorb part of it.
- **Fit:** per level (fast-variant library).
- **Test:** P08 with A8 fake-quant: healed-with-A8 vs healed-without, both scored with A8. Pass: ≤ 0.99.
- **Gain/risk:** −1…−3% of the fast variant's KL at 2.5.
- **Conflicts:** none.

**L2 · A4-aware healing at ≤ 3 bpw.**
- **Mechanism:** Heal the i4 levels with A4-g64 fake-quant on.
- **Source:** d1106 (A4's relative cost did not shrink after healing: 1.139 unhealed vs 1.161 healed); JustQuant (staged local→global needed when activations are 4-bit).
- **Why:** It could change #46-03's verdict.
- **Fit:** per level (A4 variant).
- **Test:** P08 K2 i4 + A4. Pass: A4/A8 ratio ≤ 1.08 after A4-aware healing.
- **Gain/risk:** decision for the A4 variant.
- **Conflicts:** #46-02, #46-03.

**L3 · Tied suh across shared-input siblings, then heal (speed).**
- **Mechanism:** Tie suh for gate/up, q/k/v and GDN qkv/z so one prep pass serves all siblings, and let healing recover the quality cost.
- **Source:** h1102-J2, #42.
- **Why:** It removes prep passes in the two-pass A8 path.
- **Fit:** per level.
- **Test:** P08: tied+heal vs untied+heal. Pass: ≤ 1.005, then time it.
- **Gain/risk:** speed at ≈0 KL.
- **Conflicts:** #42.

**L4 · Output-range guard after healing.**
- **Mechanism:** Record max |y| of down / ssm_out / attn_out over the long verify traces on healed containers.
- **Source:** design §1c.9.
- **Why:** Healing raises scales; an inf on the f16 tile costs more than anything gained.
- **Fit:** gate.
- **Test:** one long eval per finalist.
- **Gain/risk:** protection.
- **Conflicts:** none.

## M. Transfer, evaluation and guards

**M1 · 9B bridge pilot to estimate the shrink before S27.**
- **Mechanism:** Qwen3.5-9B untied (hidden 4096 × 32 layers). The bf16 student with fake-quant wq is 18 GB, the teacher is cached (A1), so it fits one R9700 with heal.py almost unchanged. Heal a K2.5 uniform level.
- **Source:** backlog (size-transfer point).
- **Why:** A third scale point (0.8B / 9B / 27B) before committing 60–100 card-h to G1.
- **Fit:** diagnostic.
- **Test:** gpubox card 1 when free: 9B quant (≈1 h) + heal (≈1.5 h) + 2 evals (needs a 9B golden; mkgold27 staged).
- **Gain/risk:** decision.
- **Conflicts:** none.

**M2 · Healing as a coherent-error map.**
- **Mechanism:** From any P08 or 27B heal, plot |log c|, |log r| by family × depth × K and correlate with kfit2 sensitivities and faraway M5 (early residual readers 1.35–1.74× an ideal coder's error).
- **Source:** new.
- **Why:** It shows where the LS refit is most wrong, which feeds coherent-lens refits and knapsack priors. It is free.
- **Fit:** diagnostic.
- **Test:** from existing d1104 params.pt.
- **Gain/risk:** none directly.
- **Conflicts:** none.

**M3 · Long-context, multilingual and tail guards on every healed finalist.**
- **Mechanism:** KL by position bin (golden-long2), golden-ml ratio ≤ 1.0, KL99 / KL99.9 and top-1 (pcmp's tail metrics).
- **Source:** h1102-L2/L3/A9; LOG 11-04 tails rule.
- **Why:** Healing trains on ≤ 8k windows and English-heavy qad data.
- **Fit:** gate.
- **Test:** 3 evals per finalist.
- **Gain/risk:** protection.
- **Conflicts:** none.

**M4 · Training-seed noise floor at 27B, and scale stability across seeds.**
- **Mechanism:** Run one heal twice with different data orders. Compare the KL ratio and the per-channel |Δlog c| between runs.
- **Source:** h1102-A8.
- **Why:** It sets the decision threshold for 27B healing comparisons. Large |Δlog c| means overfitting (C6).
- **Fit:** gate.
- **Test:** one extra 27B heal.
- **Gain/risk:** protection.
- **Conflicts:** none.

**M5 · Early stopping on a qad holdout, never on the search pool.**
- **Mechanism:** Hold out ~3% of qad docs for early stopping and logging. Search slices stay clean for knapsack accept/reject; verify is never read until finals.
- **Source:** w06 used 48 search docs not in g08-search1; this generalizes it.
- **Why:** Pool hygiene once healing runs on 25+ levels.
- **Fit:** policy.
- **Test:** ledger entry.
- **Gain/risk:** protection.
- **Conflicts:** #72 discipline.

**M6 · Provenance of healed scales.**
- **Mechanism:** Store healed scales keyed by (level id, window-set hash, recipe hash). Add a selftest that only suh/svh/norm planes changed, and record the container sha256.
- **Source:** h1102-L5/L6.
- **Why:** Byte-identical rebuilds of releases.
- **Fit:** policy.
- **Test:** none.
- **Gain/risk:** none.
- **Conflicts:** none.

---

## Must precede the freeze (everything else waits for the ladder)

- **H3**, for the user's pending writer-rollout decision under #111. Healing may make the per-mix writer pass unnecessary.
- **B7 / B8**, only if bias planes are wanted. The exl3i encoding and kernel epilogue would need the plane before the ladder; the values can come later.
- **C2**, only if the result says "no refit". Unlikely to change the freeze; run it if card time is free.

## Ranked top 20 by expected gain per unit cost

| # | idea | cost | expected gain at 27B 2.5 bpw | why here |
|---|---|---|---|---|
| 1 | **A1** cached top-k teacher equivalence | 1.5 h 4070 | enabler | the 27B cannot co-host the teacher |
| 2 | **B1 + B2** granularity / family decomposition | 4 h 4070 | decides J2 and the 27B parameter set | cheapest information |
| 3 | **G4** log-scale interpolation across K | 1 h 4070 | cuts 27B heals 3–4× | 25 levels → ~8 |
| 4 | **J1** logit-τ probe | 20 min gpubox | 0…−1% | first 27B coherent-error number |
| 5 | **C1** warm-start from swj / C9 / norm-restoring | 2 h 4070 | −0.5…−3% or ½ the steps | healing alone does not reach swj's point |
| 6 | **H3** plain + heal vs per-mix asym | 3 h 4070 | saves 10–15 card-h per release | user decision pending |
| 7 | **C3 + C9** K-dependent lr, fp32 master | 3 h 4070 | makes K ≥ 4 heals work | #89-06 |
| 8 | **C6** starvation emulation + regularizer | 2.5 h 4070 | protects 1–3% | 27B parameters per row 4–8× |
| 9 | **F1** 27B last-8 truncated pilot | ≈1 h card + A2 | −1…−4% | first 27B gradient heal |
| 10 | **J4** head-only KL healing | 1–2 h card | −0.5…−1.5% | no S27 needed |
| 11 | **A2 + A4 + G1** S27 + library healing + polish | 3–5 days eng + 60–100 card-h | **−5…−11%** | the main lever |
| 12 | **C4** diagonal Gauss–Newton steps | 1.5 h 4070 | 5× cheaper heals | multiplies #11 |
| 13 | **F2** 27B depth replication (with A5) | 5–9 card-h | decision (up to 2.5× cost) | 0.8B vs field report |
| 14 | **G2** two-level stochastic library | 1 h 4070, then 2× per heal | −0.5…−2% | transfer 0.8 → 0.95 |
| 15 | **E5** prefix burn-in windows | 1 h 4070 | −0.5…−2% | deep positions at forward cost |
| 16 | **B3 / B5** GDN conv / a-b compensators | 1 h 4070 each | −0.5…−2% | free unquantized capacity |
| 17 | **L1** A8-aware healing (fast variant) | 1.5 h 4070 | −1…−3% of fast-variant KL | per-variant scales are free |
| 18 | **E3** window length | 1.5 h 4070 | −0…−3% | 27B fits 8–16k |
| 19 | **M1** 9B bridge | ≈3 card-h | decision | shrink estimate before G1 |
| 20 | **G6 / G7** healed knapsack, heal finalists | ≈1 night per size | −0.5…−2% | selection after healing |

**Overall:** a library-healed and polished 2.5 bpw mix at about −7% KL (range −2…−12%) against the same mix unhealed with swj. Roughly −10…−20% at 2.0. Code refresh (I1/I2) is a further −0…−3%, with low confidence.

## The 5 cheapest diagnostics that would most change this ranking

1. **A1 cached teacher on P08 (≈1.5 GPU-h).** If top-512 + tail does not match online full-vocab KL within 0.5%, the 27B needs a co-resident or staged teacher. S27 cost then doubles and everything 27B moves down.
2. **B1 granularity share (≈2.5 GPU-h).** If per-tensor or per-band scalars carry ≥50% of the gain, J2 (forward-only SPSA on the 27B, no S27) jumps to the top 5 and a 27B healing number arrives this week.
3. **C6 starvation emulation (≈2.5 GPU-h).** If ¼ of the data keeps <80% of the gain, B6 tying, D1 prompt rows and E2 fit-pool reuse become mandatory parts of the 27B recipe.
4. **J1 τ probe and per-band gain line search on the 27B (≈1 card-h, container edits only).** It gives the size of coherent output-side error at 27B directly. ≥1% from a handful of scalars says 27B healing headroom is at least as large as the 0.8B suggests.
5. **G4 K-interpolation (≈1 GPU-h).** It decides whether the 27B library needs ~25 healing jobs or ~8, which is the difference between needing donors and fitting on gpubox in two nights.

**Sixth:** H3 (≈3 GPU-h), because it can remove the per-mix asym writer pass from the release plan.
