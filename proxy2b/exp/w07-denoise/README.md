# w07-denoise: activation-denoising LDLQ target and metric (ranked.md C1)

**Question.** Does a noise-aware LDLQ target and metric beat our tuned one-sided quant (`base`) on the 0.8B at K3 / K2?
This is activation denoising (arXiv 2610.07522): every tensor is quantized against an estimate of the error its own
input will carry once the layers below it are quantized. It needs no sequential quantization, so it would fit the
per-level library and the Phase-B requant (#58).

**Mechanism (paper, Theorem 1).**
- **Objective:** E‖Ŵ(x + λη) − W(x + ρη)‖² + α‖Ŵ − W‖² equals tr[(Ŵ − WF) G′ (Ŵ − WF)ᵀ] + const, with:
  - G = H + λ²N + λ(C + Cᵀ), and G′ = G + αI;
  - F = (H + αI + λC + ρCᵀ + ρλN) G′⁻¹;
  - H = E[xxᵀ], C = E[xηᵀ], N = E[ηηᵀ], where η is the perturbation of a group's input.
- **Mapping onto our pipeline:**
  - LDLQ quantizes the target WF in the metric G (`ourq.py --hin` = G, `--wpre` = Fᵀ).
  - α is exactly the damping ourq adds: 0.005 × mean diag on the layer-input groups, 0.025 elsewhere.
  - The scale refit and the proxy follow G and WF automatically.
- **Two parameters:**
  - λ: how much error the quantized layer is assumed to receive.
  - ρ < λ: partial denoising, which the paper found best.

## What the CPU self-tests found (these change the design)

All runs on the real 0.8B, colo CPU.

1. **The paper's noise is far too strong for our levels** (`selftest/depth-profile.txt`, every layer):
   - Paper's Gaussian, γ = 0.05 × channel max, propagated: tr N / tr H ≈ 0.25–0.75 at the layer inputs from layer 1
     on, and 0.6–2.5 at down_proj inputs.
   - The real upstream error of our K3 base quant: 0.02–0.06 at the inputs, 0.05–0.13 at down.
   - So the paper's noise is 10–20× the K3 error in variance. At that size the noise saturates through RMSNorm:
     tr(C + Cᵀ) ≈ −tr N, i.e. the noise *replaces* signal.
   - The Gaussian arms therefore use a **level-matched γ_K**: a γ = 0.01 pilot (variance ∝ γ²) sets the scale so that
     the median layer-input tr N / tr H equals the measured K-level error.
   - The paper's literal γ = 0.05 is not run. g3 (twice the level's noise) is closest to its λγ at K3.
2. **F must come from one joint distribution** (`selftest/fsub-vs-cv.txt`):
   - The obvious construction, calib's H (10M tokens) with subset C, N, makes F an inconsistent regression:
     ‖WF‖/‖W‖ = 1.6–26×.
   - The subset's own (H_sub, C, N) gives 0.98–1.01, the paper's mild shrink.
   - mkdenoise.py therefore builds:
     - **F** from the subset alone;
     - **G** in the control-variate form H_calib + (G_sub − H_sub). At λ = 0 this is calib's H exactly, so arms
       differ from base only by the denoising. Negative eigenvalues are clipped if any, and the clipped share is
       logged (small at 1M tokens is expected; at 1k tokens it was 0.2–20%).
3. **A second noise model,** "quant": the paper's own *parallel-quantized upstream error* proxy.
   - η = the input of each group when **every** trunk linear is swapped for the uniform-K base quant, minus the clean
     input. The layer's own siblings are included.
   - Level-matched by construction. We take it from the seed-offset-3 base quant (`k{K}base_s3`, never an evaluated
     seed), so no evaluated seed is denoised against its own error.

## Arms (0.8B, i8, base settings via q2b.sh: scaled Hessian, damping 0.025 / input groups 0.005)

| arm | noise | λ | ρ | meaning |
|---|---|---|---|---|
| g1 | Gaussian, γ_K | 1 | 0.9 | noise = the level's error, mild denoising |
| g2 | Gaussian, γ_K | 1 | 0.5 | strong denoising |
| g3 | Gaussian, γ_K | 2 | 1.8 | twice the level's noise (≈ the paper's λγ at K3), mild denoising |
| q1 | quant (k{K}base_s3) | 1 | 1 | F = I, G = E[x̃x̃ᵀ]: sequential-GPTQ metric from a parallel quant (no `--wpre`) |
| q2 | quant | 1 | 0.75 | partial asymmetric correction (GPTAQ-like) |
| q3 | quant | 1 | 0.5 | stronger correction |

**Design:**
- Moments come from 1M tokens of qad-pool 2048-token windows (pools-v3), disjoint from both eval goldens.
- **Screen:** every arm at seed 0, K3 and K2, scored on **g08-fit1**. Per K the arm with the lowest fit KL is picked
  (summ.py, mechanical).
- **Confirm:** the pick at seed offsets 0 / 1 / 2 on **g08-search1**, paired with `k{K}base`, `_s1`, `_s2` (existing).
- **No-op gate first:** mkdenoise at λ = ρ = 0 as `--hin` plus identity `--wpre` must reproduce every k3base wq tensor
  bit-exactly.

## Draft pass rule (for the lead to commit before data)

Per K, for the fit-picked arm, with ratio_s = KL(arm, seed s) / KL(base, seed s) on g08-search1:
- **PASS:** every ratio_s ≤ 0.985 (K3) / 0.97 (K2), **and** every arm seed's KL is below every base seed's KL.
  This is ranked.md C1's rule with #33-16's seed convention.
- **FAIL:** mean-KL ratio (3 seeds each) ≥ 1.00.
- **INCONCLUSIVE otherwise:** not adopted.
- **Gate:** the no-op gate must PASS, or nothing counts.
- **Reported beside the rule, not part of it:** every arm's fit-slice ratio, i.e. which noise model wins; γ_K; clip
  shares; ‖WF‖/‖W‖ per tensor (mkdenoise stats); paired z per seed.

## Run

```
proxy2b/exp/w07-denoise/run.sh     # 4070; after wpre.patch is merged. Results /data/yb/w07-denoise/result.txt, summary.jsonl
```
- Idempotent and resumable: moments, dirs, quants and evals are each skipped when done. The dry run's second pass
  re-runs nothing.
- VRAM: moments run in two layer passes, ~6 GB peak expected (fp32 H/C/N sums of 12 layers + the model, + quantized
  copies in quant mode). The allocator is capped by VRAM_CAP_GB = 7.2.
- **GPU time ≈ 1.5–2 h:**
  - 5 collections, ~5–8 min each;
  - 13 mkdenoise dirs, ~1 min each;
  - 17 quants, ~2.5 min each;
  - ~21 evals.

## What the result changes
- **PASS at a K:**
  - The per-level requant (Phase B, #58) gets the denoised target at and around that level.
  - **Quant source:** the 27B needs a uniform-K quant of the level (ladder) and a layer-streamed two-model forward for
    the moments (bf16 + quantized do not fit the two R9700s together). That is a new iceman script, ~1–2 h per level
    for ~1M tokens.
  - **Gauss source:** one batch-2 forward per level, but C and N for all 496 linears ≈ 2× the Hessian storage. Layer
    passes, host RAM / disk.
  - Neither 27B script is written yet. Both wait for this result.
- **Quant source wins on fit but gauss does not:** the measured error is the useful noise model, and C6 (`../w07-seqsib`,
  sibling-only error) is its weaker cousin.
- **FAIL / inconclusive:** C1 closed for the trellis at K ≤ 3 (ranked.md #21). The machinery stays usable for C5 / C6.
- Interplay with the diagonal H_out (#33-16) is not tested. It would be a follow-up arm on top of a PASS.

## Files
- **`moments.py`** (shared with `../w07-seqsib`): H / C / N per group from qad windows.
  - `--mode noise`: batch [clean, noisy], noise seeded per (window, layer, group).
  - `--mode quant`: clean, then the quantized model, on the same window.
  - `--mode sib`: block reruns with quantized siblings.
- **`mkdenoise.py`**: G (`--hin` dir, challenge layout) and Fᵀ (`--wpre` dir), plus stats.jsonl.
- **`wpre.patch`** (quantize/ourq.py, against the attached version; also applies to the current tree and in any order
  with the w01 / w02 / w04 / w05 ourq patches):
  - `--wpre DIR`: a trunk linear's checkpoint weight is read as W·F (DIR/<group>.safetensors, "Ft"). A missing group
    file is an error.
  - The option uses `default=argparse.SUPPRESS`, so when unset it is absent from the stamp's argument dict, and that
    dict is unchanged.
- **`run.sh`, `summ.py`** (pick / verdict; shared with the other two w07 folders).
- **Self-tests:**
  - `selftest/test_wpre.out`: **ALL PASS**. Real ourq.py driver (original and patched), exllamav3 replaced by a
    deterministic fake, CPU, layer 0 of the 0.8B.
    - Option off: all 30 output tensors identical, proxy.jsonl identical, stamp args identical.
    - Identity transforms: identical.
    - Random transforms: saved wq = W·F exactly.
    - Missing file: exits naming the group.
  - `selftest/test_moments.out`: **ALL PASS**. Real 0.8B, colo CPU.
    - γ = 0: C = N = 0 exactly.
    - Layer-0 noise has diag N = σ² (ratio 1.0006).
    - Split layer ranges are bit-exact with one run.
    - sib / quant modes with the model's own weights: C = N = 0 exactly, H = clean.
    - The down-input moments match an independent act(G_q x)(U_q x) recomputation to 1e-7.
  - `selftest/test_mkdenoise.out`: **ALL PASS**. Synthetic data.
    - The sample objective's differences equal the trace form's (Theorem 1 with ourq's damping), and WF minimises it.
    - λ = ρ gives F = I; Theorem 2's shrink holds.
    - F comes from the moments alone; G is the control-variate form.
  - `selftest/test_summ.out`: **ALL PASS** (pick and the three verdict rules on synthetic scores).
  - `selftest/dryrun.sh` → `dryrun.out`: all three w07 run.sh scripts end to end, twice, with the GPU programs stubbed
    (`selftest/stubs/`). Gates, picks, verdicts, and idempotence (the second pass re-runs nothing) all check out.
  - Measurements: `selftest/depth-profile.txt`, `selftest/fsub-vs-cv.txt`, `selftest/mkdenoise-real.txt`.
