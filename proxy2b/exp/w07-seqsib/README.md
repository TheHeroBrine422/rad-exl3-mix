# w07-seqsib: intra-layer sequential inputs for o_proj / out_proj / down_proj (ranked.md C6)

**Question.** Should a layer's output projections (attention o_proj, GDN out_proj, MLP down_proj) be quantized on
the inputs they will actually see once their own layer's input projections are quantized, instead of on clean inputs?
- o_proj / out_proj inputs come from the quantized q/k/v (GDN qkv/z).
- down_proj inputs come from the quantized gate/up.

This is GPTAQ-style sequential calibration restricted to one layer. It needs no multi-layer sequential pass and stays
a per-level library item. ranked.md C6; a cheap subset of C5 (quantized-prefix targets, ROADMAP "sequential Hessians",
V3.5).

**Mechanism.** Per base quant (K, rotation seed), `../w07-denoise/moments.py --mode sib`:
- A clean forward over 1M qad-pool tokens.
- Every attention / GDN block and every MLP is re-run on its same clean input with that quant's own siblings swapped
  in.
- η is recorded per target: its input with quantized siblings minus the clean one, giving H, C, N.
- `mkdenoise.py` then builds the metric and target, with the control-variate form for G: H_calib + (G_sub − H_sub).
- Only those 48 tensors are requantized (`ourq --only`, same seeds). Every other tensor is the base quant's own, so
  arm and base differ in those 48 alone.

**Arms:**
- **sym:** G = H̃ = E[x̃x̃ᵀ] (λ = ρ = 1, F = I): plain sequential GPTQ. Only `--hin`.
- **asym:** the same G, target W·F with λ = 1, ρ = 0.5: halfway from "match the quantized input's output" toward "match
  the clean output" (GPTAQ / QEP with correction strength 0.5). Adds `--wpre` (needs `../w07-denoise/wpre.patch`).
- **Noise floor rs7 / rs8 / rs9:** the same 48 tensors requantized on the calib H at seed offsets 7 / 8 / 9, with base
  seed 0's siblings. This is exactly the rounding noise an arm-vs-base pair carries, measured instead of assumed.

**Size of the effect (CPU, real 0.8B, `../w07-denoise/selftest/test_moments.out` E / F).**
- With K3 siblings, the intra-layer error at the targets' inputs is tr N / tr H ≈ 0.005 (L0 out_proj), 0.008 (L3
  o_proj), 0.016–0.019 (down).
- The *full* upstream error of the K3 quant at the same points is 0.005–0.10 (o) and 0.05–0.13 (down), rising with
  depth (`../w07-denoise/selftest/depth-profile.txt`).
- So siblings explain the whole error at layer 0 but only a small fraction deeper in. **C6 addresses a minor part of
  the input error**; the `quant` arms of `../w07-denoise` (C1) model all of it. Expect C6 at the low end of ranked.md's
  −0.5…−2%.

## Design
- **Screen:** sym and asym at seed 0, K3 and K2, scored on **g08-fit1**. Per K the lower one is picked.
- **Confirm:** the pick at seeds 0 / 1 / 2 on **g08-search1**. Each seed uses its own base's siblings and moments, and
  is paired with `k{K}base{,_s1,_s2}` (existing).
- The controls are scored on search.
- **No-op gate first:** the 48 tensors requantized on mkdenoise's λ = ρ = 0 metric (= calib's H) with base seed 0
  must equal k3base's own bit-exactly. Otherwise the base quants are stale (ourq / exllamav3 changed since) and the
  "only these 48 differ" design fails.

## Draft pass rule (for the lead to commit before data)

Per K, for the fit-picked arm, with ratio_s = KL(arm_s) / KL(base_s) on g08-search1, and s_c = the SD of the three
control ratios KL(rs_c) / KL(base seed 0):
- **PASS:** mean ratio_s ≤ min(0.98, 1 − 3 s_c / √3), i.e. the gain is beyond 3 standard errors of the measured
  noise floor and at least 2%, **and** every ratio_s < 1.
- **FAIL:** mean-KL ratio ≥ 1.00.
- **INCONCLUSIVE otherwise:** not adopted.
- **Note:** ranked.md's "as C5" bar is ≤ 0.96. That fits a full-trunk requant. Here only 48 of 150 tensors change
  and the noise floor is measured, so 0.98 is proposed; the lead may restore 0.96.
- **Reported beside the rule:** fit-slice ratios of both arms; the controls; mkdenoise stats (clip share,
  ‖WF‖/‖W‖ for asym); paired z per seed.

## Run

```
proxy2b/exp/w07-seqsib/run.sh     # 4070. sym + controls need no patch; asym needs ../w07-denoise/wpre.patch merged
```
- Results go to `/data/yb/w07-seqsib/result.txt` and `summary.jsonl`.
- Idempotent and resumable.
- **GPU time ≈ 1–1.5 h:**
  - 6 sib collections (1M tokens, two layer passes), ~6–8 min each;
  - 13 mkdenoise dirs;
  - 16 partial quants (48 tensors), ~1 min each;
  - ~16 evals.

## What the result changes
- **PASS:** sequential inputs pay even within a layer. Two consequences:
  - Rank C5 (quantized-prefix targets, all upstream layers) and the `quant` source of C1 above it.
  - Bake sym / asym into the per-level requant (#58). For the 27B this needs a layer-streamed forward with the
    level's quantized siblings (the siblings' dequantized bf16 copies do not fit beside the model): an iceman script,
    not written until this passes.
- **FAIL / inconclusive:** C6 closed. A C1 `quant` result decides whether full-upstream sequential targets are worth it
  anyway.

## Files
- `run.sh`. It uses `../w07-denoise/moments.py`, `mkdenoise.py` and `summ.py`; their self-tests cover the sib mode
  (bit-exact no-op with own weights, independent recomputation of the down input) and the ctrl verdict rule.
- `../w07-denoise/selftest/dryrun.out`: this run.sh end to end with stubs (gate, pick, ctrl verdict, idempotence).
