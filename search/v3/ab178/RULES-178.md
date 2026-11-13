# RULES-178 — GDN in_proj_a / in_proj_b quantized (w142, #178; mining A2, lead notes 11-13)

Written and committed before any arm is quantized or scored. Pre-registration is attested by this file's git commit.

**Question.** The GDN gate / beta projections (`blk.N.ssm_a.weight`, `blk.N.ssm_b.weight`: [48, 5120] each, 48 GDN
layers, 23.59M weights) are bf16 outside the knapsack: 0.01404 bpw. Is a quantized a/b worth its KL cost at 2.5 and
4.0 bpw, priced against what the freed bytes buy in the trunk?
conv1d (bf16, 0.0012 bpw), A_log and dt_bias (f32, tiny) stay as they are: not tested.

**Arms (a/b only; everything else identical to the baseline container).**
| arm | form | how | bits / a/b weight | bpw saved |
|---|---|---|---|---|
| B | bf16 (today) | — | 16 | 0 |
| G | INT8 G128 FWHT128 GPTQ = the IQ8R recipe line (`gptq codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search calib=$CALIB`, damp 0.01 default) | radiance libquant `gptqsim` (the same GPTQ, decoded and stored bf16), called on the calib layer-input Hessian (the published one, as RADGRAM2) | 8.125 | 0.00691 |
| R | the same form, RTN (no calib) | libquant `gptqsim` without `calib=` (round to nearest, rule=search scales) | 8.125 | 0.00691 |
| T4, T5, T6 | our trellis, codebook i8 (mul1i8), K4 / K5 / K6 | rxq quant (config/v3.toml settings) on a fused a\|b weight padded to 128 rows (exllamav3's LDLQ needs N % 128 = 0; 48 or 96 rows are refused), in the layer-input group (shares qkv / z's su draw, seeds j = 2), no H_out (none exists for a/b), `--save-wq` decode | K × 128/96 (padding rows are stored) | T4 0.00936, T5 0.00819, T6 0.00702 |

All quantized arms are **dequantized splices**: the decoded a/b values are written as bf16 over the baseline's a/b
planes, so the KL measures the weight error only. The A8 activation of a served int8 a/b (had_quant_act_i8g, its own
fwht128 rotation of the bf16 `h` plane, not the exl3 trunk's suh·Hadamard) is not in the KL: noted, not measured.
Bytes are the served form's (table), not the splice's.

**Containers.** The #171 incumbents (state.jsonl after the 10:11 AM CST stop): 2.5 = `v3s171-2.5-block-00626`,
4.0 = `v3s171-4-block-00420`; each tag's hostd `.mix` + `act <w> a8g128` on every `blk.*` trunk weight (w132's form),
built by radmix; the arm's a/b bf16 bytes are then written into the arm's copy in place. 2 sizes × 6 arms = 12 scores.
Scored with the rad10 build and hostd's score command (bf16 KV, max-model-len 16384).

**Data.** `golden-s171d` (w131's two-part slice: 54 calib docs from the merged fit + search pool, 130 xcorp docs;
210,258 positions; already in the ledger). The verify pool is not used.

**Statistic.** Per document, the KL sum over the golden's generated positions (`rxrun.hostd.doc_sums`).
- d = mean over p ∈ {c, x} of log(Σ_p KL_arm / Σ_p KL_B) (#171's pmean); also per part.
- 95% CIs from 2,000 bootstrap resamples of documents, stratified by part, paired (same resample for arm and B).
- Trunk price of the freed bytes: g = b_s × Δbpw with b_s the driver's last slope −d log KL / d bpw (RULES-value:
  2.5 → 1.495, 4.0 → 1.145). Net N = d − g (negative = the arm wins).

**Reading (fixed now).**
- An arm is **adopted** at a size if N < 0 at the point estimate and the upper end of d's 95% CI is < g + 0.005
  (at most half a percent of harm beyond the price if the CI's tail is true).
- It is **rejected** if the lower end of d's CI is > g (the KL cost certainly exceeds what the bytes buy).
- Otherwise **null** (no evidence either way at this slice's power).
- Every arm on the frontier of (bpw saved, d) at either size goes to w133's menu as an a/b option with its measured d
  as its price, whatever the reading (lead note: all arms in one table).
- Also reported, not used for the reading: per-part d, KLD per arm, served-form timing (below).

**Timing.** The a/b `gemm_nt` share of a decode step and an 8k prefill on the baseline container (rad-kbench op
times / the GDN block), which bounds what any int8 / trellis a/b can save; the extra activation quantizer an int8
a/b would launch per GDN layer is named against it. A served i8a8h or exl3 a/b needs a qwen35_exl3 change (not in this
experiment).

**Runs.** ds8: scoring under `/data/q27/locks/card*.lock` + `ram.lock` shared, radmix under `conv.lock` + `ram.lock`
shared; the trellis quant on one card under its card lock; GPTQ / RTN on the CPU (nice 19). Mixes deleted after their
score. Scripts in this folder.
