# #178 results — GDN a/b quantized (w142, 2026-11-13)

Rule `search/v3/ab178/RULES-178.md` (9d59b5a, before any run). ds8, rad10 build, golden-s171d (54 calib docs from the
merged fit + search pool, 130 xcorp docs; 210,258 positions), A8 trunk (`act a8g128` on all 336 trunk weights).
Arm B = w132's RULES-value score of the identical mix (reused; my own rescore of B at 4.0 gave **byte-identical rows**).
The arm containers differ from B in exactly the 96 a/b weights (census: 96 mismatches vs the B mix, nothing else).

## KL (`ab178.out`, `ab178.json`)

d = pmean over parts of log(ΣKL_arm / ΣKL_B); 95% CI from 2,000 paired document bootstraps, stratified by part;
g = b_s × bpw saved (b_s 1.495 at 2.5, 1.145 at 4.0); N = d − g.

| size | arm | bpw saved | d | 95% CI | c part | x part | g | N | reading |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | G INT8 G128 FWHT128 GPTQ | 0.00691 | +0.41 % | [+0.15, +0.68] | +0.09 % | +0.73 % | 1.03 % | −0.62 % | adopted |
| 2.5 | R same form, RTN | 0.00691 | +0.18 % | [−0.02, +0.49] | −0.09 % | +0.45 % | 1.03 % | −0.85 % | adopted |
| 2.5 | T4 trellis i8 K4 | 0.00936 | +0.37 % | [+0.03, +0.61] | −0.04 % | +0.78 % | 1.40 % | −1.03 % | adopted |
| 2.5 | T5 | 0.00819 | +0.41 % | [+0.01, +0.67] | +0.02 % | +0.80 % | 1.22 % | −0.81 % | adopted |
| 2.5 | T6 | 0.00702 | +0.21 % | [−0.11, +0.52] | −0.04 % | +0.45 % | 1.05 % | −0.84 % | adopted |
| 4.0 | G | 0.00691 | −0.56 % | [−1.29, +0.60] | −0.21 % | −0.92 % | 0.79 % | −1.35 % | adopted |
| 4.0 | R | 0.00691 | +0.76 % | [−0.29, +3.09] | +0.44 % | +1.09 % | 0.79 % | −0.03 % | null |
| 4.0 | T4 | 0.00936 | −0.54 % | [−1.31, +0.75] | −0.16 % | −0.93 % | 1.07 % | −1.61 % | adopted |
| 4.0 | T5 | 0.00819 | −0.18 % | [−1.13, +1.59] | −0.24 % | −0.13 % | 0.94 % | −1.12 % | null |
| 4.0 | T6 | 0.00702 | −0.52 % | [−1.25, +0.67] | +0.12 % | −1.16 % | 0.80 % | −1.32 % | adopted |

KLD of B: 0.20443 (2.5), 0.07193 (4.0).

**What the numbers say.** The a/b weight errors span two orders of magnitude (input-Hessian relative error, median
over layers, a / b: G 2.9e-6 / 1.1e-6, R 1.8e-5 / 7.2e-6, T6 1.9e-5 / 9.0e-6, T5 7.2e-5 / 3.4e-5, T4 2.8e-4 / 1.3e-4;
`proxy-GR.jsonl`, `proxy-T.json`), and d does not follow them. The calib part moves by at most ±0.24 % (±0.44 % for R
at 4.0). The xcorp part is dominated by two or three high-KL documents (s171d x docs 46 (8,191 tokens), 45 and 140; the x part's KL per position is 4.4× calib's at 2.5 and 11× at
4.0). They move by 50–170 KL units under any a/b perturbation, with a random sign: at
2.5 every arm is +0.45…+0.80 % on x, at 4.0 most are −0.9…−1.2 %. This realization noise is not in the document bootstrap.
So none of the arms has a measurable KL cost at this slice's power. The price of the freed bytes (0.8–1.4 %) is
larger than any credible cost.

**Frontier (bpw saved vs d).** T4 saves the most, and its d is no worse than any other arm's: it dominates T5, T6 and
R. G is the stock radiance-2.0 tensor (IQ8R's line) and saves 0.0069. Both go to the menu as a/b options; their KL
price is ~0 (use the c-part d, or 0, rather than the x-driven point estimates).

## Time (`tprof-2.5.out`; arm B 2.5 served with `--profile-ops`, 9,001-token prompt in 2 chunks + 256 decode steps)

| | all ops | a/b gemm_nt (96 ops, N 48 K 5120 bf16) | trunk A8 quantizers (exl3i_actq) |
|---|---|---|---|
| prefill (2 chunks) | 3,517 ms | 19.2 ms (0.55 %) | 224 ms (6.4 %; prefill only) |
| decode step | 36.3 ms | 0.90 ms (2.5 %; ~9.4 µs an op) | 0 (decode runs f16 activations) |

Profiled times include extra synchronisations, so read these as shares, not wall times.
- **Decode.** The a/b GEMMs are latency-bound: each moves 0.5 MB in ~9 µs, so halving the bytes saves ~nothing. Fusing
  a | b into one N = 96 op, as IQR's GDN block does, removes 48 launches a step in any format; that is a speed item of its
  own.
- **Prefill.** int8 a/b could save at most ~10 ms of the 19 ms. A stock i8a8h a/b needs its own `had_quant_act_i8g` on `h`:
  plain fwht128, no sign vector, so it cannot read the trunk's suh·H128 A8 plane. That adds one quantizer per GDN layer
  per chunk, an estimated 10–40 ms. So i8a8h is net-slower at prefill, and w8a16 (dequantize in the GEMM, bf16
  activation) is the neutral form. T4 was quantized in the qkv / z input group (the same su draw), so a served exl3i
  a | b weight could read the existing `aq_in` plane at no extra quantizer cost.
- **Neither form is served today.** Both need a qwen35_exl3 change:
  - G: an i8 weight path for a/b.
  - T4: a 128-row padded exl3i a | b weight in the qkv / z plane group, with outputs 0–47 → a, 48–95 → b, and
    96–127 dropped.

## Files
`docsums178.json` (per-document KL sums, all arms + B), `ab178.json` / `.out`, `proxy-GR.jsonl` (G / R per layer),
`proxy-T.json` (T4–T6 per layer, from rxq's rowerr), `run178.log`, `tprof-2.5.out`, `census-runs.txt` (B2: census on
packs, donors, an R16 head, the B mix and an arm container). The a/b tensors stay on ds8: `/data/q27/w142/gq/ab-{G,R}.safetensors`,
`/data/q27/w142/tq/ab-T{4,5,6}.safetensors` (bf16 decodes) and `tq/K{4,5,6}/` (rxq's trellis tensors for packing).
