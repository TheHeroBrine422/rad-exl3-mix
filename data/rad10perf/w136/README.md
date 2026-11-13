# #164-02 · R16 head end to end in the staged rad10-perf build (w136, box2, 2026-11-13 9:08–10:06 AM CST)

Scripts: `pipeline/hosts/rad10perf/` (README: build, recipe line). Build: radiance-hero `qc-exl3-v3-perf` 5c9956c in
`/data/q27/rad10-perf` (box2; copied to ds8, tree sha256 identical). Golden: `golden-ver3-8k` (calib verify, 104 docs,
156,030 positions; routine check, the same slice w130 used for #170). bf16 KV.

## Packs (all: ladder `q27v3-144dd6e1-k4-i8` final.recipe trunk, 336 tensors mul1i8 K4; text-only via home-pack)

| pack | head | how | file bytes | output.weight stored |
|---|---|---|---|---|
| A4 | R16s003 K4 i8, quantized (`exl3i ... codebook=mul1i8 rate=4 rows=r16`) | rad-convert `--no-error`, 130 s | 15,422,568,448 | 636,454,400 |
| B4 | the search's donor `hd-u4.rad` (same head, dequantized, un-permuted, bf16 round-to-nearest) | radmix of A4 | 17,333,104,640 | 2,542,796,800 |
| A2.5 | R16s003 K2.5 i8 quantized (`rate=2.5`) | rad-convert, 130 s | 15,184,181,248 | 398,067,200 |
| B2.5 | donor `hd-u2.5.rad` | radmix of A2.5 | 17,333,104,640 | 2,542,796,800 |
| B4t / B4s | same f16 head, bf16 by truncation / stochastic rounding (yardsticks, `donortrunc.py`) | radmix of A4 | 17,333,104,640 each | 2,542,796,800 |
| C4 | A4's quantized head taken by radmix into B4 (can the search splice real heads?) | radmix of B4 | 15,430,959,104 | 636,454,400 |

`radents.py`: every pair differs only in `output.weight` (A/B, A/B4t, A/B4s); **A4 vs C4: no payload differs**.
`mkf16ck.py`: the donors equal bf16(f16 dequant) value for value (1,271,398,400 / 1,271,398,400, K4 and K2.5).

**Head bytes vs the menu (menu129h, `search/v3/mkmenu.py` l. 109: vocab × 5120 × K / 8 + 1 MiB):** K4 636,454,400 vs
636,747,776; K2.5 398,067,200 vs 398,360,576. Real = trellis + 755,200 B = trellis + 3 B × 248,320 rows + 2 B × 5,120
columns, at both K: the menu overprices every head by 293,376 B (≈ 8.7e-5 bpw), conservative.

## KLD (`res/kldpair*.json`; per-document = |Δ| of a document's mean KLD)

| pair | KLD A | KLD B | \|Δ mean\| | per-doc max \|Δ\| | per-pos mean \|Δ\| | top-1 A / B | flips |
|---|---|---|---|---|---|---|---|
| A4 vs B4 | 0.1570843 | 0.1570834 | 9.2e-7 | 1.39e-4 | 2.40e-4 | 93.950 / 93.949 % | 200 |
| A2.5 vs B2.5 | 0.1584965 | 0.1584986 | 2.1e-6 | 8.5e-5 | 2.37e-4 | 93.246 / 93.233 % | 154 |
| B4 vs B4s (two unbiased bf16 roundings) | 0.1570834 | 0.1570911 | 7.7e-6 | 1.19e-4 | 2.61e-4 | | 229 |
| A4 vs B4s | 0.1570843 | 0.1570911 | 6.8e-6 | 1.94e-4 | 2.84e-4 | | 213 |
| B4 vs B4t (truncation: biased) | 0.1570834 | 0.1567163 | 3.7e-4 | 4.1e-3 | 5.81e-4 | | 242 |
| A4 vs C4 (radmix-spliced quantized head) | 0.1570843 | 0.1570843 | 0 | 0 | 0 | | 0 (rows byte-identical) |

The mean is within 1e-5 at both K. Per-document differences of ~1e-4 are the bf16 splice's own noise floor: two unbiased
bf16 roundings of the same head differ by as much as the quantized head vs the round-to-nearest donor. A tighter
reference (the f16 dequant as a dense head) cannot be packed: qwen35_exl3 declares output.weight bf16 and refuses f16
(`f16ref.log`). K4 + bf16 head (w130, rad10): 0.156978; this K4 + the R16 K4 i8 head: 0.157084.

## Text (`res/ask-*.txt`, 5 rounds × 4 packs)

All 20 answer "The capital of France is Paris.", a correct Fibonacci function, "Das Wetter ist heute schön.", and
" jumps over the lazy dog."; greedy answers identical A vs B; first-token logprobs differ ≤ ~0.01 (e.g. −0.3410 vs
−0.3417).

## Timing (card 0, `res/dstep-*`, `ask-*` prefill 8k × 3 a round; rounds 3–5 interleaved A4 B4 A2.5 B2.5)

| pack | decode step C 1 short (ms) | decode step C 1 8k ctx (ms) | prefill 8k (tok/s) |
|---|---|---|---|
| A4 (R16 K4 head) | 29.00 [28.99–29.00] | 30.15 [30.14–30.31] | 1,601 |
| B4 (bf16 head) | 32.08 [31.91–32.59] | 33.13 [33.12–33.14] | 1,600 |
| A2.5 (R16 K2.5 head) | 29.13 [28.99–29.14] | 30.92 [30.15–32.95] | 1,600 |
| B2.5 (bf16 head) | 31.91 [31.74–31.91] | 32.96 [32.96–33.13] | 1,598 |

Median [range] of rounds 3–5. During them other workers ran radiance on box2 cards 2 / 3 (w138 IQR-V4, w139) and an
HF download; rounds 1–2 overlapped my own CPU jobs (f16 un-permute, radmix copies) and are noisier (B4 35.1 / 36.3 ms).
Quantized head −3.1 ms / step (−9.6 %) = 1.9 GB less read at ~620 GB/s. The K2.5 head is no faster than the K4 head
(fractional-K decode, #164 pass 1: ×1.03–1.12 vs K4 per shape). Prefill unchanged.

## Build check (`radbuild-perf.log`, `rad10build.log.gz`)

8 min on box2 (core 6.9 min); plugin host tests pass (exl3quant 1, exl3dense 2, exl3i 3; qwen35_exl3 / exl3r4d have
none labelled host). vs `rad10/r374` (374503f + the core patch, w130): all plugin .so byte-identical; libr4d.so /
libquant.so / binaries differ only in embedded source paths, libr4d `.hip_fatbin` identical (sha256 b7a42dea…).
