# radiance 1.17.0 (2026-11-04): IQR-V4 kernels and what they mean for EXL3-V3

Source: the v1.17.0 tag (fork remote `upstream`), `libr4d/r4d_gemm_iqr.hip`, `libr4d/r4d_logits_vocab.hip`,
`libquant/lq_gptq.cpp`, release notes. Plugin ABI 33 → 35 (appends only).

## Decode kernels (the part V4 changed; prefill tile unchanged, "8–9% behind MXFP4's 4-bit tile")
- **GEMV (1 token; ≤ 8 for other forms):** a *block* per 16-row weight tile with KW waves splitting K (wave w takes
  the tile's 128-K blocks w, w+KW, …), lanes on 4 consecutive columns each (one `sudot4` per row per token), partial
  sums reduced per wave then across waves through LDS in ascending order. Replaces "a wave a tile over all of K",
  which left a ragged last round (N/16 tiles never divide the 15 waves a SIMD holds at 102 VGPRs). 5120 × 6144 at one
  token: 42.3 → 30.0 µs.
- **Biased-int fold:** under one f32 scale a row, each unit's int32 sums start at `0x4B400000` (= 12582912.0f), so
  the integer accumulator read as an f32 is exactly 12582912 + sum (|sum| < 2^22): one FMA folds it with the block
  factor, the bias removed once at the end. No int→float conversion in the loop.
- **Activation broadcast by `v_perm`:** each of a lane's 4 activation bytes is spread to 4 lanes' worth with one
  `__builtin_amdgcn_perm`, no packing.
- **Scales via scalar loads:** the (row, block) scales of a tile are 16 consecutive bytes at a wave-uniform address,
  read through the constant address space into SGPRs (no readlane broadcast).
- **16-token WMMA tile without block barriers:** same block / KW layout; each wave decodes a 128-K unit and
  transposes it into fragment order through its own LDS tile, then runs the unit's 8 WMMA steps against activation
  fragments read straight from global memory (the M × K activation sits in L2): no shared staging, no barrier.
  2–16 tokens; a 4-bit weight at 0.52–0.66 of FP8's time; 1.15–1.74× V3's kernels from 4 tokens up.
- **Fused producers:** the activation quantiser after gate_up (`gemm_iqr_gated`: gate and up tiles in one block,
  silu·up rounded, the last of a 128-column group's 8 blocks rotates and quantises the group), the GDN decode conv
  fold, and the attention gate's quantiser are folded into the kernels that produce their inputs: 128 fewer launches
  a decode step, same bytes.
- End to end (TP1, no speculation): IQ4R-S-V4 24.7 ms a token vs the MXFP4 checkpoint's 25.8.

## For our kernels
- **V3 decode GEMV (exl3r4d today):** adopt the block-per-tile / KW-waves split and the biased-int fold for the int8
  path; for the f16 path the same block / K-split layout applies. Our trellis decode costs more VALU per weight than
  IQR's computed code, so the wave count per block (KW) needs its own sweep.
- **2–16 tokens (speculative verify with DFlash2, small batches):** the barrier-free 16-token tile is the template.
- **Fusions:** gated activation quant in gate_up's epilogue is directly applicable once our A8 path exists.
- **Prefill:** `r4d_gemm_8a8_prefill.h` is unchanged in 1.17, so the two-pass plan (`iqr-v4.md` "Speed") stands.

## Quantiser-side (libquant)
- **Act order** (`tcq_order=1`): columns sorted by descending H diagonal (stable), GPTQ in that order, stored back in
  place. For EXL3: permuting K at **16-column block** granularity (LDLQ's unit) is free at runtime (the decoder
  never sees it: only the processing order changes); a finer permutation would have to be stored or folded into the
  rotation. Test block-level act order on the CPU proxy first (BACKLOG row).
- **rescale=1:** per-row least-squares scale refit after GPTQ, a = (w H x^T)/(w H w^T) — what exllamav3's
  `refit_scales` already does for EXL3.
- **Vocabulary-weighted head:** four width classes, tokens carrying the probability on the widest, one launch
  (`logits_gemm_vocab`, 16-bit head activations): 0.62 ms at one token on a TP2 rank vs 1.02 for the int8 head.
