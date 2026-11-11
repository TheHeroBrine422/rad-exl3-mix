# #130 kernel note · un-permuting the P16 output-row interleave at inference (w101, 2026-11-11; no kernel work done)

**What has to happen.** With P16 the stored weight's row order is π = `gperm(R, 16)`: stored 128-row block b holds
16-row tiles b, b + nb, …, b + 7nb of the natural order (nb = R / 128). The trellis codes, suh / svh and the 128-point
output rotation all live in the stored basis, so the GEMM's arithmetic is unchanged. Only the output columns come out
in stored order, and the consumers need natural order: ssm_qkv → causal conv1d + the q / k / v head split; ssm_z → the
gated RMSNorm per value head; attn_qg → q | gate per head + q_norm. A downstream fold is impossible because P16 mixes
heads inside every 128-column block. (MLP gate / up would need nothing, since permuting the intermediate dimension in
gate, up and down's columns is an exact model equivalence. The model predicts no gain there, though: gate / up 1.000,
down's input side 1.007 worse.)

**Option A · remap in the GEMM store (preferred).** The map is a closed form, so no table is needed:
stored column n = 128 b + 16 t + c → natural column 16 (b + nb·t) + c, with b = n >> 7, t = (n >> 4) & 7, c = n & 15.
- The A8 prefill path (production since 11-09, `plugin/exl3i/exl3i_8a8.hip.h:156-168`) writes one `uint4` (8 bf16)
  per thread at `y + m·pitch + n`. An aligned 8-column vector never crosses a 16-column tile, so each store stays one
  16 B vector. The cost is ~4 integer ops per store plus one kernel argument (nb, or a flag; nb = N / 128 is known).
- Coalescing changes: a wave's run of consecutive 16 B stores along n (up to 256 B for one row of a 128-column
  block) becomes 8 runs of 32 B, 16·nb·2 B apart. On RDNA4 these are still full 32 B sectors, so no read-modify-write
  and no extra bytes, but there are more memory transactions per wave in the epilogue. The output write is T·N·2 B
  against the weight read N·K·K_bits/8 plus the activations; it is a small share of a prefill GEMM's traffic at
  K ≈ 2.5–4, so the expected cost is near the noise. This must be measured, not assumed.
- Every kernel that writes these three tensors' outputs needs the same remap: exl3i_8a8 (A8 prefill), the nt / rows
  decode kernels (`exl3i_nt.hip.h` epilogue_block, `nt.c` / `rows.c` reference), `exl3i_rate.hip.h` (fractional K),
  and the CPU / reference paths used by the tests. It is a format bit per tensor (exl3i record or recipe option), so
  old containers stay valid.
- Decode (T = 1–8) is weight-bandwidth bound and writes T·N·2 B, so ~0 is expected.

**Option B · separate gather.** Write in stored order to a scratch buffer, then a permute kernel (or a fused
gather in the consumer's load). That costs one extra read + write of T·N bf16 per tensor and a launch: at T = 2048,
N = 10240 (ssm_qkv) about 84 MB of traffic, ~0.1 ms at ~640 GB/s, × 48 GDN layers ≈ 5 ms of qkv alone per 2k-token
prefill, before z / attn_qg. The gather is much worse than A unless it is fused into an existing consumer (e.g.
the conv1d load for ssm_qkv), and that fusion is tensor-specific work in three different consumers. In decode it is a
launch per tensor per token (≈ 112 extra launches / token), which is not negligible. B is a fallback only.

**How to time it (plan; the lead / user decide whether to run it):**
1. **Prefill microbench, card 0:** the A8 GEMM (exl3i_8a8) with the store remap behind a compile-time switch (remap
   off = byte-identical to today: assert it), on the three shapes [N × K] = 10240 × 5120 (ssm_qkv), 6144 × 5120 (ssm_z),
   12288 × 5120 (attn_qg), at K2.5 and K4 codes, T ∈ {512, 2048, 8192}. Median of ≥ 20 timed runs after warm-up,
   arms interleaved (ABAB) to cancel clock / thermal drift. Report µs per call and the ratio per shape; plus
   correctness: remap-on output == the remap-off output gathered on the host, bit for bit.
2. **Prefill end to end (the scored quantity):** radiance prefill tok/s on one mix packed twice (natural, and P16 with
   the format bit on the 112 tensors), same codes otherwise, the challenge's prompt-length mix; `gemm_us` summed over
   the 112 tensors from the profiler. A8 serving (the shipping mode).
3. **Decode:** the same two mixes, decode tok/s at batch 1 and 8 (the rows / nt kernels), context 4k and 32k.
4. **Option B for reference:** a standalone permute kernel on the same T × N shapes (µs), plus the e2e delta of the
   unfused gather on the qkv tensors alone.
Suggested adoption bar (for the lead to fix before the timing): prefill `gemm_us` of the 112 tensors ≤ +1% and
e2e prefill tok/s within noise; decode tok/s within noise.

**Runtime class:** prefill and decode (a store-side remap in every kernel writing these tensors; ~0 expected for A,
measurable for B). Quantizer side: none (π is fixed, and the quantizer sees a permuted checkpoint + H_out).
