# v3/df2rr — DFlash2 rescoring against OUR main head (#180-02, w155)

Radiance fork branch `qc-exl3-v3-rr` (radiance-hero, worktree `/workspace/wt/w155-radiance`), two commits on
qc-exl3-v3 fdd480c (rebased from 9f0a059 per the lead's 11-13 11:00 PM note): **158b1f8** (the op, the block hook,
qwen35_exl3, tests) and **b83698d** (the ring window as one `v_alignbit`). Not pushed.

w153 (#180) rescored the drafter's 32 coarse candidates against a 1.2 GiB int8 copy of the original lm_head
(`dflash.rerank_head`), because logit_rerank reads plain / fp8 / int8 rows only. This branch rescores against the
trunk's own exl3i head, read where it already lies:

- **`exl3i_rerank`** (exl3i plugin; interface `plugin/exl3import/include/exl3i_rerank.h`): logit_rerank's semantics
  over the head's record / suh / svh at exl3i_dense's operand positions, so exl3i_dense's layout hooks serve it and the
  head is stored once (same layout tags; the builder refuses two stored forms of one weight). One workgroup a (row,
  candidate): the row rotated exactly as exl3i_dense's `rotate128` (f16(x·suh), the f32 butterfly, f16(v·RS) in ONE
  rounding — the device compiler fuses that multiply and conversion into `v_fma_mixlo_f16`, measured: the
  double-rounded form differs by 1 ulp on ~1 feature in 20,000), the candidate's 128-row output-Hadamard block decoded
  from the ring (R16 row order from the record's stored dtype; mul1i8 / mad as grid indices × the step after the sum,
  as nt_pass's DQ), f32 dots, the cone of element j × RS × svh; then a per-row sort (larger first, ties to the lower id,
  pads last; in place allowed). All four codebooks, any rate.
- **`rerank_ref.c`**: the host reference, bit for bit the device row (`exl3i_rerank_ref`, `exl3i_rerank_ref_rotate`).
- **`arch/common/rad_block_dflash2.h`** (shared): a plugin-declared rescoring op (`op_rr_ext`, `rr_w[3]`) issued where
  logit_rerank would be; defaults 0, so qwen35_fp8 & co are unchanged.
- **qwen35_exl3**: with an exl3i head and no `dflash.rerank_head` line, `exl3i_rerank` over output.weight's own planes
  (32 candidates → the selector's best 8). A copy, when present, still wins. `QWEN35_EXL3_RERANK=0` turns main-head
  rescoring off (the coarse head's own scores). The drafter's plugin note (GET /graph `notes`) says which:
  "2-bit head, its top 32 rescored against output.weight itself (exl3i_rerank, no copy)".

## Files

| file | what |
|---|---|
| `build155.sh` | ds8, CPU: exl3i + exl3quant + qwen35_exl3 from `W/src` against an overlay of rad10-perf/install whose `rad_block_dflash2.h` is the branch's (the installed one is 9f0a059's, checked); host tests; `W/home` |
| `run155.sh` | ds8 HIP 7, restartable stages `gtest kld kld7 note serve serve2 acc cmp` (header has the details) |
| `gdbg155.sh`, `exp155.sh` | the debugging / kernel-experiment loops (test_rerank on HIP 7 under the lock) |
| `headprobe.py` | CPU: the served K4 head vs the original lm_head (and an int8 g128 copy) on x ~ N(0, H_calib) and isotropic x |
| `bsum155.py` | bench + fixed-prompt acceptance summaries (`python3 bsum155.py data/res-pre-rebase`) |
| `data/res` | the final (rebased, b83698d) build: gtest, kld, notes, fixed-prompt acceptance (byte-identical to the pre-rebase runs); `data/res-pre-rebase` the 9f0a059-based runs (bench rounds 1–6, the first gtests, the kernel experiments `exp-*`); `data/headprobe-K4.json`; `data/logs` |

Packs (w153's, ds8 `/data/q27/w153/packs/`): **P3-mx** (K4.125 mul1i8 a8g128 trunk, R16s003 K4 mul1i8 head, bf16
embedding, MXFP4 drafter, no copy) for main / none, **P1-mx-rr** (the same + the int8 copy) for copy. No mad pack with
a drafter exists (w150's mad packs on box2 have no drafter and a bf16 head; there is no mad-quantized head yet), so mad
is covered by the device test on real-size R16 heads (synthetic codes), not end to end.

## Results

See LOG #180-02 (2026-11-14). In short: device == host reference bit for bit; vs exl3i_dense's own logits at those
rows ≤ 0.1 u·L1 (f32 rounding; the summation bound is K u·L1); KLD rows unchanged (KL mode refuses a drafter, so
"drafter on" cannot be scored there; greedy text is the check); +1.20 GiB KV vs the copy (15.81 vs 14.61 GiB);
acceptance main 3.95 tok/step vs copy 4.23 vs none 3.82 on fixed prompts; op 0.43–0.46 ms a draft pass at batch 1
(7 rows × 32 candidates).
