# KV directions for radiance on R9700 / 9070 XT (gfx1201) — ranked, 2026-11-06

Target hardware: gfx1201 has f16 / bf16, fp8 (E4M3/E5M2), int8 (`iu8`) and int4 (`iu4`) WMMA, **no FP4**; 640 GB/s
on both cards; 32 GB (R9700) or 16 GB (9070 XT). Target model: Qwen3.8-27B unless noted (16 dense gated-attention
layers, gqa 6, head 256, partial RoPE on 64 dims, 16-token pages; **KiB/token = 4 × bits/value**, see INDEX.md).
Everything below is a lead for the autoresearch agents, not a measured result; citations point to the corpus pages.

**Why the 27B differs from #416's Flash-Next.** Attention is dense, so the whole cache is read every step: at one
128K sequence fp8 KV is 4 GiB against ~19 GiB of MXFP4 weights (KV ≈ 17 % of decode bytes), at 262K ≈ 30 %, and with
8 × 128K sessions KV (32 GiB) is most of it. Bits therefore buy speed at long context and batch, not only capacity
(cf. hybrid-sparse-decode.md: profiled on an A100 at 64K, 27.7 of the 27B's 47.1 ms per step were not weight reads). On a 16 GB
9070 XT every free GiB holds 32K fp8 tokens, 58K at 4.5 bits or 75K at 3.5 bits.

## What the corpus says about the judge (before any idea)

1. **Score decode-like, not prefill-like.** KVarN shows errors accumulate when later tokens are computed from the
   *quantized* cache ("pseudo-decode": quantize after every block of b tokens; kvarn.md). Check what radiance's KL
   mode does inside a prefill chunk; screen at small chunks (Code 000's floor run used 64) or block pseudo-decode.
2. **Free-running divergence is most of the damage.** Post-divergence exposure is 85–90 % of mismatch gaps
   (divergence-timing.md); 2-bit transforms can make reasoning never end (WUSH-KV: 100 % length-cap hits for a
   TurboQuant-style transform). Add first-divergence position, post-divergence mismatch and a length-cap rate on
   paired greedy runs vs a bf16-vs-bf16 (re-chunked) noise baseline.
3. **Agentic and multi-turn gates catch what RULER misses.** On the 27B, fp8 KV loses 4.8 SWE-bench-Verified points
   and 2.6 τ² pass^1 (flagrot.md); Quest page selection keeps GPQA/AIME but drops SWE-V 82.7 → 4.7 (jet-qsparse.md);
   SnapKV scores 100 on LongBench and 0 on every SCBench turn (scbench.md); eviction can expose refused private data
   (kvzip.md). A cheap proxy we own: replayed agent traces (our own sessions; private, scored on our own cards only) scored by
   teacher-forced KL *and* by tool-call exact match, plus SCBench-style multi-question string retrieval at 128K.
4. **Deterministic scoring.** NIAH with an LLM judge and filesystem-ordered haystacks is not reproducible
   (rabitq-vs-turboquant.md): fixed haystacks, string-match scorers.
5. **Speed at the right operating point.** Per-request and system throughput diverge once the cache limits batch
   (saw-int4.md); measure bytes/token, TPOT at batch 1 and at the max batch that fits, at 32K / 128K / 262K
   (and 512K with YaRN), and drafter (DFlash2 / MTP) acceptance (radiance-416.md: 0.63 → 0.41 from one layer's V).

## The ten directions

### 1. Port #416 to the 27B and retire fp8 as the default (i8h + per-layer recipe)
- **Why:** the 27B's default cache is fp8, and fp8 measurably hurts it on agentic work (SWE-V −4.8, τ² −2.6;
  flagrot.md — vLLM's FP8 cache; radiance's E4M3 has per-(sequence, head) descales, so re-measure). On Flash-Next i8h sits at the noise floor (+0.0016 vs fp8 +0.0027) at the same bytes, and per-layer
  recipes reach fp8 quality at −22 % and floor+0.01 at −44 % (radiance-416.md). The 27B's output-gate means (0.022–0.459
  across layers, gatesight.md) are a free prior for the layer ordering.
- **Bytes/token:** i8h 32.25 KiB (same as fp8); a mixed recipe (early layers i8h, late k6_v4 / k5_v4 rot+vmean) ≈
  22–26 KiB.
- **Kernels:** fp8_v4 / k6_v4 already exist for h256/gqa6/16-token pages; k5_v4, k4h_*, i8h, rotation and vmean have
  libr4d rows only at Flash-Next's geometry (gqa 12 / 6, block 4) — new h256/gqa6/block16 variants.
- **Tail / RULER risk:** low for i8h; per-layer narrowing must pass the agentic and drafter gates (DFlash2 cache stays
  fp8 per GUIDE; check its acceptance anyway).

### 2. Calibrated selective rotation + key centering instead of all-or-nothing Hadamard
- **Why:** on the 27B, whole-head Hadamard cuts INT4 KL only 6.7 → 6.2 (×10³), while a calibrated base reaches 5.2 and
  FlagRot-L 4.4 (flagrot.md Table 25); on Qwen3-8B Hadamard alone gives 43.5 → 31.2. Consistent with the 27B's
  qk-normed keys being already flat, so the remaining gains are in calibrated, query-weighted choices. Key centering is the K-side twin of #416's V mean (nsnquant.md, jet-qsparse.md).
- **Bytes/token:** +0 (flag in the group's scale word or a sign bit); makes k4h_v4 (18 KiB) closer to k6_v4.
- **Kernels:** q rotated per 32-group (H₃₂ inside a WMMA tile), per-group flag; a static K mean subtracted at store
  is exact for free: q·(k − μ) = q·k − q·μ shifts every score of a query by the same constant, which softmax ignores
  (nothing to add back; unlike V mean).
- **Risk:** calibration domain (use calib fit / qad pools, not verify); per-key decisions cost ~5 % prefill.

### 3. Fused vpred (and a fused V-mean / K-mean path)
- **Why:** vpred halves k4h_v2h's KL above the floor and lowers k4h_v4's by a fifth (radiance-416.md), but needs a
  second attention pass. AQUA-KV shows V-from-K prediction holds at 2 bits (aqua-kv.md). With gqa 6 the extra term
  P·K̂ is a [6 × 256] accumulator per KV head — it fits in registers next to P·V_res in the same pass.
- **Bytes/token:** k4h_v4 vpred 18 KiB, k4h_v2h vpred 14 KiB.
- **Kernels:** one decode/prefill kernel family with a second accumulator and a 256×256 map per layer/head (22 MB
  calibration file on Flash-Next).
- **Risk:** 2-bit V failed recall significantly even with vpred (ΔNLL +0.024 on Flash-Next's recall corpus) and
  broke drafter acceptance on the last layer; keep 2-bit V off the last layer and off early layers.

### 4. High-precision recent window (and sink, if the 27B has one) around a narrow body
- **Why:** the recent window is the single largest gain in SKVQ (+10.2 LongBench points over RTN; skvq.md); every
  2-bit method that holds keeps a 16-bit recent window (KIVI) or sink + recent window (KVarN, WUSH-KV, FlagRot S4W128), and DeepSeek ships
  its sliding-window KV in FP8 "due to its sensitivity" next to an FP4 body (deepseek-v4-1-flash.md). The 27B's gated
  attention removes most of the sink (46.7 % → 4.8 % first-token attention in the Qwen gated-attention study;
  gated-attention.md) — measure before paying for a sink group.
- **Bytes/token:** body + W × 64 KiB / context: W = 256 adds 0.125 KiB/token at 128K (≈ nothing).
- **Kernels:** a per-sequence bf16 tail group (the shape #412 already uses for sink + tail), transcode of a page when
  it leaves the window (pages change bytes once → prefix hashing on the packed form; spec rollback stays in the tail).
- **Risk:** low for tails; it is what makes ideas 5, 6 and 8 safe at ≤ 3 bits.

### 5. Separate K and V widths per layer (and per KV head) in the knapsack
- **Why:** K vs V sensitivity flips between models (Llama-2: K2V4 8.06 ≫ K4V2 5.75 PPL, quarot.md; Qwen2.5-7B: K4V8 PPL
  221 vs K8V4 9.4, kvtuner.md; Flash-Next: V is the bottleneck, radiance-416.md). #416 offers fixed K/V pairs; KVTuner
  treats them as separate items. The 27B has only 64 KV heads in total, so a per-head screen (DuoAttention-style
  retrieval vs streaming heads, duoattention.md) is cheap.
- **Bytes/token:** e.g. k8_v4 (i8h-style K 8.06 b + int4 V 4.5 b) ≈ 6.3 b → 25 KiB; k4_v8 the same; per-head mixes
  in between.
- **Kernels:** one layout per chosen pair (keep the menu small); per-head layouts = two KV groups per layer.
- **Risk:** non-additive costs (Code 000) → keep rad-autoquant's verify-and-rescale; per-head choices can overfit a
  4-document corpus — screen on the search pool, confirm on verify.

### 6. Demote, don't evict: per-page precision tiers from a utility score
- **Why:** keeping "evicted" tokens at INT4 instead of dropping them restores retrieval 43.2 % → 100 % (mikv.md);
  ZipCache's saliency keeps 200-line retrieval at 100 % at 4.43× (zipcache.md). On the 27B, GDN gate signals predict
  KV utility before any query (gatesight.md, 0.25 s at 256K), and a model-self-scored context (KVzip) is query-agnostic.
  Code 000 found attention-count importance no better than a position hash for demotion on QSA — GDN signals were not
  tried.
- **Bytes/token:** e.g. 30 % of pages i8h + 70 % k4h_v4 → 22 KiB; with k4h_v2h for the cold 70 % → 19.5 KiB. Unlike
  QSA, the 27B reads every page every step, so this also cuts decode bytes.
- **Kernels:** two KV groups + a per-page layout tag (the #412 tiered-precision machinery); scoring op after prefill.
- **Risk:** query dependence across turns (SCBench), and 2-bit V on demoted pages is exactly where recall fails; gate
  on multi-turn string retrieval.

### 7. Sparse decode for the dense 27B, then #412 residency
- **Why:** D-LSA keeps the 27B within 5 RULER points of full attention to 1M and beats it at 10M (63.9 vs 59.9) with a
  352M-token distilled indexer (a training-free PCA variant is also offered; d-lsa.md); a dense low-bit q·k scan + top-64 blocks keeps
  SWE-V at 83.5 vs 82.7 with 1.94× decode at 128K / batch 16 (jet-qsparse.md); per-layer window + budget recipes for the
  Qwen3.5-27B layout keep 21-dataset averages (hybrid-sparse-decode.md). With a selection, #412 can page the unread
  middle to host memory — on a 16 GB 9070 XT the context limit becomes host RAM.
- **Bytes/token on card:** a D-LSA-style index ≈ 0.5 KiB/token (16 layers × one 64-dim bf16 index key per 4 pooled
  keys; our arithmetic) or, for a Jet-QSparse-style dense scan, the int4 K half of a k4 layout ≈ 9 KiB/token; plus
  sink/tail pages. V and cold K pages live in host memory.
- **Kernels:** radiance's QSA path (score → select → page → paged attention) with a new indexer; on gfx1201 the dense
  scan runs as int8-q × int4-K on `iu8` WMMA.
- **Risk:** the highest of the ten — training-free page selection (Quest) collapsed the 27B on SWE-V (4.7 vs 82.7);
  decode gains are small at batch 1 below ~256K (D-LSA: TPOT 0.87–1.00× at 512K–1M on H200 TP8). Agentic + SCBench gates
  are mandatory.

### 8. Trellis / lattice-coded keys at 2–3 bits, reusing our weight-quant decoders
- **Why:** E8-lattice keys at 2.25 bits beat scalar 3-bit keys (PPL +0.23 vs +0.60, e8-keys.md); CQ/NSNQuant show coupled
  channels beat scalar codes below 3 bits (cq.md, nsnquant.md). We already run E8P / QTIP-trellis decoders for weights.
  FlagRot's 2-bit KIVI result on the 27B (LongBench unchanged) says the 27B may tolerate 2-bit K better than Llama.
- **Bytes/token:** K 2.25 + V int4 with scales 4.5 → 3.4 b → 13.5 KiB; K 2.25 + k4h_v2h's 2.5-bit V (vpred) → 2.4 b →
  9.5 KiB.
- **Kernels:** codebook decode to f16 in the attention kernel (E8P: 256-entry table + sign bits, LDS-resident) — no
  integer WMMA; decode is bandwidth-bound so ALU cost is acceptable at batch 1, less so at batch 16.
- **Risk:** plain Hadamard at 2-bit K is catastrophic on Qwen3 (PPL 258, wush-kv.md); E8 was tested only to 16K; VQ for V
  lost to int4 on Flash-Next. Needs the window of idea 4 and RULER-128K / MRCR gates.

### 9. A store-tier codec for the prefix cache and idle agent sessions
- **Why:** multi-turn agent work reuses long prefixes; kvtc stores KV at ~16× with lost-in-the-middle 99.3 vs 99.4
  (kvtc.md); DCKV, built for hybrid Qwen models, reaches 57× at −0.45 RULER-32K (dckv.md). #412's store holds pages at
  pool precision.
- **Bytes/token (stored):** 64 KiB bf16 → ~4 KiB at 16×, ~1.1 KiB at 57×; live-pool bytes unchanged.
- **Kernels:** PCA back-projection matmul + entropy decode on promotion (GPU ANS/Huffman for gfx1201), or CPU decode on
  the host side for idle sessions; GDN state stored alongside at each reuse point.
- **Risk:** sinks must be exempt (kvtc at 64× without them: LITM 0.0); decompression on the TTFT path; codec fit on
  in-domain data.

### 10. GDN state at f16 or lifetime-aware 6-bit
- **Why:** the 27B's 48 GDN layers hold ≈151 MB of f32 state per sequence (our arithmetic) — at 32K context that is ~15 %
  of an fp8 KV sequence and dominates for short agent turns. #416 shipped an f16 state on Flash-Next; STEPQuant keeps
  the 27B at fp32 accuracy with 4–6-bit states, where uniform INT8 loses 8.7 points on long generations (stepquant.md).
- **Bytes/sequence:** 151 MB → 75 MB (f16) → ≈30 MB (6-bit).
- **Kernels:** GDN recurrence read/write-back in the packed format; prefix checkpoints in the same format.
- **Risk:** errors persist in long-lived (slow-decay) heads — a long-generation tail that short benchmarks hide
  (INT8: −1.5 short vs −8.7 long); f16 anchor replay already made text depend on draft acceptance on Flash-Next.

## Considered and ranked lower (evidence against, or outside a runtime-option loop)
Token eviction / merging (H2O, SnapKV, PyramidKV, DMS, DistillCache): irreversible, conflicts with page hashing, fails
multi-turn. Low-rank / latent (Palu, xKV, MLA, MHA2MLA): retrieval collapses first, or needs training; Code 000's spectra
show little slack. TurboQuant / PolarQuant / QJL as the main K: lose to group-scaled INT4 on Qwen-family models,
including the 27B. Cross-layer KV or index reuse: needs training (DeepSeek-V4.1) and failed untrained on Flash-Next.
KVLinC's linear-attention correction state and GEAR's low-rank residual: interesting zero-ish-byte correctors but
trained / per-sequence state; revisit if ideas 3–4 stall.
