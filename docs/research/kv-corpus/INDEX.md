# KV-cache idea corpus — index (2026-11-06)

Read-only context for a future KV autoresearch task (BACKLOG: "Autoresearch on KV quantization"). One page per work
(`<slug>.md`: method, numbers with setting, runtime cost, failure modes, new vs #416, radiance fit); the ten
directions we think most promising are in [IDEAS.md](IDEAS.md). Every number is quoted from the cited paper (arXiv
id on each page) or, for radiance, from PR #416 / #412 and Code 000's public paper "Infinite KV Cache" (IKV); see
[radiance-416.md](radiance-416.md), the baseline every "new vs #416" refers to; the IKV paper's own page is
[infinite-kv-cache.md](infinite-kv-cache.md), and how a new format enters `rad-autoquant` and the kv-task judge is in
[AUTOQUANT.md](AUTOQUANT.md). Built with our paper
pipeline (arXiv sweep of 19 KV queries since 2024-01 → 1 006 candidates → triage by title/abstract → 52 fetched
full texts → digests).

**Bytes per token on Qwen3.8-27B.** 16 full-attention layers × 4 KV heads × 256 dims × (K + V) = 32 768 values per
token, so **KiB/token = 4 × bits/value**: bf16 64, fp8 / i8h 32, k6_v4 22, k4h_v4 18, k4h_v2h 14, 2.5 bits 10.
At 128K one sequence holds 4 GiB of fp8 KV (8 GiB at 262K) next to ~19 GiB of MXFP4 weights; the attention is dense,
so every decode step reads all of it (unlike Flash-Next's QSA). The GDN state adds a fixed ≈151 MB per sequence in f32
(our arithmetic: 48 layers × 48 heads × 128 × 128 × 4 B).

**Columns.** *bits* = bits per cached value as reported (incl. scales where the paper counts them). *Quality* =
the most informative published number, long-context / needle / RULER / agentic where available ("27B" marks results
on Qwen3.8-27B or its Qwen3.5 twin). *Overhead* = extra passes, calibration, training, kernel needs. *Paged fit* =
fits radiance's per-token rows in 16-token pages? **y** (drop-in row layout or op), **tile** (needs multi-token tiles
+ an fp16 tail group), **tier** (host / store tier, not the live pool), **sel** (needs a selection op over pages,
QSA-style), **n** (model change or conflicts with page hashing / prefix cache).

## Baseline (radiance, Code 000)

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [Infinite KV Cache](infinite-kv-cache.md) | 2026 | QSA window residency + per-layer / per-tier layouts chosen by measured search (rad-autoquant) | 3.5–16 per layer; host tier k4h_v4+vm | compact 3.76 KiB/token (−44 % vs fp8), ΔNLL +0.0008; host tier −44 % for +0.0049 above floor; **floor 0.0243 is MoE noise (dense bf16 2B: 0.00018)** | search ~70 s a run, 112–113 recipes a limit; pager 3–15 % decode | y (layouts) / tier (demote) |

## Quantization: scalar, rotation, mixed precision

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [KIVI](kivi.md) | 2024 | K per-channel, V per-token, 2-bit, fp16 residual tail | ~3.0–3.2 eff. | LongBench ≈ fp16; **RULER-32K 39.8 vs 56.4** (KVQuant's run) | residual buffer, packing kernel | tile |
| [KVQuant](kvquant.md) | 2024 | pre-RoPE per-channel K, Fisher-weighted non-uniform codes, 1 % sparse outliers | 2.33–4.35 | RULER-32K 53.7 (3-bit) vs 56.4; 1M ctx on one A100 | k-means calib, sparse outlier path | y (codes) / n (outliers) |
| [SKVQ](skvq.md) | 2024 | channel reorder + clip, recent 128 tokens + sink kept fp16/fp8 | 2.25 K / 1.5 V | window alone +10.18 LongBench pts over RTN (13B) | offline reorder/clip | y (+tail group) |
| [GEAR](gear.md) | 2024 | quant + low-rank residual + sparse outliers | 2–4 (+r=4) | CoT avg 38.34 at 2-bit vs 25.25 KIVI, 40.52 fp16 | power-iter SVD per flush | n (per-seq factors) |
| [ZipCache](zipcache.md) | 2024 | per-token 4/2-bit by normalized-attention saliency | 4/2 mixed | 200-line retrieval 100 % at 4.43× (MiKV 90 %, H2O 0 %) | probe attention (10 % queries) | page-level only |
| [MiKV](mikv.md) | 2024 | "demote, don't evict": unimportant tokens in INT2–4 | mixed | line retrieval 43.2 % evicted → 100 % INT4-retained | attention-score tracking | page-level only |
| [QuaRot (KV)](quarot.md) | 2024 | Hadamard on q/k after RoPE, V rotation folded in weights | 4 | K4V4 PPL 5.51 vs 5.47 (Llama-2-7B); K2V4 8.06 ≫ K4V2 5.75 | online WHT on q, k | y (in radiance) |
| [SpinQuant (KV)](spinquant.md) | 2024 | learned (Cayley) rotations incl. per-head V rotation | 4 | W4A4KV4 64.0 vs W4A4KV16 64.1 avg (Llama-2-7B) | rotation training (GPU hours) | y (folded) |
| [KVTuner](kvtuner.md) | 2025 | per-layer (K bits, V bits) pairs by MOO search | 3.25–4 avg | Qwen2.5-7B: K4V8 PPL 221 vs K8V4 9.4; LongBench KV4 0.634 → tuned 0.796 | offline search | y |
| [AQUA-KV](aqua-kv.md) | 2025 | predict K from layer l−1, V from K; quantize residual (HIGGS VQ) | 2.09 | LongBench 44.30 vs 44.61 (Llama-3.2-3B); KIVI 39.64 | 1–6 h calibration; cross-layer reads | y (V-from-K) / n (cross-layer) |
| [KVLinC](kvlinc.md) | 2025 | Hadamard V + trained linear-attention correction of key error | 2.71 | RULER-8K 51.1 vs 41.0 KIVI vs 85.0 bf16 (Qwen2.5-3B) | trained adapters, d×D state/head | y (+state) |
| [KVarN](kvarn.md) | 2026 | Hadamard + Sinkhorn dual-axis variance normalisation, 2-bit | 2.3 | AIME24 60.0 vs 61.1 fp16 (Qwen3-4B); pseudo-decode eval | per-128-token-tile normalise (0.18 %) | tile |
| [WUSH-KV](wush-kv.md) | 2026 | data-adaptive per-head transform from both factors' statistics | 2 (+sink/recent bf16) | **RULER NIAH-128K 57.3 vs 79.2 bf16; MRCR 9.6 vs 25.3** (Qwen3-8B) | stats calibration; d×d K transform | y |
| [SAW-INT4](saw-int4.md) | 2026 | token-wise INT4 + block Hadamard fused in the store kernel | 4 | **plain INT4 = 0.00 on all tasks**, BDR-128 73.8 vs 75.6 (Qwen3-4B-Thinking) | none (fused) | y |
| [FlagRot](flagrot.md) | 2026 | rotate each 32-ch piece or not, by calibrated attention error | 2.5 / 4.5 | **27B: 2-bit KIVI = bf16 on LongBench; fp8 KV −4.8 SWE-V, −2.6 τ²; Hadamard only −7 % KL at INT4** | calibration, flag bit per group | y |
| [BitDecoding](bitdecoding.md) | 2025 | tensor-core decode for low-bit KV: fragment layouts, GQA q-packing | 2–4 | 3× lower single-batch latency at 128K (Llama-3.1-8B) | kernels only | y |

## Vector / lattice / codebook quantization

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [CQ](cq.md) | 2024 | couple c channels into one learned codebook | 1–2 | 2-bit PPL 4.81 vs 4.57 fp16 (Llama-2-13B); no long-ctx test | Fisher k-means, LUT decode | y (gather) |
| [QJL](qjl.md) | 2024 | JL projection + sign bits, unbiased inner product | 3 | LongBench ≈ fp16 (Longchat-7B-32K) | random projection per token | y (indexer key) |
| [TurboQuant](turboquant.md) | 2025 | random rotation + Lloyd–Max per coordinate (+QJL residual) | 2.5–3.5 | NIAH 0.997 (Llama-3.1-8B); **27B INT4-class KL 8.2 vs 6.7 RTN** (FlagRot) | LUT decode | y (but loses) |
| [RaBitQ vs TurboQuant](rabitq-vs-turboquant.md) | 2026 | symmetric re-evaluation: uniform-grid RaBitQ wins; TurboQuant claims not reproducible | 2.5–3.5 | LongBench-E 48.64 vs 47.78 at 2.5 b | — | y |
| [PolarQuant](polarquant.md) | 2025 | quantize polar angles after rotation, no scales | 3.3–3.9 | LongBench 48.11 vs 48.63; AIME24 52.2 vs 61.1 (Qwen3-4B, KVarN) | trig per element, 4× prefill (ref impl) | y (slow) |
| [CommVQ](commvq.md) | 2025 | additive VQ with RoPE-commuting key codebook | 1–2 | GSM8K 76.04 vs 76.27 at 2 b (Llama-3.1-8B) | EM-trained codebooks, encoder | y (gather) |
| [NSNQuant](nsnquant.md) | 2025 | normalize–shift–normalize + Hadamard → one Gaussian VQ codebook | 1.23–2.23 | C4 PPL 9.15 vs 8.43 at 2.23 b (Llama-3.1-8B) vs KIVI 15.80 | running channel mean, VQ encode | y |
| [E8 keys](e8-keys.md) | 2026 | E8-lattice VQ (2¹⁶ words / 8 dims) for keys, LUT scoring | K 2.25 / V 2 | PPL +0.23 vs +0.60 scalar k3v2 (Llama-3.1-8B, 4K) | lattice encode | y |
| [Block-GTQ](block-gtq.md) | 2026 | allocate key bits per RoPE frequency block by q/k energy | K2–3 avg | K2V2 NIAH 70.6 → 97.4 (Llama-3.1-8B) | short calibration | y (static sub-layouts) |

## Transform / entropy coding for stored caches

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [kvtc](kvtc.md) | 2025 | PCA + DP bit allocation + DEFLATE for stored KV | 16/(8–88×) | 16×: LITM 99.3 vs 99.4; **no sink exemption at 64× → LITM 0.0** | compress 379 ms / 8×8K, decompress 267 ms | tier |
| [DCKV](dckv.md) | 2026 | end-to-end trained codec for hybrid-model prompt caches | 57–109× | RULER-32K −0.45 at 214 B/token (Qwen3.5-2B) | trained codec ≈5 % of model | tier |

## Eviction, merging, importance

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [H₂O](h2o.md) | 2023 | keep heavy hitters + recent | 16 (20 % kept) | GSM8k CoT 1.67 vs 41.62 at 40 % (ZipCache's run) | attention accumulation | n |
| [StreamingLLM](streamingllm.md) | 2023 | 4 sink tokens + rolling window | 16 | PPL 5158 → 5.40 with 4 sinks; no long retrieval | none | n (sink idea: y) |
| [SnapKV](snapkv.md) | 2024 | select per head from end-of-prompt observation window | 16 (1–4K kept) | **SCBench turns: 0 vs 62–70** | window attention pass | n |
| [PyramidKV](pyramidkv.md) | 2024 | larger budgets in low layers | 16 | line retrieval 61 % at 7× vs 100 % (KVarN's run) | as SnapKV | n |
| [DuoAttention](duoattention.md) | 2024 | retrieval heads full, streaming heads sink+window | 16 | LongBench 40.21 vs 40.08 at 50 %; 3.3M ctx with quant | gate optimisation per model | per-head groups |
| [KVzip](kvzip.md) | 2025 | query-agnostic importance via context reconstruction | 16 (3–4×) | SQuAD 94.4 vs SnapKV 32.2 at 30 %; **reveals refused PII** | 2× prefill once | tier / page score |
| [DMS](dms.md) | 2025 | learned delayed eviction, 1K-step retrofit | 16 (4–8×) | NIAH 96.8 vs 96.4 at 4×; AIME24 73.0 vs 75.0 at 8× (Qwen3-8B) | 10–100 GPU-h per CR step | n |
| [DistillCache](distillcache.md) | 2026 | RL eviction policy with per-step KL reward | 16 (25 %) | LongBench 39.1 vs 41.5 at 25 % (Mistral-7B, ≤8K) | policy per step (1.4 ms) | n |
| [GateSight](gatesight.md) | 2026 | GDN gates predict KV utility in hybrids; per-layer budget ∝ output-gate mean | 16 (10–70 %) | **27B RULER-4K 76.6 at 10 % (KVzip 29.4); gate-proportional > PyramidKV** | small MLPs, 0.25 s at 256K | page score / layer prior |
| [SCBench](scbench.md) | 2024 | benchmark: shared-context multi-turn / multi-request | — | KIVI-2 string retrieval 12.0 vs 57.1 at ~125K | — | judge |
| [Divergence timing](divergence-timing.md) | 2026 | first-divergence + post-divergence exposure decomposition | — | post-divergence exposure = 85–90 % of mismatch gaps | paired rollouts | judge |
| [Gated attention](gated-attention.md) | 2025 | the 27B's output gate removes attention sinks | — | first-token attention 46.7 % → 4.8 %; RULER-128K 58.8 vs 31.7 (YaRN) | — | context |

## Low-rank and latent KV

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [MLA](mla.md) | 2024 | cache a 512-d latent + 64-d RoPE key | 576 el./layer | KV −93.3 %, throughput 5.76× (DeepSeek-V2) | pre-training | n |
| [Palu](palu.md) | 2024 | SVD-factor W_k/W_v, cache the low-rank latent | 16 / 3 (30–50 % rank) | **synthetic tasks 16.88 vs 44.87 at 50 %** (Mistral-7B) | rank search, reconstruct in kernel | n |
| [MHA2MLA](mha2mla.md) | 2025 | partial RoPE + joint SVD → MLA, fine-tune 0.6–1 % tokens | d_kv 16–64 | LongBench 25.1 vs 27.4 at −87.5 % (Int2: 21.2) | fine-tuning | n |
| [xKV](xkv.md) | 2025 | shared cross-layer SVD basis per prompt | 16 (8×) | RULER-64K 88.5 vs 91.9; single-layer SVD 45.7 | per-prompt SVD | n |
| [ShadowKV](shadowkv.md) | 2024 | low-rank pre-RoPE K on GPU, V on CPU, landmark-selected chunks | 16 | RULER 86.9 vs 86.7; 2.97× throughput at 122K | SVD at prefill, PCIe V fetch | sel + tier |
| [DeepSeek-V4.1-Flash](deepseek-v4-1-flash.md) | 2026 | CSA2 cross-layer KV/index reuse + QAT FP4 KV; SWA window kept FP8 | ~4.5 (E2M1+E4M3/16) | 890 B/token global KV | training | n |

## Sparse attention as KV reduction

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [Quest](quest.md) | 2024 | per-page key min/max bound → top-K pages per query | 16 | passkey 99 % at 64-token budget; **27B SWE-V 4.7 vs 82.7** (Jet-QSparse's run) | page metadata, select | sel |
| [NSA](nsa.md) | 2025 | trained compression + selection + window | 16 | LongBench 0.469 vs full 0.437 (27B-param model, pretrained) | pre-training | n |
| [DSA](dsa.md) | 2025 | FP8 lightning indexer, top-2048 tokens | 16 | parity with dense on long-ctx evals | indexer warm-up 2.1B tok + 944B sparse training | sel (QSA-like) |
| [D-LSA](d-lsa.md) | 2026 | distilled latent index + bucket centroids (or PCA, training-free) | 16 | **27B: RULER within 5 pts to 1M; 63.9 vs 59.9 at 10M** | 352M-token distillation | sel |
| [Hybrid sparse decode](hybrid-sparse-decode.md) | 2026 | per-layer window + Quest budget found by NSGA-II | 16 | **Qwen3.5-27B avg 53.9 vs 53.8**, decode 1.11× at 64K (A100) | search (~520 GPU-h incl. eval) | sel |
| [Jet-QSparse](jet-qsparse.md) | 2026 | dense 4-bit q·k scan, top-N 128-token blocks, FP16 V | 4.625 | **27B N=64: SWE-V 83.5 vs 82.7, decode 1.94× at 128K** | FP4 MMA (Blackwell), key centering | sel |

## Recurrent state (the 27B's other per-sequence memory)

| work | year | idea in a line | bits | quality (setting) | overhead | paged fit |
|---|---|---|---|---|---|---|
| [STEPQuant](stepquant.md) | 2026 | lifetime- and key-row-aware GDN state quantization | 4–6 | **27B long-gen avg: INT8 71.9, STEPQuant@4 80.5 vs fp32 80.6** | calibration, GDN kernel | state (not paged) |
