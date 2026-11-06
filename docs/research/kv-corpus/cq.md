# Coupled Quantization — "KV Cache is 1 Bit Per Channel" (2024) · arXiv:2405.03917

**Idea.** Channels of a key/value vector are strongly dependent (joint entropy grows slower than the sum of
marginals), so quantize **c channels jointly** with a learned codebook (vector quantization over channel groups);
at equal bits per value, more coupled channels = less error.

**Method.** Split each K/V vector (per layer, per channel group) into c-channel sub-vectors; learn 2^b centroids per
group (k-means, optionally Fisher-weighted from calibration gradients). "CQ-ccbb": c channels share one b-bit code →
b/c bits/value (4c8b = 2 bits, 8c8b = 1 bit). No per-token scales.

**Numbers.** WikiText-2 PPL (Table 1), LLaMA-7b / LLaMA-2-13b / Mistral-7b: FP16 5.68 / 4.57 / 4.76; CQ-2c8b (4 b)
5.70 / 4.59 / 4.79; **CQ-4c8b (2 b) 5.97 / 4.81 / 5.11** vs INT2 11779 / 3942 / 573, KVQuant-2b 8.17 / 29.25 / 7.33,
KVQuant-2b-1 % (2.32–2.35 b) 6.06 / 4.92 / 5.16; **CQ-8c8b (1 b) 8.09 / 6.55 / 7.25**; CQ-8c10b (1.25 b) 6.78 / 5.47 /
5.90. Ablation at 2 bits (Table 4), LLaMA-2-13b: c = 1 / 2 / 4 without Fisher 890.42 / 171.96 / 6.62, with Fisher
6.06 / 4.91 / 4.81. Centroid learning 4–94 min per model; centroids 0.23–1 % of weight size (Table 5).

**Runtime cost.** Codebook lookups per c-channel chunk at decode (or q·centroid tables: precompute q·c for all 2^b
centroids per chunk and gather → "asymmetric distance computation"); offline k-means with Fisher information.

**Failure modes.** Only perplexity and zero-shot (WinoGrande, PIQA, ARC) at short context; no long-context or
retrieval tests. Codebooks are per layer/channel-group and data-fit; distribution shift on unseen domains untested.

**New vs #416.** VQ with coupled channels on the **device** tier. Code 000 tested VQ codebooks for V only (−30 %
proxy error vs scalar at 3.5 bpw, still above int4 at 4.5) and kept VQ for the host tier only (IKV Table 9). CQ's
regime is ≤ 2 bits where scalar codes collapse — a regime #416 avoids.

**Radiance fit.** Per-token codes fit pages (8-bit index per c channels). Decode cost on gfx1201: score via a per-step
LUT of q·centroid (256 entries × 256/c chunks per KV head = 64 KB at c = 4 in LDS, rebuilt per query group) then
gathers — no WMMA; V side needs centroid gathers into an f16 accumulate. Feasible at decode (bandwidth-bound), poor
for prefill.
