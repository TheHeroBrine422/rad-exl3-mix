# MLA — Multi-head Latent Attention (DeepSeek-V2, 2024) · arXiv:2405.04434 (architecture)

**Idea.** Cache one low-rank **latent** c_t^KV (d_c dims) per token and layer instead of per-head K and V; up-project
at use (absorbable into W_q / W_o), with a small decoupled RoPE key (d_h^R dims) shared across heads because RoPE does
not commute with the absorption.

**Method.** c_t^KV = W^DKV h_t; k, v = W^UK c, W^UV c; W^UK absorbed into the query side and W^UV into the output so
attention runs in latent space; RoPE on a separate 64-dim key part. Cache = (d_c + d_h^R) · l elements per token
(DeepSeek-V2: d_c = 512, d_h^R = 64 → 576 per layer), "equal to GQA with only 2.25 groups" yet stronger than MHA
(paper's ablations, App. D).

**Numbers.** DeepSeek-V2 vs DeepSeek 67B: **KV cache −93.3 %**, max generation throughput 5.76×, training cost −42.5 %.

**Runtime cost.** Architecture (pre-training); decode attention in latent space is compute-heavier per byte (MQA-like
with a 576-dim key), which suits matrix units.

**Failure modes.** Requires training from scratch or conversion (MHA2MLA, TransMLA, X-EcoMLA) with fine-tuning; the
latent is already "whitened" and quantizes differently (KVarN's MLA log: their transforms inert on it; radiance's
GLM-5.3 latent cache is bf16-only).

**New vs #416.** Not applicable as a runtime option. Relevant as the reference point for low-rank KV: the 27B's KV
(4 heads × 256 = 1024 dims K + 1024 V per layer) is already GQA-compressed; MLA-style conversion would be a model
change.

**Radiance fit.** Radiance serves MLA for GLM-5.3 (`glm5next`, bf16 latent cache). No path for the 27B without
retraining.
