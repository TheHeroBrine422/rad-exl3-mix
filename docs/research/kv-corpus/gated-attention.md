# Gated Attention — "Non-linearity, Sparsity, and Attention-Sink-Free" (2025) · arXiv:2505.06708

**Why in a KV corpus.** Qwen3.8-27B's 16 attention layers use this head-specific sigmoid output gate
(`attn_output_gate: true` in its config), so the 27B's KV statistics differ from the Llama-era models most KV papers
use: no attention sink, fewer massive activations.

**Method.** Y = SDPA(Q, K, V) ⊙ σ(X W_g) per head and element (gate G₁ after SDPA), trained from scratch (15B MoE and
1.7B dense, 3.5T tokens; 30 variants compared).

**Numbers.** Proportion of attention on the first token, averaged over layers: **baseline 46.7 % → gated 4.8 %**;
layer 21: 83 % → 4 % (Fig. 2). Gate scores are sparse and query-dependent; massive hidden-state activations drop
(Table 4). RULER (Table 5), trained at 32K: baseline 88.89 / 79.50 at 4K / 32K, gated 90.56 / 79.77; with YaRN to
128K: baseline 82.90 / 37.94 / 31.65 at 4K / 32K / 128K vs **gated 88.13 / 72.88 / 58.82**.

**Runtime cost.** One gate matmul per attention layer (part of the model).

**Implications for KV quantization on the 27B (our reading).** (1) Sink-token fp16 exemptions (StreamingLLM, KVQuant,
SKVQ, KVarN, WUSH-KV, FlagRot) may be near-useless — measure the first-token attention share on the 27B first;
(2) without a sink absorbing mass, attention is spread more evenly, so **errors on many mid-weight tokens** matter
more than errors on one huge one; (3) the gate sits between attention and o_proj, so V-side rotations cannot be folded
into W_o (an online un-rotation is needed, as #416 does); (4) YaRN to 512K (radiance supports factor 2) is where gated
models degrade least — and where KV bytes per token matter most.

**New vs #416.** Context, not a method. Flash-Next (#416's model) is a QSA hybrid; whether its attention is gated
too is not stated in #416.

**Radiance fit.** n/a (architecture property).
