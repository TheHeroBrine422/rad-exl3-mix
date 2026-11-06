# MHA2MLA (2025) · arXiv:2502.14837

**Idea.** Convert a trained MHA/GQA model to MLA with little data: drop RoPE from the query/key dimensions that
contribute least (keep a *partial RoPE*), then joint-SVD the NoPE key part and the values into a shared latent;
fine-tune on 0.6–1 % of pre-training tokens.

**Method.** Partial-RoPE selection by 2-norm contribution per frequency pair (S_2-norm) or high-frequency keep;
SVD_joint of [W_k,nope; W_v] to rank d_kv; brief fine-tuning; compatible with latent quantization (HQQ / Quanto int4).

**Numbers.** Llama2-7B LongBench avg (Table 4): BF16 27.4; Int4 HQQ 27.5 (−75 % memory); **Int2 HQQ 21.2, Quanto
18.5 (−87.5 %)**; MHA2MLA d_kv = 64 BF16 26.7 (−68.75 %), + Int4 26.4 (−92.19 %); d_kv = 16 BF16 **25.1 (−87.5 %)**,
+ Int4 25.0 (−96.87 %). At equal memory (−87.5 %), low-rank + fine-tune beats 2-bit quantization by ~4–7 points.
Commonsense (Table 1): 135M SmolLM d_kv = 32 −1.21 avg, Llama2-7B d_kv = 32 −0.30. Partial-RoPE ablation: keeping
low-frequency dims (S_low) costs −5.25 vs −0.82 for high-frequency (135M).

**Runtime cost.** Fine-tuning (0.6 % of pre-training tokens; for a 27B model trained on tens of T tokens that is
tens of billions of tokens — out of reach here); MLA inference kernels.

**Failure modes.** Needs training; evaluated on LongBench averages (no RULER), small to 13B models.

**New vs #416.** Low-rank latent KV with *learned recovery* (not a runtime option). Interesting fact for the 27B: it
already uses partial RoPE (64 of 256 dims), i.e. MHA2MLA's first step is done; the 192 NoPE dims are the low-rank
candidates — but Code 000's spectra (above) suggest little low-rank slack in radiance-era models.

**Radiance fit.** Model conversion + MLA kernels; out of scope for a runtime-option autoresearch loop.
