# KIVI (ICML 2024) · arXiv:2402.02750

**Idea.** Keys have fixed outlier *channels*, values do not: quantize K per-channel and V per-token, 2-bit, tuning-free.

**Method.** Asymmetric RTN (scale + zero) in groups of G = 32: K grouped along tokens per channel, V along channels
per token. Because a per-channel group needs G tokens, the newest R tokens (residual, R = 32–128) stay fp16 and are
packed once a full group exists; attention is a tiled matmul over the packed and residual parts. No calibration.
Effective rate with fp16 scale/zero per 32 values: ~3.0–3.2 bits/value (KVQuant counts KIVI-2-gs32-r128 as 3.05–3.17,
arXiv:2401.18079 Tables 3–4).

**Numbers.** Llama-2-13B, simulated 2-bit g32 (Table 1): K-per-channel/V-per-token CoQA 63.53 vs 16-bit 66.37; the
other three axis choices collapse (2.80–52.93). KIVI-2 on GSM8K (Table 3): Llama-2-13B 20.77 vs 22.67; Mistral-7B
36.01 vs 38.36; Falcon-7B 3.41 vs 4.55. LongBench avg (Table 4): Llama2-7B 44.27 vs 44.52, Mistral-7B 45.85 vs 46.58.
Llama-3-8B-Instruct LongBench avg 44.37 (KIVI-2) / 45.31 (KIVI-4) vs 45.21. Group-size ablation: G = 128 drops
Llama-2-13B GSM8K to 17.29 (vs 20.77 at 32). Peak memory 2.6× lower, batch 4×, throughput 2.35–3.47× (Llama-2-7B).

**Long context / tails.** Paper's NIAH figure (Fig. 4) claims retrieval kept. Independent RULER at 32K on
LLaMA-2-7B-32K (KVQuant, arXiv:2401.18079 RULER table): KIVI-2 avg **39.78 vs fp16 56.40**; NIAH-1 76 vs 100, MKey2
11.4 vs 68.2, MKey3 0 vs 11. Passkey 2K–32K 0.68–0.76 retrieval vs 1.0 (KVQuant). Mean metrics hide a large
retrieval failure at 2 bits.

**Runtime cost.** No extra passes or calibration; needs an fp16 residual buffer per sequence and per-channel packing
when a group completes; custom CUDA (Triton) kernels for mixed fp16 + packed attention.

**Failure modes.** Multi-key retrieval at 2 bits (above); quality falls with larger groups; GSM8K drops 1–2 points on
13B / 7B. On a post-norm, post-RoPE K (radiance) the per-channel outlier premise is weaker; Code 000 measured KIVI-2
granularity at proxy 0.201 at 3 bpw vs 0.104 for k4h_v2h+vmean at 3.5, and KIVI-4 at 5 bpw 0.036 vs k4h_v4+vm 0.039 at
4.5 (IKV Table 9: rejected; per-channel key groups span eight QSA blocks).

**New vs #416.** Per-channel K grouping across tokens is the one axis #416 does not have; measured negative on
Flash-Next.

**Radiance fit.** Partial: per-token V fits the existing rows; per-channel K needs 32-token tiles (two 16-token
pages) plus an fp16 tail group (the KVarN integration problem: prefix hashing, spec rollback, tile completion).
