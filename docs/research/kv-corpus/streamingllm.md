# StreamingLLM — attention sinks (2023) · arXiv:2309.17453 (foundational)

**Idea.** Window attention fails once the first tokens are evicted, because models dump surplus attention on the
initial tokens ("attention sinks"); keep **4 sink tokens + a rolling window** and the model streams indefinitely.

**Method.** Cache = first x tokens + last y tokens; positions assigned within the cache. Optional: pre-train with a
dedicated learnable sink token.

**Numbers.** Llama-2-13B PG19 first book (65K): window 0+1024 PPL **5158.07**, 4+1020 **5.40**. Llama-2-7B, 400K
tokens (Table 2): 0+4096 3359.95; 1+4095 11.88; 4+4092 9.59; 8+4088 9.54. Streams up to 4M tokens; 22.2× faster than
sliding-window recomputation.

**Runtime cost.** None (a fixed-size cache).

**Failure modes.** Everything outside sink + window is gone: no long-range retrieval (PolarQuant's LongBench table:
StreamingLLM 38.36 vs 48.63 exact, arXiv:2502.02617 Table 1).

**Why it matters for quantization.** Sinks carry huge attention mass, so their quantization error is amplified: kvtc
at 64× collapses without exempting the first 4 tokens (lost-in-the-middle 0.0, arXiv:2511.01815 Table 6); AQUA-KV
keeps a 16-bit sink (PPL 7.03 vs 7.15 without, arXiv:2501.19392); KVQuant, SKVQ, KVarN, WUSH-KV, FlagRot all keep sinks
in 16 bit. **But the 27B uses gated attention**, which largely removes sinks (first-token attention 46.7 % → 4.8 %,
arXiv:2505.06708, see gated-attention.md) — the sink exemption may buy little there; measure before building.

**New vs #416.** #416 has no sink exemption (radiance's #412 keeps 512 sink tokens *resident*, not at higher
precision).

**Radiance fit.** Sink exemption = one extra bf16 KV group for the first page(s) of each sequence; cheap.
