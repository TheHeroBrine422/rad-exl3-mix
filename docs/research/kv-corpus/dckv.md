# DCKV (2026) · arXiv:2610.36950 — hybrid-model prompt-cache codec

**Idea.** In hybrid models (Qwen3.5-style: few full-attention layers + linear attention) the KV cache is the only
prompt state that grows with length; store it with a **learned, end-to-end trained transform codec** (transform +
quantizer + conditional entropy models) optimised for code length *and* the change of the model's output.

**Method.** Learned linear (plus small nonlinear correction) transform per layer with a joint K/V basis, scalar
quantization, entropy coding with conditional models (token-embedding-conditioned + checkerboard context from
already-decoded neighbouring positions); rate–distortion training against the frozen LM (λ trades bytes vs output
change). Restores the full bf16 cache before decoding.

**Numbers.** Qwen3.5-2B (bf16 = 12 288 B/token; Table 1), payload B/token and score change (points) on RULER 4K /
LongBench / MMLU / RULER 32K: KVTC 16× 627 B: −0.53 / −0.31 / −0.62 / −0.02; KVTC 32× 304 B: −3.88 / −4.17 / −0.93 /
−6.94; xKV ranks 192/288 ~3.2–3.7 kB: −2.65 / −3.73; **DCKV λ = 3·10⁵: 214 B (57×): −0.56 / −1.75 / −1.10 / −0.45**;
DCKV λ = 10⁵: 113 B (109×): −1.69 / −3.55 / −0.87 / −2.15. 72 % fewer bytes than KVTC at equal RULER-4K score.
Conditional entropy models save 12–16 % of bytes at equal quality. TTFT with stored cache: 115 ms (4K) and 287 ms
(32K) vs 862 / 4 601 ms prefill (A100).

**Runtime cost.** Training per model (codec weights ≈ 5 % of the 2B model); decode needs entropy decoding with
context models (sequential-ish) then bf16 restoration; does not reduce peak memory or TPOT.

**Failure modes.** Trained on 4K prompts, measured to 32K only (beyond is computed); one model, few tasks; rate
control requires one codec per λ.

**New vs #416.** First KV codec designed for **hybrid** models like the 27B's family, at ~50–100× for stored
prompts; complements #412 tiers and prefix-cache reuse rather than the live pool.

**Radiance fit.** Host/disk tier for prefix caches and multi-turn agents (store per prompt chunk; restore to the
pool's layout on admission). The 27B has 16 attention layers × 4 KV heads × 256 × 2 = 64 KiB/token bf16, so 57× ≈
1.1 KiB/token stored. The GDN state (fixed size per sequence) must be stored alongside at each reuse point.
