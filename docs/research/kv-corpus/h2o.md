# H₂O — Heavy-Hitter Oracle (2023) · arXiv:2306.14048 (foundational)

**Idea.** A few tokens ("heavy hitters", correlated with frequent co-occurrence) receive most attention mass; keep
those plus a recent window, evict the rest (greedy policy on accumulated attention, framed as dynamic submodular
maximisation).

**Method.** Per layer and head: keep k heavy hitters by accumulated attention score + k recent tokens; evict one
token per decode step. Needs explicit attention probabilities.

**Numbers.** OPT-30B, 20 % KV budget, few-shot (Table 1): PiQA 79.22 vs full 80.09, Winogrande 71.67 vs 71.51, local
window only 57.94 / 51.30. Throughput up to 29× vs DeepSpeed Zero-Inference / HF Accelerate and 3× vs FlexGen
(OPT-6.7B / 30B, offloading settings); latency −1.9× at equal batch.
**Independent failure evidence:** line retrieval at 50 % budget 43.2 % (MiKV, arXiv:2402.18096 Table 1); Mistral-7B
GSM8k CoT at 40 % budget **1.67 vs 41.62** (ZipCache, arXiv:2405.14256 Table 3); Llama 3.1 8B at 8×: lost-in-the-
middle **20.2 vs 99.4**, RULER variable tracking 50.4 vs 99.8 (kvtc, arXiv:2511.01815 Table 2).

**Runtime cost.** Accumulated attention per token (not FlashAttention-compatible without a side output), per-head
variable token sets (ragged layouts).

**Failure modes.** Irreversible loss: tokens evicted early are needed later (retrieval, multi-turn, reasoning chains).
Mean metrics on short tasks look fine; long-context retrieval tails collapse.

**New vs #416.** Eviction was rejected by Code 000 ("unbounded error, conflicts with positions and the prefix cache",
IKV Table 9). Lesson kept: heavy hitters + recency is the structure every later selector exploits.

**Radiance fit.** Poor as eviction (16-token content-hashed pages, prefix sharing, spec rollback). As a *demotion*
signal (precision tiers) only — see MiKV.
