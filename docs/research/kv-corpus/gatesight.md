# GateSight (2026) · arXiv:2610.14754 — **Qwen3.8-27B and Qwen3.8-Flash-Next**

**Idea.** In GDN–attention hybrids the **recurrent layers already know what the attention cache will need**: the GDN
write gate, decay and write magnitude emitted during prefill predict a KV entry's future utility before any query
is seen. Learn that map with small per-head MLPs, score the cache once after prefill, keep the top entries per layer,
and size each layer's budget by the **native attention output gate**.

**Method.** Utility label U = change of the attention output over the model's own continuations when an entry is
removed (rollouts). Observations on Qwen3.8-27B: which GDN heads keep long memories is fixed by the weights, not the
document; untrained signals predict little (best single head Spearman ρ +0.232), a ridge over 867 GDN features +0.548
(Table 1), MLPs recover more of the high-utility tail. **Per-layer budget ∝ mean output-gate magnitude, which spans
0.022 to 0.459 across the 27B's attention layers** (gate maxed over the 6 query heads of a KV head). vLLM plugin, no
kernel changes: each layer compacts kept entries to the front of its pages, freed blocks return to the pool.

**Numbers.** Kept fraction p = 0.7 / 0.4 / 0.1 (Table 3), **Qwen3.8-27B**: SCBench GateSight 63.3 / 51.7 / 37.4 vs
KVzip 64.5 / 60.4 / 27.9; RULER-4096 97.4 / 91.9 / 76.6 vs KVzip 97.7 / 95.5 / 29.4. Allocation ablation on 27B
(Table 4, mean over 9 fractions): SCBench gate-proportional 60.6, flat 53.7, PyramidKV schedule 44.3; RULER 99.8 /
99.4 / 90.4 (PyramidKV 52.1 at p = 0.1). Scoring time on 27B (Table 2): 0.246 s at 256K tokens vs KVzip 100.3 s,
SnapKV 0.452 s. Evaluated up to the models' 262K limit.

**Runtime cost.** Estimator MLPs trained once (2 428 Nemotron documents, 500 positions each); scoring ≈ 0.1 % of
prefill time at 256K; physical compaction after prefill.

**Failure modes.** Eviction is still irreversible: SCBench multi-query at 10 % keeps only ~37 points; KVzip wins at
intermediate fractions; RULER at 4K only (long-context results via SCBench).

**New vs #416.** (1) **A free per-layer sensitivity prior for the 27B: the output-gate mean** — candidate for the
knapsack's initial ordering of layers (bits instead of tokens); (2) GDN-derived utility as a query-agnostic *page
precision/demotion* score — Code 000 found attention-count importance no better than a position hash on QSA;
GDN signals were not tried.

**Radiance fit.** Precision tiers per page (score once after prefill → demote low-utility pages to a narrower layout)
fit #412-style tiering; deletion conflicts with prefix hashing. Gate-mean layer prior: trivial (offline number per
layer).
