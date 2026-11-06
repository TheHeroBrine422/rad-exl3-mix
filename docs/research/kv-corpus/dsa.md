# DSA — DeepSeek Sparse Attention (DeepSeek-V3.2, 2025) · arXiv:2512.02556

**Idea.** A cheap **lightning indexer** scores every previous token for each query (few heads, FP8, ReLU), and
attention reads only the **top-k = 2048** tokens. Retrofitted onto a dense MLA model by continued training.

**Method.** I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s) (Eq. 1); top-k tokens per query (shared across the
MLA heads in MQA mode). Training: (1) **dense warm-up — model frozen, only the indexer trained to match the main
attention distribution (KL), 1000 steps × 16 × 128K tokens = 2.1B tokens**; (2) sparse stage — model + indexer,
15 000 steps × 480 × 128K = 943.7B tokens, indexer aligned on the selected set only.

**Numbers.** Parity with V3.1-Terminus on standard benchmarks; independent long-context evaluations: AA-LCR +4 points
(reasoning mode) and Fiction.liveBench ahead of the dense predecessor. Core attention O(L²) → O(Lk); per-token serving
cost vs position measured on H800 (Fig. 3) — flat for decode where the dense model grows linearly.

**Runtime cost.** Indexer keys cached per token (FP8, small head dim) and an O(L) indexer pass per query (cheap but
not free; IndexCache, arXiv:2603.12201, reuses indices across layers to cut it).

**Failure modes.** Retrofit cost: the 2.1B-token indexer warm-up is affordable, the ~1T-token sparse stage is not
(for us). Selection reuse across layers without training hurts (Code 000: ΔNLL +0.014–0.026 on Flash-Next).

**New vs #416.** Recipe for a **retrofit indexer** for a dense model (the 27B) via KL distillation with the model
frozen — the same shape as D-LSA's distilled indexer (d-lsa.md).

**Radiance fit.** Radiance already runs QSA (indexer + block keys + select + paged attention) for Flash-Next; a
distilled indexer for the 27B would reuse that path, with #412 residency on top.
