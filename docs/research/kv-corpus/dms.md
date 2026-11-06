# DMS — Dynamic Memory Sparsification / "Inference-Time Hyper-Scaling" (NVIDIA, 2025) · arXiv:2506.05345

**Idea.** Retrofit a model in ~1K training steps to **learn which tokens to evict**, with *delayed* eviction (a token
marked for eviction stays visible for a sliding window, so its information is absorbed first); spend the saved
memory reads on longer / more parallel reasoning ("hyper-scaling").

**Method.** A per-head eviction decision derived from an existing projection (no new parameters, one neuron of q/k
reused), Gumbel-sigmoid during training, logit distillation from the original model while the target compression
ratio ramps linearly from 1× to the target; at inference, marked tokens are dropped after the window (16–512 tokens).

**Numbers.** Llama 3.2 1B-class comparison (Table 1), 4× CR: GSM8K vanilla 47.0 → H2O 14.7, TOVA 20.2, Quest 39.9,
DMC 3.6, **DMS 42.3**; **NIAH 96.4 → H2O 13.4, TOVA 28.0, Quest 95.8, DMS 96.8**. Qwen3-8B DMS 8× (window 512,
Table 2): AIME24 73.0 vs 75.0; MATH-500 95.5 vs 95.1; GPQA 57.6 vs 58.8. Hyper-scaling: Qwen-R1 32B +12.0 AIME24,
+8.6 GPQA, +9.7 LiveCodeBench at equal memory reads. Retrofit cost per CR step (Table 3): 10 GPU-h (1B) to 100 GPU-h
(8B); kvtc reports DMS 8× on Qwen-R1 1.5B AIME24 23.3 vs vanilla 26.2 (arXiv:2511.01815).

**Runtime cost.** Training (hundreds of GPU-hours for 8B-class at 8×); inference: variable per-head token sets
(paged sparse layouts), eviction bookkeeping; decode reads shrink 8×.

**Failure modes.** Needs model training (changes weights' behaviour slightly); per-head ragged caches; long-context
generalisation beyond the retrofit length shown for NIAH/VT only on small models.

**New vs #416.** Learned eviction with delayed merging — the only eviction family whose NIAH holds at 4× in the
evidence collected here. Requires training, i.e. outside an autoresearch loop that only changes runtime options.

**Radiance fit.** Hard: per-head variable lengths conflict with page-granular, content-hashed, GQA-shared blocks;
a page-granular variant (evict whole 16-token pages per KV head) would need its own retrofit.
