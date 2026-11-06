# DistillCache (2026) · arXiv:2608.08878

**Idea.** Learn the eviction policy with RL (REINFORCE) using a **per-step KL-divergence reward** to the full-cache
output distribution, from cheap internal signals (attention statistics, value norms, entropy, position).

**Method.** Lightweight policy network per decision; state = per-token features; action = keep/evict (Gumbel-top-k);
reward = −KL(p_full ‖ p_compressed) at each step. Trained on Mistral-7B-Instruct-v0.3, applied zero-shot to others.

**Numbers.** Mistral-7B LongBench (Table 1), budgets 50 / 25 / 12.5 % of full 41.5: H2O 39.2 / 36.4 / 29.8; SnapKV
39.6 / 37.0 / 31.0; ForesightKV 40.3 / 38.2 / 34.4; **DistillCache 40.5 / 39.1 / 36.6**. Context length at 25 %
(Table 2): 2K 40.5, 4K 40.2, 8K 38.1 (H2O 34.1 at 8K). Throughput (2× RTX 4090, 25 %, Table 3): full 18.0 tok/s,
H2O 42.8, DistillCache 38.0 (feature 0.9 ms + policy 0.5 ms per step). Zero-shot to Llama-3-8B (Table 4): LongBench
at 25 % competitive with retrained RL methods.

**Runtime cost.** Per-step feature extraction + policy inference (~1.4 ms here), RL training offline.

**Failure modes.** Evaluated only to 8K context; 7–8B models; still eviction (no recovery); quality falls with
context (40.5 → 38.1 from 2K to 8K at 25 %).

**New vs #416.** Using **KL to the full-cache model as the training/selection signal** is the same objective as
radiance's KL mode; cited by Code 000 among methods considered. Nothing for the per-layer quantization recipe itself.

**Radiance fit.** Eviction: poor (see h2o.md). The per-step KL reward idea transfers to the autoresearch judge
(KL-on-device as fitness), which #416's rad-autoquant already does.
