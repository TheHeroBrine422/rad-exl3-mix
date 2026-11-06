# Palu (2024) · arXiv:2407.21118

**Idea.** Compress the **hidden dimension** of KV post-training: SVD-factor W_k, W_v into low-rank down/up
projections, cache the low-rank intermediate, reconstruct K/V on the fly (fused with RoPE in a Triton kernel);
quantization-friendly via Walsh–Hadamard on the latent.

**Method.** Group-head low-rank decomposition (G-LRD: a latent shared by a group of heads; between per-head M-LRD and
joint J-LRD), Fisher-information-based automatic rank search per layer, WHT to remove latent outliers before
quantization; V's up-projection fused into W_o; K reconstruction + RoPE + q·k fused.

**Numbers.** Llama-2-7B 50 % rank (Table 1): Wiki2 PPL 5.47 baseline → J-LRD 5.62, G-LRD 6.01, M-LRD 6.75. With
quantization, Llama-2-7B Wiki2 at 4K (Table 2): KVQuant 3 b 5.35 (12.0 GB at 128K); **Palu-30 % + 3 b 5.33 (8.4 GB)**;
Palu-50 % + 3 b 5.77 (6.0 GB); Palu-30 % + 2 b 5.76 (5.6 GB). LongBench Mistral-7B-v0.2 (Table 3): baseline avg 42.54,
synthetic 44.87; Palu-30 % 41.55 (synthetic 37.92); **Palu-50 % 36.23 (synthetic 16.88)**; KIVI-2 41.45 (synthetic
40.28). Speedup up to 1.89× (RoPE attention) / 2.2× (non-RoPE) at 50 %, 2.91× with quantization.

**Runtime cost.** Offline SVD + rank search; online K reconstruction per decode step over the whole context (fused,
but FLOPs grow with context × rank × head_dim).

**Failure modes.** **Retrieval collapses first**: LongBench synthetic tasks lose 62 % at 50 % rank while other
categories lose ~10 %; low rank removes exactly the fine detail needle tasks need. Code 000: V of Flash-Next needs
129–183 of 256 components for 90 % variance; dropping 32 of 192 non-rotary key dims costs more than all of 4-bit
quantization (IKV Table 9).

**New vs #416.** Low-rank hidden-dim compression (rejected by Code 000's measurements).

**Radiance fit.** Requires changing k/v projections (container change) and a reconstruct-in-kernel path; poor value
for the 27B given the retrieval failure and Code 000's spectra.
