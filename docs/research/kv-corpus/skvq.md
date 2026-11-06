# SKVQ (2024) · arXiv:2405.06219

**Idea.** Keep a sliding window of recent tokens (and the sink) at high precision; quantize the rest at ~2 bits after
a calibrated **channel reorder** that groups similar-range channels, with a calibrated per-group clip.

**Method.** Offline (256 × 4096 wikitext2 samples, "a few minutes"): permute channels so a quantization group holds
channels of similar magnitude (permutation folded into the projection weights), fit a clip ratio α per group.
Online: per-token asymmetric group quantization (group 64–128), last 128 tokens and attention sinks in fp16.
Effective 2.25 bits K (g 64), 1.5-bit V configs.

**Numbers.** Ablation, Llama-2-13B LongBench avg (Table 3): FP16 48.66; RTN 35.55; + window-128 **45.73 (+10.18)**;
+ group clip 46.44; + channel reorder 47.99; + sink 48.14; window kept in FP8 E4M3 48.04. Llama-2-13B Wikitext PPL
(Table 2): 2-bit SKVQ 4.87 at 2.25 bits vs KVQuant 4.92 at 2.32–2.35 vs RTN-sym 26.83. LongBench Llama-2-7B-chat
(Table 1): SKVQ 37.50 vs KIVI 35.91 vs FP16 38.50. NIAH on Llama2-7b-80k, 1K–32K: SKVQ above KIVI at lower bits (Fig. 5).

**Runtime cost.** Offline calibration (reorder + clip). Online: window buffer in fp16 (or fp8), per-token group
quant. "Up to 7× faster decoding" and 1M tokens on 80 GB are **LLM-Viewer roofline estimates** (Sec. 9), not measured.

**Failure modes.** No RULER / multi-key test; window size matters more than any other component (most of the gain
is "recent tokens exact"), so errors on far-back retrieval are not probed.

**New vs #416.** The **recent window at high precision** and **sink at high precision** (neither exists in #416, whose
layouts are uniform per layer); a calibrated channel permutation (a cheaper alternative to rotation; Hadamard already
mixes channels, so likely redundant after rotation).

**Radiance fit.** Yes for the window: a per-layer "tail" group (last N tokens in bf16/fp8, re-stored packed when a
16-token page leaves the window) — it is the same mechanism #412 uses for sink + tail. Costs a transcode on page
retirement and complicates prefix-cache hashing (a page's bytes change once). Channel reorder: free if folded into
k/v projections (needs per-KV-head permutation consistent with GQA and RoPE pairs).
