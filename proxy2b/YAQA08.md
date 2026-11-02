# YAQA dose-response on Qwen3.5-0.8B (pass rule written 2026-11-02, before any output-side Hessian was collected)

Purpose (user, 2026-11-02): settle whether YAQA's two-sided Hessian helps on this architecture, at paper-like data
scale, on a model whose output-side sums fit our card. No 27B run either way; a positive result may get a 2B
confirmation on an R9700 slot after V3.

**Data.** xcorp v1 (96M tokens, the challenge's upstream sources, apart from calib / public): 2048-token windows in a
fixed shuffled order. Snapshots of H_out after 1,024 / 4,096 / 16,384 sequences (2M / 8M / 34M tokens). The paper's
Sketch-B setting is ~64K sequences; 16K is a quarter of it. Collector: `proxy2b/yaqa08.py` (sampled labels -- the
true Fisher --, w_t = ||x_t||^2, fp32 activations and gradients).

**Arms**, each a whole-trunk uniform quant of the 0.8B (ourq, the trunk ladder's settings, i8 codebook, scaled
Hessian), at K2 and K3:
- one-sided (input Hessian only, `/data/h08/calib`): the baseline;
- two-sided with H_out at 1K, 4K and 16K sequences (`--hout`).

Scored by `evalq.py` on a 0.8B golden of search-pool documents (~300k tokens, 2k cap), paired with `pcmp.py`
(per-document bootstrap).

**Pass (YAQA works here):** at K2 or K3, two-sided beats one-sided by more than 2 paired SE at some snapshot, AND
the gain at 16K is at least the gain at 1K (the trend does not reverse with more data).

**Fail (dead for us):** no snapshot beats one-sided by 2 SE at either K, or the gain shrinks with more data.
Recorded with the caveat that the 0.8B is width 1024 (the 27B 5120).

**Inconclusive:** a gain only at 16K that still rises from 4K -> 16K: then the 64K point is the next step (about
4x the 16K run's time), decided with the user.

**Amendment (2026-11-02 10:40 UTC, before any collected data was used):** the 2048-token gate ran out of the 4070's
memory (model + 5.7 GB of fp32 sums). Windows are 1024 tokens instead, and the run goes to 32,768 sequences, with
snapshots at 1K / 4K / 16K / 32K sequences (1M / 4M / 17M / 34M tokens: the same top end as planned). The pass rule
applies over these four snapshots unchanged ("16K" in it now reads "the last snapshot").
