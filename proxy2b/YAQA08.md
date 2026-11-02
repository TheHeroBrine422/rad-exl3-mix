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

**Amendment 2 (2026-11-02 ~10:55 UTC, before any Sketch B data):** the first collector (yaqa08.py) computed neither of
the paper's sketches (it summed ||x_t||^2 dy_t dy_t^T over tokens -- half a step of Sketch A's power iteration from
H_I = I -- and kept the calib input Hessian). Stopped at 525 sequences, nothing used. Replaced by **Sketch B exactly as
the reference code** (Cornell-RelaxML/yaqa-quantization, custom_linear_B.py, power_iters 1): per sequence the weight
gradient G_b, H_in += G_b^T G_b and H_out += G_b G_b^T per linear, sampled targets, 2048-token context
(`yaqaB.py`, gated: H_out / H_in traces equal, resume bit-exact). Quantizer as the reference's: both Hessians
normalized by their mean diagonal, damping 0.01 each side, no scaled Hessian (ourq: per-linear --hin, --hout;
exllamav3's ldlq_2hess checked: L_out = 0 is bit-identical to one-sided LDLQ). Snapshots 1K / 2K / 4K / 8K / 16K
sequences (the paper's Table 9 range; their main setting is 64K). Arms at K2, K3: `base` (our tuned one-sided),
`ldlq` (the paper's one-sided baseline: damping 0.01, no scaled Hessian), `ybN`. The pass rule above applies with
`ldlq` as one-sided (the paper's comparison); yb vs `base` is reported beside it (does it beat our best).

## Amendment 3 (2026-11-02 ~10:45 CST, written before any data): factorial #33-10

The YAQA arms changed H_in (calib x xᵀ → Sketch B's gradient-weighted H_in from xcorp) and added H_out at once
(#33-09). Three K3 arms, i8, damping 0.01 both sides, no scaled Hessian, 8K snapshot, golden g08-search1:
- `k3hin`:  Sketch B H_in, no H_out (one-sided) — the cost of the H_in swap alone;
- `k3hout`: calib H_in + Sketch B H_out (full) — H_out with the correct H_in;
- `k3hdiag`: calib H_in + diag(H_out) as the output factor (off-diagonal zeroed) — the precise part of H_out only.
Reference `k3ldlq` (calib H_in, one-sided, same damping). Paired per-document comparison (pcmp.py), z = diff / se.
- **Attribution:** an arm worse than ldlq with z ≥ 2 carries a cost; if `k3hin` is worse and `k3hout` is not, the
  YAQA loss was the H_in swap.
- **H_out has value** if `k3hout` or `k3hdiag` beats ldlq with z ≤ −2. Then: the same arm against `base` (scaled
  Hessian, tuned damping) and at K2 before anything else. If neither beats ldlq, two-sided rounding is closed on the
  0.8B and H_out stays a measurement / pricing tool only.
