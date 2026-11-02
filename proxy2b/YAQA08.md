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

## Amendment 4 (2026-11-02 ~10:35 CST, written before any data): follow-ups #33-11

Same golden, snapshot 8K, i8. Arms:
- `k2hout`, `k2hdiag`: as `k3hout` / `k3hdiag` at K2 (paper damping 0.01, no scaled Hessian), vs `k2ldlq`, `k2base`;
- `k3bhout`, `k2bhout`: `base`'s settings (scaled Hessian, damping 0.025 / input groups 0.005) plus the full H_out
  with output damping 0.01 — the arm that would ship — vs `base` at the same K.
Rules (paired, z = diff / se):
- **H_out holds at K2** if `k2hout` beats `k2ldlq` with z ≤ −2.
- **Stacks with our settings** if `bhout` beats `base` with z ≤ −2 at that K.
- **The user's switch bar on the 0.8B** ("beat our tuned base"): `bhout` beats `base` with z ≤ −2 at **both** K3 and
  K2. Even then, nothing changes for V3 until a 27B check; this only decides whether to plan one.

## Amendment 5 (2026-11-02 ~10:45 CST, written before any data): 27B sizing, #33-12

All on `base`'s settings + H_out at output damping 0.01 (as `bhout`), i8, g08-search1. Arms:
- `bmix` (K3, K2): full H_out everywhere except mlp.gate_proj / up_proj, which get only diag(H_out); 8K snapshot;
- `bdiag` (K3, K2): diag(H_out) for every linear; 8K snapshot;
- `bhoutN` (K3): full H_out from the N = 1024 / 2048 / 4096 snapshot (dose).
Rules (gain = paired diff vs `base` at that K):
- An approximation **keeps the gain** if its gain is ≥ 2/3 of `bhout`'s at both K and it is not worse than `bhout`
  by z ≥ 3 at either K. The cheapest arm that keeps the gain sets the 27B collection: `bdiag` → diagonal only
  (MB, one pass); else `bmix` → full H_out except gate/up (~25 GB, see the 27B plan); else full (~160 GB).
- **Sequences needed:** the smallest N whose K3 gain is ≥ 80% of the 8K gain.

## Amendment 6 (2026-11-02 ~11:50 CST, written before any data): diagonal dose, #33-13 (0.8B part)

On `base`'s settings + diagonal H_out (output damping 0.01), i8, g08-search1:
- `k3bdv8192`: the 8K diagonal written as a `hout_diag` vector file (ourq.py's new path) — **must equal `k3bdiag`**
  (same KL to 1e-6); otherwise the new path is wrong and nothing else here counts.
- `k3bdiagN`, `k2bdiagN` for N = 1024, 2048 (diagonals of those snapshots).
**Sequences for the 27B diagonal collection:** the smallest N whose gain vs `base` is ≥ 80% of the 8K diagonal's at
both K (8K gains: K3 −0.00790, K2 −0.04263). If 1K qualifies, the 27B run is planned at 1K (~8.5 h fp32, one night,
both cards); else 2K (~17 h); else 4K.

## Amendment 7 (2026-11-02 ~12:20 CST, written before any data): split-precision check, #33-14

`proxy2b/prectest.py` on the 0.8B: 16 xcorp windows, the same sampled targets, trunk linear GEMMs in fp32 / x2 / bf16
(everything else fp32). **x2 is usable for the 27B collector** if, for the summed diagonal over the 16 sequences,
the median relative L2 error vs fp32 is ≤ 1e-3 and the worst linear's ≤ 1e-2 in every depth band (the diagonal's
own sampling noise is ~1e-2, #33-09), and bf16 shows a clearly larger error (the test can see a difference).
Otherwise the 27B run stays fp32.

## Amendment 8 (2026-11-02 ~13:05 CST, written before any data): split precision judged by KL, #33-15

The user asked for the KL test of x2 (#33-14 failed its matrix-error bound). `prectest.py --save` collects the
diagonal over the first 1,024 windows of yaqaB's order with yaqaB's targets (sampled from fp32 logits, seed s per
sequence), linear GEMMs in x2, and as a control in bf16. The fp32 reference is the existing 1K snapshot's diagonal
(`k3bdiag1024`, `k2bdiag1024`, same windows and targets). Quants on `base`'s settings + diagonal H_out (output damping
0.01), i8, g08-search1, paired against the fp32 arm at the same K.
- **x2 passes** if, at both K3 and K2: KL(x2) / KL(fp32) ≤ 1.005 and z < 2 (no detectable loss, and at most 0.5%).
  Then the 27B collector runs with YAQA_PREC=x2 (its gate also reports one linear against fp32 on the 27B).
- bf16 is the control: if bf16 also passes, the KL test cannot see collector precision at 1K sequences, and x2's pass
  is recorded with that caveat (the 27B gate's per-linear check then decides).

## Amendment 9 (2026-11-02 ~13:30 CST, written before any data): replicates and the identity H_out, #33-16

#33-15 showed quant-to-quant spread of ~2–3% that the paired per-document SE does not cover. Arms (i8, g08-search1):
- replicates: `base` and `bdiag` (base settings + 8K diagonal H_out, output damping 0.01) at K3 and K2 with rotation
  seed offsets 1 and 2 (`ourq.py --seed-offset`), next to the existing offset-0 arms: 3 seeds per arm;
- `bident` at K3 and K2: base settings + H_out = I (output damping 0.01). Two-sided LDLQ with an identity output factor
  still weights rows by svh² in the rotated basis (`prepare_H_out`), which one-sided LDLQ does not.
Rules, per K:
- **The H_out gain is real** if every `bdiag` seed beats every `base` seed, and the mean gain ≥ 3× the between-seed SD
  of the per-arm mean KL. Report the seed SD as the 0.8B quant noise for later comparisons.
- **The svh weighting explains part of it** if `bident` gets ≥ 1/3 of the mean bdiag gain vs mean base; if ≥ 2/3, the
  gain is mostly the weighting and the 27B diagonal run is not needed for it (the weighting needs no backward pass).
- Best-of-N: report the best/worst seed per arm; any pick of a seed is confirmed on another slice before use.

## Amendment 10 (2026-11-02 ~14:25 CST, before the 27B run): stopping rule for the 27B diagonal collection, #33-17

Reference (0.8B, `diaghalves.py`): split halves of 1,024 sequences each (where 1K sufficed, #33-13/16) agree at
median cos 0.995, p5 0.964, min 0.916 over the 150 linears. The 27B collector writes `halves.json` (even vs odd
sequences) at every snapshot. **Stop at the first snapshot ≥ 1,024 sequences whose median ≥ 0.995 and p5 ≥ 0.964**
(halves of n/2 each, i.e. stricter than the 0.8B reference at equal n); otherwise continue to 2,048, then 4,096.

## Amendment 11 (2026-11-02 ~15:10 CST, written before any data): estimators and dose below 1K, #33-18

Critic items A2 / A5 (`docs/ideas/ranked.md`). One fp32 collection with the 27B collector (`collect.py` YAQA_DIAG +
YAQA_DIAG_EXTRA, one GPU) on the 0.8B, calib qad-pool 2048-token windows (as the 27B run), snapshots 128 / 256 / 512 /
1024. Estimators at 1024: Sketch B diag (`B`), Σ_t‖x_t‖²dy_t² (`xw`, the weighting of the failed 27B gates
#33-02…05), Σ_t dy_t² (`dy`). Quants on base settings + the diagonal (output damping 0.01), i8, g08-search1, K3 and
K2, rotation seed offsets 0 / 1 / 2, against the existing base seeds. Gain = mean(base) − mean(arm) over 3 seeds.
- **A2 (estimator):** `xw` keeps the gain if its gain ≥ 2/3 of `B`'s at both K and every xw seed beats every base
  seed. If it does, the old 27B negatives were not the estimator → a short 27B pilot swap before the full night.
  `dy` is "as good" if its mean is within one per-seed SD (#33-16: ~1% K3, ~2% K2) of `B`'s.
- **A5 (dose):** the smallest N ∈ {128, 256, 512} whose gain is ≥ 80% of the 1024 gain at both K, with every seed
  beating every base seed. If one qualifies, the 27B stopping rule (amendment 10) is amended to allow an early
  K3 swap test from that snapshot before the collection continues.

## Amendment 12 (2026-11-02 ~15:50 CST, written before any data): H_out on our own agentic traces, #33-19

Corpus: `pisess v1` (our pi sessions rendered as Qwen3.8 agentic traces, redacted; 18.5M tokens; training /
statistics only). Collection as #33-18 (`collect.py` YAQA_DIAG, fp32, one GPU, 2048-token windows, 1,024 windows),
on pisess windows instead of calib qad windows; same seed and settings. Quants: base settings + diagonal (output
damping 0.01), i8, g08-search1, K3 and K2, seed offsets 0 / 1 / 2. Reference: #33-18's `eB` arms (qad windows,
1,024). Also the diagonal of the 50/50 mix (mean of the two diagonals).
- **Different but not better:** report the per-linear cosine between the pisess and qad diagonals.
- **pisess helps** if its 3-seed mean KL ≤ 0.98× the qad arm's at both K and every pisess seed beats every qad seed
  at one K at least; **the mix helps** under the same rule. If neither, the 27B collection stays on qad windows.
- Caveat recorded: g08-search1 has few long agentic documents (capped at 2k); a gain specific to agentic text may not
  show on it. If pisess is not worse (≤ 1.01×) and the cosine is < 0.95, re-check on a slice of long agentic
  search-pool documents before deciding.
