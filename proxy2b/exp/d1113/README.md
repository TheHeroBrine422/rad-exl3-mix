# d1113 · swj-style joint scale refits beyond the MLP: attention q / k / v / gate and GDN in_proj (0.8B, 4070 queue)

Worker w60 (2026-11-06). swj (#85 C10, #85-03; 27B #104) keeps the codes frozen and refits the f16 per-output-channel
scales (svh) of gate_proj and up_proj jointly, so that silu(gate x) · up x matches bf16. Question: does the same
mechanism pay at the other nonlinear junctions where quantized projections meet?
**Rules below written 2026-11-06 ~6:45 PM CST, before any data.**
**Runtime cost: none** for every arm. Only svh changes (one f16 multiplier per output channel, folded into the existing
vector), so the format, kernels and bpw stay the same.

## Which scales reach the logits (Qwen3.5-0.8B; the 27B has the same block design)

Per-output-channel scales c (svh) of the quantized projections. in_proj_b / in_proj_a (GDN β, decay) and the conv,
norm and A / dt parameters are bf16 and are not quantized by ourq (the 0.8B's 150 tensors and the 27B's ladder alike),
so they have no scales to fit.

| tensor | path to the output | what of c reaches |
|---|---|---|
| q_proj, query half (per head 256) | q_norm (RMSNorm over the head dim) → partial RoPE (64 dims) → softmax(qkᵀ) | **all but a uniform scale per head** (absorbed by q_norm); the relative scales inside a head reach the logits (RoPE comes after the norm, so it mixes already-scaled channels) |
| k_proj (2 kv heads × 256) | k_norm → RoPE → logits | same: uniform per-head part absorbed |
| q_proj, gate half (attn_qg's gate) | out = attn · sigmoid(gate) | all (nonlinear, as swj's gate) |
| v_proj (2 kv heads × 256) | linear through softmax · v, then × sigmoid(gate) | all; one v channel feeds 4 q heads (GQA), so it is shared |
| in_proj_qkv (q 2048, k 2048, v 2048) | depthwise causal conv (linear per channel) → **silu** → q, k l2-normalised per head; v into the delta rule | all (c sits before the silu, so it is not a pure scale of q / k / v); for q / k the l2 norm absorbs only the post-silu head norm |
| in_proj_z (2048) | out = RMSNorm_head(core) · w · silu(z) | all (the swj analogue: a nonlinear gate on a fixed other path) |

o_proj / out_proj are the writers: the adopted stack requantizes them afterwards (asym ρ0), so their scales are not
fitted here. An input-side scale (suh) of o_proj could absorb a per-q-head v scale, but the writer requant would
overwrite it.

## Method (`jfit.py`)

Codes frozen. For each layer, log-scales on the chosen tensors' output channels are fitted by Adam to minimise the
block's output error **before its writer** against bf16, given the bf16 model's own block inputs (as swj):
- attention: ‖attn(q̂, k̂, v̂) · sigmoid(ĝ) − bf16‖²;
- GDN: ‖RMSNormGated(core(q̂k̂v̂), ẑ) − bf16‖²;
- MLP (solver check only): ‖silu(ĝ) û − bf16‖² (= swj's objective).

Both sides run in fp32 through the same code (the GDN core is fla's `chunk_gated_delta_rule`), so the target holds no
bf16 rounding. Loss per layer is normalised by the target energy. The fit takes one Adam step per window per layer,
streaming: one pass over the windows, lr 1e-2 cosine-decayed to 5%. Scales are clamped to [0.5, 2] as swj.
- **Data:** qad-pool windows of 2,048 tokens (calib-meta, pools-v3, the moments.py cut, shuffle seed 11), 600 windows
  (~1.2M tokens). The last 64 windows of the order are held out: per-layer relative residual before / after, written to
  `jfit.jsonl`. **No fit touches fit, search or verify.**
- **Solver calibration (before any KL data, 2026-11-06 ~7:20 PM CST).** On d1109's K2 trunk `k2rq` (not an arm here),
  600 qad windows, mean held-out residual ratio fitted / base:
  - lr 3e-3: mA 0.932, gd 0.764, at 0.813;
  - lr 1e-2: mA 0.928, gd 0.755, at 0.810;
  - swiglufit's own held-out ratio on that quant is 0.942 (fit-pool halves).

  lr 1e-2 is taken. The Adam solver matches swj's exact grid on the MLP.
- Variants are fitted in one pass. They are independent, because every variant sees bf16 inputs:

| variant | tensors (scales fitted) | what it tests |
|---|---|---|
| `qk` | q_proj query half, k_proj | the logits junction |
| `vg` | q_proj gate half, v_proj | the gate × v junction (closest to swj) |
| `at` | q_proj (both halves), k_proj, v_proj | the whole attention block jointly |
| `gz` | in_proj_z | the GDN output gate (swj analogue) |
| `gd` | in_proj_qkv, in_proj_z | the whole GDN block jointly |
| `mA` | gate_proj, up_proj (from the trunk codes, replacing swj) | **solver check**: Adam vs swiglufit's exact grid |

## Arms and the stack

The control is the adopted stack `k{K}st`, d1112's recipe (= d1104's `k{K}edyswjasym_r0`):
trunk edy (`q2b.sh 08 K i8`, c18 Σdy² H_out, σ_out 0.01, seed offset s) → swj (swiglufit, g08-fit1) → moments (sib,
through the swj weights, qad, 1M tokens, seed 0) → G, F (λ 1, ρ 0) → writers asym ρ0 (the 48 in `only.txt`).
- **New fits go right after swj**, before the moments and writers. This is the order #85-02 / w11 B2 require: sibling
  refits first, so the writers' F sees the final upstream.
- Full-stack arms: `k{K}fa` (attention pick), `k{K}fg` (GDN pick), `k{K}fc` (both).
- Control reuse: if my trunk + swj reproduce d1112's `k{K}ste{s}_swj` search score file byte for byte, d1112's
  `k{K}st{s}` / `vf-k{K}st{s}` score files are the control (same recipe, same code path). Otherwise st is rerun here.

Goldens:
- **fit** = g08-fit1: the screen only. swj's own fit slice, so the screen's base is slightly in-sample for swj. Both arms
  share swj, so the screen contrast is unaffected.
- **search** = g08-search1: the deciding slice.
- **vf** = g08-verify1: the overfitting check only.

Pairs: `pcmp.py`, per seed. geo = the geometric mean over seeds of the per-seed ratios.

## Stage 1 · screen (fit golden, on edy + swj, no writers)

Seeds 0, 1, 2 at K3 and K2. K4 seeds 0–2 are reported only. Per variant, the ratio is variant / `k{K}e_swj` on g08-fit1.
- **Family pick:**
  - attention: among {qk, vg, at}, the variant with the lowest mean log ratio over K3 + K2, provided its geo is
    ≤ 0.995 at both K3 and K2;
  - GDN: the same over {gz, gd};
  - a family with no qualifying variant is **null at the trunk** and gets no full-stack arm.
- **Solver check** (search, seed 0–2 at K3 / K2): mA vs `k{K}e_swj`.
  - geo within [0.98, 1.02]: the Adam solver matches swj's exact fit;
  - > 1.02: it underfits, and any null in this experiment is reported as "solver-limited", not as "the mechanism does
    not pay".
  - Also reported: the held-out residuals of jfit vs swiglufit's `swiglu.jsonl`.
- If both families are null, stage 2 is skipped (reported), and the queue fills with the screen at seeds 3–4 to check
  the null.

## Stage 2 · decision (full stack, search + verify)

Arms fa / fg / fc (fc only when both families qualify; with one family, fc = that arm) vs st, seeds 0–2, K3 and K2.

| verdict (per arm) | condition |
|---|---|
| **adopt-for-27B** | search geo ≤ 0.98 at K3 **and** K2, every seed < 1 at both, **and** verify keeps ≥ ⅔ of the search gain at each K (vf mean log ratio ≤ ⅔ × search mean log ratio, as #93 / d1112) |
| **real but below the bar** | every seed < 1 at both K, geo in (0.98, 0.995] (or ≤ 0.98 without the verify keep): not worth a 27B pass of its own; fold in only if it rides on another pass (SWJ27's wording) |
| **harmful** | geo ≥ 1.01 with every seed > 1 at either K |
| **null** | anything else |

- The 27B recommendation reads fc first, then fa / fg to say which junction carries it. If fc passes but one family
  alone is null, only the passing family is recommended (less 27B tooling).
- **K4** (seeds 0–2, same arms): reported with the same numbers. Flag: geo ≥ 1.01 at K4 would limit the step to
  K ≤ 3.5.
- **Seeds 3–4** (queue fill, K3 / K2): a robustness check. If they move a geo across a threshold above, the verdict is
  labelled *fragile*, never flipped silently.

## Amendment 1 (2026-11-06 ~7:20 PM CST, before any data of the new variants)

What prompted it: the first screen (K3 seed 0, fit golden) has attention qk 1.0038 (z +3.6), vg 1.0009 (+1.5),
at 1.0034 (+3.5), against held-out residual ratios 0.96 / 0.98 / 0.95. Lowering the plain pre-writer MSE does not lower
KL at the attention junction. GDN: gz 0.9905 (−7.2), gd 0.9921 (−4.7). mA vs swj (search): 0.9996 (−0.7), so the
solver check passes.

The obvious suspect is the metric: plain MSE before the writer weights every output channel alike. Two variants
are added, with the same tensors as `at` / `gd` and the loss taken **after the bf16 writer**, ‖W_o (y − t)‖²
(o_proj / out_proj):
- `aw` joins the attention family;
- `dw` joins the GDN family.

Same screen and pick rule. All seeds and K get them; K3 seed 0 is refit for them only (the six scored variants are
kept). Cost: jfit ~+35% time, two more fit-golden evals per screen.

## Order, time, disk

Queue (`run.sh`):
1. prep;
2. screen K3 / K2 seeds 0–2;
3. screen K4 seeds 0–2;
4. stage 2 K3 / K2 seeds 0–2;
5. stage 2 K4 seeds 0–2;
6. screen + stage 2 at K3 / K2 seeds 3–4.

Every GPU unit holds `/data/yb/gpu.lock` < 15 min.

Estimated (d1112 timings): trunk + swj ~4 min, jfit ~5 min, an eval ~0.6 min, a stage-2 arm (moments 3.2 + G / F +
writers 1.7 + 2 evals) ~7 min. Totals: screen ~13 min per (K, seed), stage 2 ~21–28 min per (K, seed). **~10–11 h**
in all.

Disk: an arm's moments, G / F and wq copies are deleted once its goldens are scored. Kept: shards, logs, score files,
the variants' `scales/` (small, for the 27B). The trunk wq of a (K, seed) is kept until all its arms are scored.
Footprint ≲ 20 GB (limit 40).

## Files and how to run

- `jfit.py`: the fitter (above).
- `run.sh`: the queue. It is idempotent: arms are skipped by their score files, and every output is written `.part`
  then renamed.
  - It runs from the snapshot `/data/yb/d1113/snap/`. Output goes to `/data/yb/d1113/`: `result.txt` and `report.md`.
  - Stop: `touch /data/yb/d1113/STOP` (or `/data/yb/night.stop`).
  - `D1113_STEPS` selects steps.
- `report.py O`: applies the rules → markdown.
- Registry: `/data/queue.d/40-d1113.sh`. Watchdog: `local/watch.conf` (`d1113`, `d1113out`, `d1113fail`).

## Status

- **Start.** 2026-11-06 6:43 PM CST (00:43 UTC), via `resume-queues.sh` (registry `40-d1113`).
- **Paused and relaunched for amendment 1** at 7:16 PM CST. Units interleave under gpu.lock with d1112 (last arms)
  and the lead's d1107.
- **Reproduction.** k3e_swj is byte-identical to d1112's k3ste_swj (search), so d1112's st scores are the control
  (control reuse, as planned).
- **ETA (CST), re-estimated 8 PM.** A screen takes ~22 min per (K, seed). If attention stays null, stage 2 is only fg
  (~8 min per (K, seed)). Screen s0–2 done ~10:30 PM → stage 2 s0–2 ~11:45 PM → seeds 3–4 ~1:45 AM. That is
  ~6 h, not 12: extend with K5 or more GDN seeds only if fg passes.

## Results

Interim, seed 0 only (2026-11-06 ~8 PM CST). Not a verdict: the screen decides on seeds 0–2.

Screen ratios (fit golden, variant / edy + swj, paired z):

| K | qk | vg | at | aw | gz | gd | dw | mA / swj (search) |
|---|---|---|---|---|---|---|---|---|
| 3 | 1.0038 (+3.6) | 1.0009 (+1.5) | 1.0034 (+3.5) | 0.9995 (−0.5) | 0.9905 (−7.2) | 0.9921 (−4.7) | 0.9906 (−4.8) | 0.9996 (−0.7) |
| 2 | 1.0121 (+8.3) | 1.0026 (+3.2) | 1.0125 (+7.8) | 1.0102 (+5.8) | 0.9666 (−9.0) | 0.9644 (−8.2) | 0.9601 (−8.2) | 1.0035 (+4.3) |

Held-out block residual (fit / base), K3 → K2:
- qk 0.96 → 0.88; at 0.95 → 0.87; aw 0.94 → 0.85;
- gd 0.95 → 0.92; dw 0.95 → 0.91;
- mA 0.985 → 0.969.

Reading so far:
- **Attention: the residual falls but KL rises.** All four attention variants are ≥ 1 at K2; writer weighting (aw) only
  removes the K3 harm. Lowering the attention output error on qad windows does not carry to the scored text, so scale
  refits at the logits / gate × v junction look like a negative.
- **GDN pays:** about −1% at K3 and −3.5–4% at K2 (z ≤ −4.7), with gz ≈ gd ≈ dw. The z gate alone (the swj
  analogue) carries most of it. Expected stage-2 arms: `st` + `fg` only (fc = fg, if attention stays null).
