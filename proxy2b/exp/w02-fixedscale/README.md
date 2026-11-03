# w02-fixedscale: fixed-code scale fits (ranked.md B1, C9, C10)

Worker w02, 2026-11-02. Status: **prepared, no GPU run yet**. The pass rules below are drafts for the lead to commit
before any data.

## Shared mechanism: the codes never change

`quantize_exl3` stores W_q = diag(su) · Had Q_r Had · diag(sv), where Q_r is the decoded trellis. Its
`refit_scales` only multiplies su by r and sv by c. Any further scale fit is therefore
**W_q' = diag(r) · W_q · diag(c)**, applied to the dequantized weight that `ourq.py --save-wq` already wrote
(`/data/yb/q-<tag>/wq`, f16, read back in f32).

- No codes, rotations or trellis decode are needed for the 0.8B. Selftest 5 checks the identity.
- The container form is `suh *= r, svh *= c`. `refitq.py --shards` writes it; selftest 6c checks it reproduces the new wq.
- Every arm is a fixed-code comparison against its own source quant. The paired per-document z then carries no
  rounding noise (#33-15/16), so **one seed suffices**.

## Files

| file | what |
|---|---|
| `run.sh [b1\|c9\|c10\|all]` | 4070 entry point (allocator cap 7.2 GB). Idempotent; results in `/data/yb/w02-fixedscale/result.txt` |
| `refitq.py` | B1 refit (rounds, undamped, h-weighted input solve) and C9 per-family svh multipliers, on an existing quant |
| `swiglufit.py` | C10 joint gate/up output-scale fit through the SwiGLU (bf16 model's MLP inputs, fit-pool documents) |
| `pstat.py` | pcmp.py's paired statistics, the C9 α pick from fit-slice arms, and the draft-rule verdict tables |
| `fsutil.py` | shared helpers (Hessian / checkpoint / wq I/O, container edit, SwiGLU sums) |
| `refit-exllamav3.patch`, `refit-ourq.patch` | B1 in the quantizer: `refit_scales(out_weight=, fit_undamped=)`, `ourq.py --refit-rounds / --refit-undamped / --refit-hout` |
| `selftest.py` → `selftest.txt` | CPU, synthetic: patch no-op, refit equivalence, h-weighting, container edit, refitq end to end, SwiGLU fit vs brute force, ourq options off = old stamp settings |
| `selftest-run.sh` → `selftest-run.txt` | run.sh's control flow on stubs: all stages finish, a second run makes no new score files, B1_FULL makes 18 B1 arms |
| `cpucheck.sh` → `cpucheck.txt` | pre-GPU predictors on the real 0.8B data (4070 CPU, read only) |

## B1: H_out-aware, undamped, multi-round `refit_scales`

**Question.** `refit_scales` ignores H_out. Does an input-side solve weighted by diag(H_out), fitted in the undamped
Hessian with more rounds, lower KL on the bdiag quants (and on base)?

**Mechanism.** Minimise Σ_n h_n e_nᵀ H e_n:
- Output step: per column, unchanged (h_n cancels).
- Input step: ((Q diag(h) Qᵀ) ∘ H) r = rowsum(Q ∘ (HW) ∘ h).

**Arms** (search slice `g08-search1`, h = `dv8192`, as the bdiag quants used):
- `h6`: `--rounds 6 --undamped --hout dv8192`, on k{3,2}bdiag and k{3,2}base.
- `u6`: the same without h, on bdiag. Attributes the gain between undamping/rounds and h.
- Default is seed 0 only (4 evals). `B1_FULL=1` runs all 3 seeds plus u6 (18 evals).

**CPU predictor (`cpucheck.txt`): B1 is nearly inert.**
- u6 changes nothing: the stored 2-round damped refit is already converged.
- h6 lowers the h-weighted proxy by only 0.08% (K3) and 0.15% (K2) over the trunk (best tensor −2% / −4%).
- Reason: with a diagonal H_out only the k-vector of input scales can respond to h, and it cannot target output rows.

**Draft pass rule.**
- *Seed-0 screen.* `h6` vs its bdiag source at K3 and K2. If ratio > 0.995 or z > −2 at both K, B1 is closed; do not run B1_FULL.
- *Full rule* (B1_FULL, per K, on bdiag):
  - every seed ratio < 1.000;
  - mean ratio over the 3 seeds ≤ 0.990;
  - z ≤ −3 on at least 2 of 3 seeds.
- *Same rule on base.* If base passes too, H_out-via-refit works on one-sided quants.
- *Attribution.* If u6 gets ≥ 2/3 of h6's mean gain, ship u6 (needs no H_out).
- *Harm.* Any seed > +1% at z ≥ 3: that variant is not merged.

**Decision it drives.**
- PASS: merge both patches. The 27B swap test (ranked K1) and the Phase-B requant (#58) run with `--refit-hout`. The F1 re-pricing must use the refit quants.
- FAIL (expected): drop B1 from the top 25 and leave the patches unmerged. That also removes "B1 joins the 27B swap test" from ranked §6.

**GPU time.** ~6 min default (4 refits of ~20 s, 4 evals). B1_FULL adds ~20 min.

## C9: KL-optimal svh multiplier per (level, family)

**Question.** The least-squares output refit shrinks each row: a fitted row has |q|²_H = |w|²_H (1 − e). Is a
different output scale better for KL?

**Parametrisation.** One scalar α per (level, family), applied per row as c_n *= (1 − e_n)^(−α/2):
- α = 0: no change.
- α = 1: restores each row's H-norm (undoes the shrink).
- α < 0: more shrink.

This is the ranked "ε" scaled by each row's own relative error, so one α fits every tensor of a family.

**Families:** attn_in (q/k/v), attn_o, gdn_in (qkv/z), gdn_out, gate_up, down. q/k (and the GDN q/k rows) pass through
per-head norms, so they are a built-in near-null control.

**Arms.** All on k{3,2}base seed 0.
- *Fit slice* `g08-fit1` (fit pool):
  - base;
  - α_all ∈ {−1, 1, 2};
  - each family alone at α ∈ {−1, 1, 2} (18 per K).
- *Pick* (`pstat.py pick`, written before data):
  1. Per family, fit a quadratic through (0, −1, 1, 2) to the paired fit-slice diffs.
  2. α\* = its vertex, clipped to [−1, 2], or the best measured point if the quadratic is concave. Rounded to 0.05.
  3. Set α\* = 0 unless the best measured point beats base at z ≤ −2 on fit.
- *Confirm:* the combined α\* arm on search vs k{3,2}base, and the same α\* applied to k{3,2}base_s1 (transfer to other codes).

**Draft pass rule (per K).** All three must hold:
- α\* arm vs base on search: z ≤ −3 and ratio ≤ 0.995;
- the seed-1 transfer has ratio < 1 (same sign);
- α\* is not all zeros.

**Decision it drives.**
- PASS: per-(level, family) α joins the quantizer as a free final step (a follow-up `ourq` option, or `refitq.py --shards` on containers). The 27B gets it by container edit after one 27B fixed-code check at K3. The sign of α also says whether LS shrink costs KL, which feeds B1/H3's design.
- FAIL: closed. Stored scales stay LS-optimal.

**GPU time.** ~50 evals (fit-slice ≈ 40 s, search ≈ 70 s) ≈ 45 min.

## C10: SwiGLU-joint gate/up output scales

**Question.** down_proj reads a = silu(g)·u, but gate and up are each refit only to their own outputs. Fitting
(c_g, c_u) per MLP channel to a directly, on fit-pool activations: does that lower KL?

**Mechanism.** Per channel j, min Σ_t (silu(g_t)u_t − c_u silu(c_g ĝ_t) û_t)².
- c_u is closed form for each c_g; c_g is a grid search over 0.6..1.4 in 0.01 steps.
- Sums are streamed over positions, so nothing is stored per token.
- Inputs x are the bf16 model's MLP inputs over the `g08-fit1` documents (~100k tokens).

**Arms** (search slice):
- `swj` (joint) and `swu` (c_g = 1, up only) on k{3,2}base;
- `swj` on k{3,2}bdiag.

**CPU predictor (`cpucheck.txt`, 22k fitting tokens, scored on held-out documents).**
- MLP-output residual −1.3% (K3) and −2.5% (K2); up-only gets about half of that.
- c_g > 1 in every layer (+1.4% at K3, +5% at K2): the gate is systematically shrunk through the nonlinearity.
- Expected KL −0.2 to −1%.

**Draft pass rule.**
- `swj` vs base: ratio ≤ 0.995 and z ≤ −3 at both K3 and K2.
- The "joint" part counts only if `swj` beats `swu` at z ≤ −2. Otherwise ship up-only, which is cheaper to explain.
- The bdiag arm is reported as stacking evidence and does not gate.

**Decision it drives.**
- PASS: a 27B version, i.e. a layer-streamed fit-pool forward on iceman writing gate/up svh. Not written yet; ~half a day of work. It becomes a free per-level step (fit pool only). It also hints that a gate-specific shrink fix (α for gate alone) might capture most of it.
- FAIL: closed.

**GPU time.** 4 fits (~3 min each, mostly the forward) + 6 evals ≈ 20 min.

## Totals, run, review

- Command: `bash /workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale/run.sh all` on the 4070, or one stage at a time with `b1` / `c9` / `c10`.
- Time: **~1.3 h of the 4070 GPU** (+20 min with `B1_FULL=1`).
- Needs the existing `/data/yb/{k3,k2}{base,bdiag}{,_s1,_s2}` quants and score files, `/data/yb/dv8192/hout`, `/data/h08/calib`, `/data/g08-{fit1,search1}`.
- Verdict tables: `python3 pstat.py b1|c9|c10 /data/yb/w02-fixedscale /data/yb`.
- Data discipline: C9 α and C10 scales are tuned on fit-pool data only; B1 has no tunable parameter. Search is the judging slice. No public data.
- Shared code: two patches, no-op when off (selftest 1, 8):
  - `refit-exllamav3.patch` applies with `git apply -p1` in `forks/exllamav3-frac`;
  - `refit-ourq.patch` applies in `rad-exl3-mix`.
  - With the options unset, the ourq stamp's settings hash is unchanged; the file hash changes, as with any ourq edit.
  - The weighted solve holds one extra (k, n) copy; fine for the trunk, ~5 GB on the 27B head.
- Not covered:
  - B1 on the 27B by container edit needs `--save-wq` output, or a trellis decode of the shards (not written).
  - The patch covers fresh quants.
