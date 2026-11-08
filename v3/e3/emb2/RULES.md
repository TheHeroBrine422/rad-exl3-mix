# emb2 screen rules (w83, #119) — written 2026-11-08 ~5:10 AM CST, before any emb2 data

Scope: CPU rate–distortion screens of new embedding / head representations against the current menu, on the real
bf16 rows of `/data/models/qwen38-27b-bf16` (colo). No GPU, no KLD (KLD only via the lead).

## Baseline (the menu we ship)

Embedding: fwht128 per 128 columns (orthonormal), one bf16 scale per row (RMS × factor, best of 0.9 / 1.05 / 1.2),
mul1 trellis on 256-value tiles at quarter-step K (sub-1 through the 0-bit-step pattern), `embtrellis2.py`'s method.
Rebuilt on the **same rows** as every candidate (paired), not read from `embmenu.jsonl`. Bytes = K + 16/5120 bit a
value, plus any side table a candidate needs (rotation matrices, centroids, band tables), charged in full to the
embedding's 1,271,398,400 values.

## Metrics

- **Primary (embedding): unweighted relative error** Σ‖w − ŵ‖² / Σ‖w‖² over the sampled rows, every row weight 1
  (design §6: H = I; #32 rules out frequency weights, and no per-row embedding Fisher exists).
- **Secondary (embedding): layer-0 reader metric**: error e weighted by M0 = diag(g)·(Σ_r W_rᵀW_r)·diag(g) / ‖·‖, g =
  layer 0's input-norm gain (1 + w), W_r = layer 0's input projections (the tensors reading the normed embedding).
  Reported as a ratio to the baseline's M0 error. A candidate whose primary passes but whose M0 ratio is worse than
  the baseline (> 1.00) at the same rates **fails** (it moved error into directions the model reads).
- **Head (secondary study):** relative proxy error tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) with H = the published calib
  `lm_head.input` Hessian (damped as ourq: 0.01 × mean diag), rows uniform (the head Fisher lives on iceman; row
  allocation is the slab head's job, #34, and is orthogonal to a column-side change).

## Rates and pass rule

Rates: 1, 1.5, 2, 3 bit/value (total bytes incl. side tables). A candidate **passes** if at **≥ 2 of the 4 rates** it
has **≥ 10% lower primary error at equal bytes**, or **saves ≥ 0.25 bit/value at equal primary error** (bits read
off the baseline's log-linear quarter-step curve), and the secondary check above holds at those rates.
Equal-bytes comparisons interpolate the baseline log-linearly between its quarter steps (on the hull, #97).

Sample: 4,096 rows (seed 5, as #35 / #97) for every trellis run; the analytic bounds (Gaussian reverse water-filling
from the covariance) use all 248,320 rows. A Gaussian bound is a ranking aid, never a pass.

## What a pass means

A CPU pass only buys a lead-run KLD check (fake-quant embedding / head donors swapped into one base at 1.5 and 2.0
bpw, paired, search-pool slice + multilingual guard `golden-ml`, as #32 / #35). Nothing ships on this proxy.

## Amendment 1 (2026-11-08 ~5:45 AM CST, after the analytic bounds in `specan.json` / `m0an.log`, before any trellis run)

The bounds (all 248,320 rows) close the transform candidates on paper: the normalized embedding rows are white to
second order (eigenvalue AM/GM 1.14: a KLT + water-filling saves ≤ 0.093 bit/value, less than an int8 rotation's
0.165 bit side cost); a head-predicted embedding leaves 85% of the energy; the head's H-metric water-filling beats the
LDLQ model by 0.12 bit before a 0.165-bit rotation. They are not screened further (`embklt.py` stays unused).

The remaining candidate changes **the quantizer's metric**, not the format: LDLQ on the embedding rows (fwht128 basis,
EXL3 16×16 tiles, per-row scale with the factor chosen per row) against a sensitivity matrix H = M instead of H = I.
The CPU proxy cannot say which metric predicts KLD (that is the question), so for this candidate:
- Metrics: I (unweighted, as above), **M0** (layer-0 reader metric, from the weights), **F** (diagonal Fisher of the
  residual stream at layer 0: Σdy² of layer 0's linear_attn.out_proj output, the 27B H_out diagonal collection
  `hdiag-dy`, qad pool). Arms: H = I (control, same tiles), H = M0, H = F; K 1, 1.5, 2, 3.
- **Pass (buys a KLD check, nothing more):** the task's rule, in **both sensitivity metrics** (its own and the other
  one, M0 ↔ F) at ≥ 2 of the 4 rates: ≥ 10% lower error at equal bytes, or ≥ 0.25 bit/value saved at equal error
  (read off the H = I control's curve). Its I-metric cost is reported (expected worse). Winning only in its own metric:
  fail (self-fulfilling). Note before data: the analytic high-rate bounds are ~0.2 bit (M0) and ~0.1 bit (F), so the
  0.25-bit branch is not reachable; only the 10% branch can pass.

## Amendment 2 (2026-11-08 ~6:25 AM CST, after the K1 / K1.5 arms of H = I / M0 / F)

M0 and F contradict each other (each arm wins only in its own metric: M0 arm −33…−38% in M0, +20…22% in F; F arm
−9…−10% in F, +3.5…4.8% in M0): both FAIL the rule above at K1 and K1.5. One more arm is added *after* seeing these
data: **H = MF = (M0 + F) / 2** (each normalized to mean diagonal 1), K 1 / 1.5 / 2 / 3, judged by the same rule (≥ 10%
lower error in **both** M0 and F at ≥ 2 rates). Because it was chosen after the first results, a pass counts only as
"KLD check optional", never as a ranking win on its own.
