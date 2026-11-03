# #81 · Faraway A1 on the CPU trellis proxy: down_proj coded as the residual after predicting from decoded up / gate

Idea FA-A1 (`docs/ideas/2026-11-02-faraway/faraway.md`; BACKLOG "Faraway lens"). Tool `a1test.py`, driver `a1run.sh`
(Thor and colo, half the layers each). Layers 30 10 50 2 6 14 18 22 26 34 38 42 46 54 58 62 (L30 first: the pass-rule
layer). K 2 / 2.5 / 3; codebook cubic0.02_i8 (the adopted i8c); predictors = the decoded gate_up at the same K
(stage 1 quantizes every gate_up row of the layer); down on trellis2's 256 sampled rows.

Arms per (layer, K): `D` (control), `ug` (residual after up + gate), `u` (up only: one extra GEMM instead of two),
`ugtrue` (oracle: unquantized predictors; checks faraway's measured 0.09–0.23 bit). Metric rel_D: coded error
relative to D's H-weighted energy, the same for every arm.

Pass rules (2026-11-02 ~20:25 CST, written before data):
- **A1 worth a GPU test** (0.8B / 27B KL with load-time expansion, then the kernel-cost question) if `ug / D` ≤ 0.85
  at K2 and K3 on L30 (faraway's rule) **and** the parameter-weighted mean over layers is ≤ 0.90 at K2.5.
- **Up only** is the preferred variant if `u / D` keeps ≥ 75% of `ug`'s reduction (1 − ratio) at K2.5 (mean).
- Constraint (deadcode, 2026-11-02): VRAM is the target, so a pass only buys a kernel study (online prediction,
  +1–2 GEMMs in the MLP); the gain must then be weighed against that speed cost.
- In-sample caveat as #77 (the same H fits a, b only through D itself, so not much); a, b side info not counted
  (0.006 bpw).

## #81-02 / #81-03 (2026-11-03 ~8:45 AM CST, written before data)
Result of #81: ug/D 0.83, but the predicted share of D's H-weighted energy is only ~4.5% (negative on L54/58/62, where
the coded error still drops 3–9%). So part of the drop may come from coding *any* residual, not from the prediction.
- **#81-02 control:** arms `ugperm` / `uperm` = the same fit with each neuron's predictors taken from a random other
  neuron (fixed permutation), K2.5 / K3, the 16 layers. **The reduction is real** if (1 − ugperm/D) ≤ 0.5 × (1 − ug/D)
  at K2.5 (mean over layers); else A1 is mostly an artifact of the residual coding and goes no further.
- **#81-03 (only if #81-02 clears):** the other 48 layers at K2.5 (D / ug / u) → the whole-model parameter-weighted
  ratio, the number a GPU test would be priced on.

## Runtime cost (user rule, 2026-11-03)
**P+D**: online prediction adds a transposed GEMM over the up weights (+ one over gate for `ug`) in every MLP: roughly +1/3 of MLP work in prefill (`gemm_us`) and, in decode, re-reading up_proj's bytes (~+1/3 of MLP bandwidth). The VRAM target rules out expanding at load time. A pass here buys a kernel timing, then a comparison against spending the same bytes / FLOPs on higher K.
