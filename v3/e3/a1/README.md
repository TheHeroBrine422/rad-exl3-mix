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
