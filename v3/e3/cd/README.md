# #77 tile coordinate descent (C2) and #78 Gaussianized mul1 codebooks (D3): CPU proxy sweep

Ideas C2 / D3 of `docs/ideas/2026-11-02/ranked.md`. Tool `v3/e3/trellis2.py` (trellis.py's block-LDLQ + bitshift
trellis on 256 sampled rows of real 27B tensors, calib Hessians, the 128-point Hadamard, damping 0.01, scale factor
from {1, 1.15, 1.3}). Thor, `run.sh`.

Tensors (16): layers 10 / 30 / 50 (GDN) × ssm_qkv, ssm_out, ffn_gate_up, ffn_down; layers 19 / 43 (full attention)
× attn_qg, attn_output. K = 2, 2.5, 3, 4. Metric: rel = tr(dW H dW^T)/tr(W H W^T), paired per tensor (same rows,
seed, scale factor). Summary = mean over tensors of the per-tensor ratio, and the parameter-weighted mean.

Pass rules (written 2026-11-02 ~16:30 CST, before data):
- **#77 C2** (codebook mul1_i8, 3 sweeps): passes to a 0.8B KL test if the cd3/ldlq ratio is ≤ 0.98 at K2.5 in
  both means and ≤ 1.0 for at least 13 of 16 tensors; report K2 / K3 / K4 too. Caveat: in-sample (the same H
  searches and scores; DRQ warns proxy gains can be in-sample), and the model's tail-biting / start-state constraint
  is not modelled (free start), as in trellis.py.
- **#78 D3** (LDLQ only): a codebook passes to a 0.8B KL test if its ratio to mul1_i8 is ≤ 0.995 at both K2.5 and
  K3 in both means and ≤ 1.0 at K4. Codebooks: gmul1_i8 (exact Gaussianization, 1,021-entry table), cubic0.02_i8,
  cubic0.05_i8 (v + C v^3, 2 FMAs). If more than one passes, the cheaper decode wins unless the better one leads by
  ≥ 0.5%.
