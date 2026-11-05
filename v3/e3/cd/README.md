# #77 tile coordinate descent (C2) and #78 Gaussianized mul1 codebooks (D3): CPU proxy sweep

Ideas C2 / D3 of `docs/ideas/2026-11-02/ranked.md`. Tool `v3/e3/trellis2.py` (trellis.py's block-LDLQ + bitshift
trellis on 256 sampled rows of real 27B tensors, calib Hessians, the 128-point Hadamard, damping 0.01, scale factor
from {1, 1.15, 1.3}). cpu2, `run.sh`.

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

## Final read (2026-11-02 ~19:10 CST, all 16 tensors; `data/e3/cd-cpu2.jsonl` + `cd-cpu1.jsonl`)

- **#77 cd3 / ldlq** (mean, param-weighted, tensors ≤ 1): K2 0.990 / 0.991 / 15 of 15; **K2.5 0.992 / 0.993 / 16 of
  16**; K3 0.993 / 0.995 / 16; K4 0.991 / 0.993 / 16. Fails the written 0.98 bar; under the two-question rule
  (2026-11-02) it is real (every tensor, every K) but small (−0.7%) → replicate + stacking below, then decide on a
  GPU version (2–3× quant time).
- **#78 vs ldlq mul1_i8:** cubic0.02 K2.5 0.987 / 0.989 (16 of 16), K3 0.982 / 0.986, K4 0.979 / 0.985 → **passes**;
  gmul1 equal within 0.1% (no reason for the table); cubic0.05 worse at K2–K2.5. 0.8B KL: YAQA08.md #78 / amendment 15.

## #79 / #80 (2026-11-02 ~19:15 CST, rules written before data)

- **#79 stacking (cpu2, `run5.sh`):** `CD=1` with codebook cubic0.02_i8, K2.5 / K3, rows seed 0 (paired with the
  ldlq_cubic0.02 rows above). CD **stacks** on the cubic codebook if cd3_cubic / ldlq_cubic ≤ 0.995 at K2.5 and K3
  (both means) with ≥ 13 of 16 tensors ≤ 1.
- **#80 row-sample replicate (cpu1, `run4.sh`, `E3_ROWSEED=1`, other 256 rows):** cd3 / ldlq (mul1_i8) and
  cubic0.02 / mul1_i8 at K2.5 / K3. #77 **holds** if cd3 / ldlq ≤ 0.995 at both K (both means) with ≥ 13 of 16
  tensors ≤ 1; #78 holds if cubic ≤ 0.99 at K3 (both means). Output `cd-cpu1-rs1.jsonl` (the JSON has no row-seed
  field: never mix it with the seed-0 files).
- Both stay in-sample (same H searches and scores).

## Runtime cost (user rule, 2026-11-03)
CD (#77): none (codes only; 2–3x quant time). Cubic codebook (#78, i8c): P+D? (2 FMAs per weight in the decode; may hide in memory-bound decode, maybe not in the int8 prefill tile).
