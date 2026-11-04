# hconv · quantizer-level convergence of the 27B H_out collection (#33-25)

Question (user, 2026-11-04): when can the 27B Σdy² collection (#33-24) stop, without a whole-model swap test?

Method: split the cumulative snapshots into independent 128-sequence blocks (`mkhalf.py`). Quantize layers 10 (GDN),
39 (full attention) and 50 (GDN) at K2.5 with the ladder's settings (i8, qtf2, scaled Hessian, damping 0.025 /
0.005), once without H_out and once per snapshot N with the **even-block** Σdy² mean up to N (≈ N/2 sequences), output
damping 0.01. Score every arm's dequantized linears by the output-weighted proxy error under the **odd-block** mean
(independent data): E_out = Σ_i h_i δ_i H_in δ_iᵀ / Σ_i h_i w_i H_in w_iᵀ (`hscore.py`), reported as the param-weighted
ratio to the no-H_out arm. Same rotation seed for every arm. 4070 (`hconv.sh`), snapshots synced from iceman by
`sync.sh` (VM). Statistical curve beside it: `v3/yaqa/hdmon.py` on iceman.

Reading rule (written 2026-11-04 ~1:05 AM CST, before data):
- The curve E_out(train-N) / none vs N should fall and flatten. **Collection converged** at the first N where
  doubling the training sequences (train-N vs train-N/2) improves the param-weighted ratio by < 0.5% **and** no layer
  improves by > 1%. Since train-N uses half the sequences, the full collection at 2N is then past the plateau.
- Caveats: three layers only; the reference (odd blocks) is itself noisy at small N (it tightens as the collection
  grows; earlier arms are rescored each run); proxy error is not KL (the swap test still decides whether H_out helps).
- Runtime cost: none (measurement only).
