# d1107 · #81-04 · A1 in KL on the 0.8B (load-time expansion)

Question (user, 2026-11-04): is A1 (down_proj coded as the residual after a per-neuron prediction from the decoded
up_proj, optionally also gate_proj) possibly on the frontier as a knap4 option (quality vs one extra GEMM)? This
measures the KL side; the kernel cost is timed separately on gfx1201.

Setup: the plain tuned base quants `k{K}base[_s1]` (K 2, 3, 4, 5; seeds 0, 1). Per arm, gate/up and everything else
are the base's; only the 24 down_proj are requantized, as R = D − pred (`a1q.py res`), at the same K, codebook i8,
seed offset, ladder settings (`q2b.sh` with `Q2B_CKPT`), then expanded exactly with the base's decoded up/gate
(`a1q.py expand`). Bytes equal the base's plus the coefficients (2 × fp16 per neuron, ~0.002 bpw at 27B scale).
Arms `a1u` (up only, the preferred kernel variant) and `a1ug`. Scored on `/data/g08-search1`, paired vs the base.

Pass rules (written 2026-11-04 ~5:25 PM CST, before data):
- **A1 is real in KL** at a K if `a1u` vs base has mean ratio < 1 with every seed < 1.
- **Worth a knap4 option** (subject to the kernel cost) if `a1u` ≤ 0.97 at K2 and K3 (mean of the 2 seeds); the 27B
  per-layer proxy gains (#81-03) then set its per-layer pricing.
- `a1ug` vs `a1u` reported for the record (the 27B proxy: ug keeps ~19% more gain for a second GEMM).
- Caveat: the 0.8B has 24 layers and its own depth profile; the 27B's gain was strongly layer-dependent.
