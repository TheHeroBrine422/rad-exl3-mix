# d1109 · REAL-Q on the 0.8B, on top of our stack (#92)

Question (user, 2026-11-02 queue; started 2026-11-05): does REAL-Q (arXiv 2609.00049: block-output Fisher surrogate
F = mean g gᵀ from one end-to-end backward; after every 128-column block, one Adam step on the still-unquantized
columns; sequential, quantized-prefix inputs) improve on our V3 quantizer stack?

Arms (0.8B, 4070, gen-masked KL on `/data/g08-search1`, paired; K3 and K2; seeds 0/1/2):
- **stack** = edy (Σdy² H_out) + swj + writers asym ρ0 (existing `k{3,2}edyswjasym_r0[_s1,_s2]`).
- **rq** = REAL-Q with our settings and the same H_out, then swj + writers asym as in the stack.
- **rq-noH** = REAL-Q without H_out (does it replace H_out?). Reported, no rule.

Rule (written 2026-11-05 ~3:20 AM CST, before code or data): **rq earns a 27B test if its mean paired ratio vs stack
is ≤ 0.95 at both K3 and K2** (every seed < 1). The paper's gain shrinks with size (−49% at 1.7B, −5…−9% at 8–70B),
so a 0.8B pass is necessary, not sufficient. REAL-Q is per mix (quantized prefix), so it cannot ride on ladder levels:
if it passes, its place (final pass on shipped sizes vs V4 with healing) is the user's call.
Runtime cost: none (quantizer only); quant time recorded.

## Amendment 1 (2026-11-05 ~3:45 AM CST, before code or data): does REAL-Q survive healing?
Cost context (user): REAL-Q on the 27B is ~40–50 card-hours per mix (sequential, per mix), scale healing ~2–3.
- **Arms added:** `stack+heal` and `rq+heal`: d1105's single-pass scale healing (`heal.py`, qad data, 2,048 windows,
  1 epoch, the d1105 settings) on top of `stack` and of `rq`. K3 and K2, seeds 0 and 1.
- **Rule:** REAL-Q earns a 27B test only if **both** the base rule above passes **and** `rq+heal` vs `stack+heal` is
  ≤ 0.95 at K3 and K2 (every seed < 1). If the base rule passes but healing absorbs the gain (ratio > 0.95 after
  healing), REAL-Q is recorded as not worth its 27B cost, since healing is far cheaper per mix.

## Amendment 2 (2026-11-05 ~4:00 AM CST, before code or data): parallel REAL-Q
- **Arm `rqp` (+ `rqp+heal`):** REAL-Q's block-output Fisher and the per-128-column Adam step kept, but every layer
  quantized on **bf16** (unquantized-prefix) inputs, so layers are independent (per-layer jobs, ladder-compatible,
  volunteer-friendly). Same K, seeds, H_out and healing settings as `rq`.
- **Rule:** with `rq` passing the rules above, `rqp` **replaces** `rq` (as a per-level ladder method) if it keeps
  ≥ 75% of rq's gain vs stack in mean log ratio, both unhealed and healed, at K3 and K2. If `rq` fails but `rqp`
  alone meets the base rule and amendment 1's rule, `rqp` earns the 27B test instead. Otherwise sequential REAL-Q
  stays per mix (one mix per donor or our cards) and only if it passed.
