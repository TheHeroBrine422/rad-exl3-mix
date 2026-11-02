# 2B proxy calibration study (pass rule written 2026-11-02, before any 2B quant was scored)

Question: does Qwen3.5-2B reproduce known Qwen3.8-27B quantization results closely enough to screen trunk methods on?

Setup: ourq with the trunk ladder's settings (`proxy2b/q2b.sh`: qtf2, scaled Hessian, damping 0.025 / input 0.005,
calib input Hessians of the 2B, `/data/h2b/calib`), whole trunk at one K (embedding / head bf16: tied), fake-quant
scored by `evalq.py` on a 2B golden of **search-pool** documents (`g2b-search1`: ~300k tokens, 2k cap, seed 1;
replication checks are not tuning, but the fit pool also feeds nothing here). Paired comparisons with a
per-document bootstrap (`pcmp.py`).

| # | 27B result | 2B test | reproduces if |
|---|---|---|---|
| 1 | KLD falls smoothly with K (fractional levels on the curve) | uniform K2, 2.5, 3, 3.5, 4 (i8) | strictly decreasing, and log KLD vs K within 15% of linear between neighbours |
| 2 | i8 codebook costs ~nothing to K6 (proxy x1.000) | K3 mul1 vs K3 i8 | paired difference within 2 SE of 0, or < 2% |
| 3 | output scales off is worse (K3 +2.5%, z 2.1) | K3 i8, output scales on vs off (all tensors) | off worse, z > 2 |
| 4 | scaled Hessian helps (held-out proxy ratio 0.88 at K2.5) | K2.5 i8, scaled Hessian on vs off | on better, z > 2 |

**Pass:** at least 3 of 4 reproduce. Then the 2B screens trunk-method ideas (direction and rough size); a winner is
still confirmed on the 27B before anything ships. **Fail:** 2 or fewer -- record which, and the 2B is not used for
screening until the cause is understood.

Not tested here (needs more): mixed vs uniform at equal bpw (needs a 2B knapsack fit), damping sweep gain (small on
the 27B, mostly noise), YAQA (0.8B, separate study).
