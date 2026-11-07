# w64 · proposed INDEX rows and LOG entry (not applied: shared files; the lead merges)

INDEX rows (after #108):

| 109 | Per-rate int8 factor ("clipped i8", α finer int8 step) at K6.5–8, order-fixed CPU proxy (settle64) | weight codes | D | negative | 11-07 | 11-07 | rule `v3/e3/settle/w64.md` item 2; w64, cpu1+cpu2; α never helps at K ≤ 7 (K7 i8/mul1 1.239 at α 1, worse at every α > 1), ≤ 3% above; i8 saturates at mul1 K≈6.9; i8 stays as priced |
| 110 | bf16 (row, 128) scales on attn_output / ssm_out at equal bits (K−⅛), order-fixed CPU proxy (#95-02 retest, IQR-V4's setting) | format | P+D | fail | 11-07 | 11-07 | rule `v3/e3/settle/w64.md` item 4; post / pre / prepost ×1.14–1.18 vs plain at K4–7 (⅛ bit is worth ×1.18); closed for V3 |
| 111 | Writers' asym (#87) under knapsack mixes: per-level asym writers next to siblings of another K, 0.8B | quantizer | none | **fails** | 11-07 | 11-07 | rule `v3/e3/settle/w64.md` item 1; 4070; xs/xc geo 1.077 (3 on 2.5) / 1.229 (3 on 3.5) / 1.239 (2 on 2.5), per-mix writer pass xm/xc 0.81 / 0.90 / 0.88; verify agrees; lead picks per-mix pass / tied K / plain writers |
| 112 | C9 gain refit (α) on the writers after asym (and plain), 0.8B | quantizer | none | running | 11-07 | 11-07 | rule `v3/e3/settle/w64.md` item 3; 4070 g64.sh; readout `/data/yb/w64/report.md` |

#106 row update: "re-run on exllamav3's LDLQ order (w64 item 5): o_proj **holds** 0.971 / 0.970 / 0.971 (−3%, was −7.3%),
out_proj **flips to pass** 0.980 / 0.978 / 0.980 (marginal); heaviest heads must be stored **last** (stored first: +10%)".
#103 row update: "order-fixed re-run: still fail, 0.990 at K2 / 3 / 4 (L0 0.91, L63 0.96)".
settle.py note (README / LOG): LDLQ order fixed in `w64/settle64.py` (`SETTLE_ORDER=rev`), matches w50's GPU tile-CD
ratios within 0.4% on 9 / 9 blocks; the forward order moved L3 attn_qg LDLQ error by +14…+26%.

LOG entry (2026-11-07 ~3:45 AM CST · #109–#112, #106 / #103 re-run · w64): see `v3/e3/settle/w64.md` "Results".
**Runtime:** #109 decode (not adopted), #110 prefill + decode (not adopted), #111 / #112 none, #106 none (permutation
folded into stored tensors).
