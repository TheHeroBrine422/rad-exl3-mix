# w75 · proposed INDEX rows and LOG entry (not applied: shared files; the lead merges)

Rules and results: `v3/e3/settle/w75.md` (rules written before data; amendment 1 to #106-01's bf16 gate after its data).

INDEX rows / updates:

| 112-01 | C9 gdn_out α beyond the grid edge (α ∈ {3, 4, 5, 6, 8}), st / pl stacks, K2 / K3, 0.8B | quantizer | none | **keep α 3** | 11-07 | 11-07 | rule `v3/e3/settle/w75.md`; 4070 g75.sh; fit picks 4–4.75; confirm vs α 3: geo 0.9987 / 0.9961 / 0.9987 / 0.9973 (st K3 / st K2 / pl K3 / pl K2), every seed < 1, verify agrees, none ≤ 0.995 → keep 3; no clamp needed (c ≤ 1.9 at α 3; clamp at 2: no effect) |
| 112-02 | C9 on the 27B writers (attn_o −1, gdn_out 3), fixed codes, swj K2.5 levels base / s1, golden-search27 + verify27 | quantizer | none | running | 11-07 | | rule `v3/e3/settle/w75.md`; gpubox `/data/c9w27/` (card 0); multipliers narrow at 27B (gdn_out c 1.000–1.069, mean 1.030); identity gate pass; interim s1 pair search 0.9911 (z −4.2), verify 0.9936 (z −3.1); base pair pending |
| 106-01 | o_proj head act order (heaviest stored last) on the real 0.8B quantizer, K2 / K3, 3 seeds | quantizer | none | **partial** | 11-07 | 11-07 | rule `v3/e3/settle/w75.md`; o_proj quantizer error −9% (0.910 / 0.916; L7 0.755) but search KLD 0.9978 (K3) / 1.0033 (K2, every seed > 1); no 27B pair proposed; bf16 fold exact (B == C bitwise) |

#112 row update: "follow-ups (w75): α > 3 not worth it (keep 3, #112-01); 27B pair #112-02 running".
#106 row update: "real-quantizer check (w75, #106-01): partial — quantizer error −9% but no KLD gain on the 0.8B (K2
+0.3%); no 27B pair on this evidence".

LOG entry (2026-11-07 ~6:45 AM CST · #112-01, #106-01, #112-02 · w75):
- **#112-01 (4070, 3 seeds, search deciding):** gdn_out C9 α past 3 gains only 0.1–0.4% more (α*_new 4–4.75; vs α 3:
  st K3 0.9987, st K2 0.9961, pl K3 0.9987, pl K2 0.9973; every seed < 1, verify agrees) → below the 0.995 bar:
  **keep α 3**; no second 27B arm. Multipliers ≤ 1.9 at α 3, no clamp needed (a clamp at 2 changes nothing).
- **#106-01 (4070, K2 / K3, 3 seeds):** o_proj heads stored heaviest-last, folded into q/k/v/o (exact: the permuted
  checkpoint == the original with only o_proj's summation order permuted, bitwise). The quantizer's error on o_proj
  drops **9%** (ourq proxy_err and H-metric agree; L7 −25%, where the heavy KV group was stored first), but the model's
  KLD does not follow: search 0.9978 (K3, mixed seeds), **1.0033 (K2, every seed > 1)**, verify agrees → **partial**;
  no 27B pair proposed.
- **Finding (0.8B bf16 noise):** permuting o_proj's summation order alone (one bf16 GEMM, both orders equally accurate
  vs f64) moves the 0.8B's bf16 logits by mean KL 7.2e-3 vs the original (median 4e-4, p99 0.17, top-1 98%), ~50× the
  "~1.5e-4 kernel noise" noted in `proxy2b/common.py`. A quantized model's mean KLD moved only 0.03% under the same
  change (plumbing check), so paired same-path comparisons are unaffected; arms that change the evaluation arithmetic
  itself carry noise of this size.
- **#112-02 (gpubox card 0):** running; s1 pair search **0.9911** (z −4.2), verify 0.9936 (p99 0.988 / 0.986, top-1
  +0.04 pt); base pair pending; verdict in `/data/c9w27/verdict.txt`.
- **Runtime:** none for all three (svh multipliers; a permutation folded into stored tensors).

BACKLOG "27B confirmation queue": #112 (C9 on the writers) is being tested (#112-02, no new row); #106 o_proj order:
no row (the 0.8B check did not pass).

Cleanup once logged (4070): `/data/yb/w75/m08perm` (1.7 GB permuted checkpoint), `p08perm`, `mix-*`, `dec-*`, `q-o*`.
