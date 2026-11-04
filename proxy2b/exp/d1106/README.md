# d1106 · A4 (int4 activations) quality at low K, on the full stack (0.8B, 4070)

Question (user, 2026-11-04): is A4 on the MLP usable for the fast variant at ≤ 2.25 bpw? Quality only: activations
fake-quantized exactly as the int kernel rounds them (`evalq.py` EVALQ_AQ: x·suh → 128-point Hadamard → symmetric int
per group → exact inverse; suh from the stack's own shards). Speed needs the R9700 (radiance's a4 tile), separately.

Stacks: edy → swj → writers asym ρ 0 at K3 / K2 (d1104) and K1.5 (built here), seeds 0–1; healed versions (d1105
single pass) at K3 / K2. Arms: `a4` = gate / up / down at int4, groups of 64, clip 1.0; `a4c` = clip 0.85 (screen,
seed 0); `a8` = every trunk linear at int8, groups of 128 (the fast variant's base path, reference).
Pass rules (written 2026-11-04 ~3:40 PM CST, before data):
- **A4 on the MLP goes into V3's fast variant at ≤ 2.25 bpw** if its KLD ratio vs the same stack without it is
  ≤ 1.10 at K2 and K1.5 (mean of the 2 seeds, every seed ≤ 1.13), on the unhealed and the healed K2 stack alike.
- K3 is for the record (expected well above 1.10: the cost is roughly constant in absolute KLD).
- The clip screen picks the clip for the 27B if a4c beats a4 by ≥ 1% at K2.
