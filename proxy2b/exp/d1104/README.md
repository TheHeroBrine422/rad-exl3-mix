# d1104 · the 0.8B stack in the order it would ship (H_out → sibling refits → writers ρ 0 → healing)

Driver `run.sh` (4070, one GPU job at a time, idempotent; reuses d1103's helpers and arms). Results
`/data/yb/d1104/result.txt`. Pass rules (2026-11-03 ~11:35 PM CST, written before data):

| step | # | what | rule |
|---|---|---|---|
| swj | 85-03 | C10 joint gate/up scale refit alone on edy (no α\*), seeds 0–2 | real if every seed < 1 and mean ≤ 0.99 (K3, K2) vs edy |
| order | 87-08 | edy → swj on the siblings → moments on the refitted siblings → writers asym ρ 0, vs edy → writers ρ 0, seeds 0–2 | swj pays on top in this order if mean ≤ 0.99 and every seed < 1 at K3 and K2 |
| heal0 | 89-03 | healing (heal.py defaults, scored with its folded norms) on the best of the two stacks above, seeds 0–1 | stacks if ≤ 0.95 (K3) / ≤ 0.90 (K2) vs its stack, both seeds |

Runtime cost: none for all (quantizer-side; healing folds into svh / suh / norms).
