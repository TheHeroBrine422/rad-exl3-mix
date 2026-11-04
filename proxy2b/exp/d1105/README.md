# d1105 · healing recipe for the 27B, on the 0.8B (4070)

Stacks: edy H_out → swj → writers asym ρ 0 (d1104 `k{K}edyswjasym_r0[_s1]`). Scored on the search golden
(`/data/g08-search1`) with `pcmp.py`, paired, as d1104. Baseline healing = d1104 heal0 (2 epochs × 1,024 qad windows,
512 steps). Every arm has the same 512 steps (2,048 window-passes), so compute is equal.
Pass rules (written 2026-11-04 ~3:50 AM CST, before data):

| step | # | arm | rule |
|---|---|---|---|
| sp | 89-04 | one pass over 2,048 fresh qad windows (`--epochs 1 --ntrain 2048 --per-doc 64`), K3 / K2, seeds 0–1 | adopt single pass if mean ratio vs heal0 ≤ 1.01 at both K; "data-limited" (use all data on the 27B) if ≤ 0.98 at both K, every seed < 1 |
| mix | 89-05 | one pass over 2,048 windows of xcorp v1 (80%) + own agentic traces pisess-main (20%), assistant turns scored, same settings | mix is the 27B data if mean ratio vs sp ≤ 1.01 at both K (broader data at no cost on challenge-like text); if > 1.03, stay on qad |
| k45 | 89-06 | build edy → swj → asym ρ 0 stacks at K4 / K5 (seeds 0–1), heal with heal0's recipe | heal a size on the 27B if its K's healing ratio vs its stack ≤ 0.95 on both seeds |

Held-out early stopping stays on search-pool calib documents (not in the golden) for every arm. Runtime cost: none
(scale multipliers fold into the stored suh / svh).
