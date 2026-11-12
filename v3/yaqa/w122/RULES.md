# w122 rules: head donor build on ds8 (w119's adopted recipe), E3 (d) whole-model check, E5 guards, head price table

Written 2026-11-12 ~12:30 PM CST, before any quant, dump copy, slice, mix or score of this work. Committed alone before
launch. Task: `local/workers/instr3/w122.md`. Parent rule: `v3/yaqa/w119/RULES.md` (#155–#157); nothing there is
changed. Machine: ds8 (`/data/q27/heads/`, locks `/data/q27/locks/card<N>.lock` + `ram.lock` shared); box2 only to copy
w119's files; Thor (xcorp golden backup) and colo (xcorp chunk text) only to copy inputs. No gpubox, 4070 or dad1.

## Build (no decision; a reproduction of w119's adopted U recipe)

- Code: w119's staged tree (box2 `/data/q27/w119/code`, `exl3`), copied unchanged; same `ourq.py` flags as
  `run119.sh worker`: `--only-head --save-wq --head-perm R16.npy --qtf2 --frac-ext SRC --scaled-hessian --sigma-reg 0.025
  --codebook i8|mul1 --hout hd-R16 --hout-blocks group --sigma-reg-out 0.03`, same seed, f16 activations.
- **Identity gate (before any grid quant):** R16s003 K4 rebuilt on ds8 must equal w119's box2 donor byte for byte
  (`q-head.safetensors` and `wq/lm_head.safetensors` sha256). If not, STOP and report (no grid build on a different
  result).
- Grid (17): i8 K1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5 and K7 mul1. w119's R16s003
  K3, K4, K5, K6, K6.5 and R16s003m1 K7 are copied, the other 11 built.

## E5 — rare-script / rare-token guards on the 17 donors (report + flag)

On w119's fresh search-gen and fit-gen dumps (score119 unchanged; RULES #155 strata and definitions: rare-script =
gen rows of the 12 non-Latin-script sources, pooled, unweighted; rare-token = gen rows whose bf16 argmax has qad-gen
Fisher rank ≥ 20,000): absolute guard KL per donor, ratio to w119's S donor at the same K (K3–K7), and for K between
two S donors the ratio to the S guard interpolated log-linearly in K. **Flag** (not a re-decision of E2): a donor whose
rare-script ratio > 1.10 or rare-token ratio > 1.30 (E2's thresholds), or whose guard KL is above its lower-K
neighbour's in the grid on both sets. Below K3 (no S donor): reported only.

## Price table (report)

fit-gen, calib-weighted head-only KL (w119 pricing), top-1, rare-token guard KL / rows, per donor, written as a
headsim-style file `hcurve122.json` (`u<K>`; K7 mul1 as `u7`) that `v3/levels/reprice/headgrid.py` reads; the
unweighted KL beside it.

## E3 (d) — whole-model check of the uniform R16 head vs S-slab (w119 #156 (d), feasible form)

V3 pick trunks don't exist yet, so the trunks are **V2 3.00 and V2 4.00** (released containers, HF `Hero422`).
Pairs (head budget = bits per head weight; U = R16s003 uniform at that K; S-slab = w119's `slab/plan.json` plan
`s<B>-3` on the box2 S donors, or S uniform `S-K3` at 3 as w119's E3 used):

| trunk | budgets |
|---|---|
| V2 3.00 | 3, 4 |
| V2 4.00 | 4, 5 |

Heads spliced as bf16-dequantized rows in vocab order (`headslab.py build` on a copy of the bf16 container's
`output.weight`, then `radmix` into the trunk), identically for U and S-slab; V2's own head reported beside them.
Scores: ds8 `rad/install` + `rad/home`, bf16 KV, `--max-model-len 16384`, both parts.

**Slice (fresh, two-part):** `golden-hd122-x` = 130 pieces of night1105 `xc8k-b2-c1` (Thor backup of the bf16 golden,
corpus chunk from colo), `pickx.py` seed 11222, heavy categories capped at 2,048 scored positions, excluding the xcorp
documents of golden-search27b-x … 27f-x (27e-x rebuilt from its piece list) and of 27g-x if it exists when the slice is
built; hin27/mX's list is not reachable from here (gpubox): none of the arms is fitted on xcorp, noted in the registry.
`golden-hd122-c` = the **37** documents of w119's `docs119.json` `search_rest` (reserved by #156 (d)) that only 0.8B /
2B proxy slices have used, each cut to 4,096 tokens; the 27 never-used ones are left for golden-search27g. Ledger rows
(role check, line v3) written to `v3/yaqa/data/w122/` for the lead to merge before any other use.

**Statistic:** per part, mean whole-model KL per scored position; U vs S-slab paired per document (2,000-resample
document bootstrap, seed 0, resampled within each part); pooled = Σ KL over both parts' scored positions, ratio U / S.
**Rule (per pair):** w119's search-gen head-KL ratio U / S-slab at that budget (`rep/e3-T03-R16s003.json`) picks the
criterion: inside [0.95, 1.05] → **no-harm**: |pooled ratio − 1| ≤ 0.003; otherwise → **pass** iff ΔKL < 0 on both
parts and the pooled 95% upper bound < 1. Every pair passing = U confirmed whole-model. A failure is reported as
unverified until a confounder check (tensor hashes of the two mixes outside `output.weight` equal; a second slice).

**Runtime class:** build / E5 / price table: none (same format; R16 adds the decode-side logit un-permutation w119
already listed). E3 (d): none (a check).
