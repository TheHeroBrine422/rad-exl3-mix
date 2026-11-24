# V3.1 QAD idea list (lead, 2026-11-22 ~9 PM CST)

Every QAD idea raised since the w214 pilot, with its status, plus new ones (marked **new**). Status words: **done** (result in LOG),
**running**, **queued**, **backlog** (not started), **dead** (tested negative), **dropped** (user / not worth it), **later** (V4 or gated).
Numbers are j217 KL ratios vs unhealed unless noted. Details: `docs/experiments/LOG.md`, `BACKLOG.md`.

## A. Optimizer and gradient noise

| # | idea | status | result / note |
|---|---|---|---|
| A1 | lr sweep (1e-3 / 3e-4 / 1e-4) | done; 4M curves running | 11-23: 3e-4 constant plateaus at ~0.85 from 1.5M (4M run); 1e-4 to 4M and the 1.5M→1e-4 step fork pending.  2.5: 3e-4 best (0.842 @ 2M), 1e-4 0.861–0.866 @ 2M, 1e-3 drifts. 3.0: 3e-4 noisy. 4.0: only 1e-4 is stable (~1 % gain). 3e-4 / 1e-4 to 4M at 2.5 running |
| A2 | low-rank projected scale steps (GaLore-style) | dead | w218: held-out share ≈ 0 at 100k–500k rows; directions don't transfer |
| A3 | late layers only (32–63 / 48–63) | dead | worse at 2.5 (0.93 vs 0.89 @ 0.5M), tie at 3.0, null at 4.0 |
| A4 | signal-masked scales (3-SE mask) | dropped | would freeze the front half, which A3 shows still gains from training |
| A5 | tied scales in the front half (128-groups) | inconclusive | tested only at lr 1e-3 (0.939 [0.83, 1.02] vs a blow-up control); a 0.5M retest at 3e-4 is cheap |
| A6 | gradient clipping + β2 0.99 | done, null | clip ties at 2.5 / 3.0 / 4.0; β2 was always 0.99 |
| A7 | curvature line search (step = −g·d / dᵀHd along Adam's update) | demoted (11-23) | mostly an automatic lr schedule; with noise-dominated gradients the curvature estimate on a small batch is itself noisy; the schedule tests cover the same ground |
| A8 | low-rank + diagonal Fisher preconditioner | dropped | the low-rank part is dead (A2); the diagonal part is Adam's v |
| A9 | batch size (1× vs 4×) | done | 4× at equal rows worse by 2M (lr unscaled); √-scaled lr untested; user: not in practice |
| A10 | rounding-divergence diagnostic | backlog, low | — |
| A11 | Adam ε | done | keep 1e-8 (ε acts like a smaller lr) |
| A12 | parameter EMA (Polyak averaging) | dropped (user 11-24) | EMA 0.9 1.010 vs raw at 0.5M; 0.97 helped 3.0 slightly at 1–2M (0.965–0.993), not worth the complexity |
| A13 | trust region on scale moves | dropped | no outlier scale moves |
| A14 | per-document gradient clipping / normalisation | **dead** (11-23) | CD25 1.021 [0.996, 1.061], CD30 1.037 [1.002, 1.075] at 0.5M; was:  w218: chunk gradient norms vary ~35×; GN: front-half batch gradients heavy-tailed (below-chance significance at 2M) |
| A15 | robust batch aggregation (trimmed mean / median-of-means across documents) | dropped (11-23) | same idea as A14, which lost | the heavy-tail fix without picking a clip threshold; per-doc gradients need one backward per doc group (cost ↑) |
| A16 | Sophia / AdaHessian-style diagonal second-order step | backlog, low | not just a schedule (per-scale step ∝ g / h), but the h estimate is noisy where g is; low priority | the "second derivative" step per scale; Sophia was built for noisy LLM training; ~1 extra Hessian-vector product every k steps |
| A17 | per-layer-group lr (front 1e-4 / 3e-5, back 3e-4) | running (LF25-1e4, LF25-3e5) | sdelta: the front's 1M→2M movement is a random walk under Adam | the front gains slowly under noise; a higher front lr lets the weak drift accumulate (risk: noise grows too) |

## B. Schedule and data

| # | idea | status | result / note |
|---|---|---|---|
| B1 | step decay forked at 0.5M (3e-4 → 1e-4) | done | hurts: 1.028 [1.003, 1.060] vs staying at 3e-4 @ 2M |
| B2 | whole-run cosine vs linear decay | done @ 1M; 2M queued | no gain at 2.5; 3.0 linear 0.919 (best 3.0) |
| B3 | long multi-step schedule (3e-4 → 1e-4 → 3e-5) | partly running | I25-step15-4M (1e-4 from 1.5M); the 3e-5 tail later |
| B4 | calib-anneal cooldown (xcorp constant phase, calib decay; 20 / 40 %) | backlog | after lr is settled |
| B5 | very long run (10M rows) | pending decision | decide from the 4M curves (user) |
| B6 | prompt-side rows in the loss (0.1 / 0.3×) | backlog | `--prompt-weight` exists; ~1 card-h |
| B7 | bigger data plan | done | 10.04M rows, one epoch |
| B8 | self-generated responses (IQR-V4 8 bpw writer, bf16 targets) | backlog, deprioritised | user unsure it's worth it |
| B9 | 1−√ cooldown shape | backlog | — |
| B10 | WSD decay at all | done | nothing at 0.25 / 0.5M (1.000 / 1.002) nor at 4M on j217 (0.858 vs raw 0.853); training KL drops ~8 % with decay at 2M | buys nothing there (1.000 / 1.002 vs constant lr); longer budgets: from the 4M runs' raw vs decayed points |
| B11 | **new: skip decay branches in short screens** | new (process) | if B10 holds, raw snapshots replace decayed points ≤ 0.5M: ~20 % cheaper screens |

## C. What gets trained

| # | idea | status | result / note |
|---|---|---|---|
| C1 | norms (N), head input scales (H) | dead | N 1.007, H 1.027 at 2.5; dropped at 3.0 / 4.0 |
| C2 | LoRA, GDN a/b / conv, attention | dropped (user) | kernel / format changes |
| C3 | A8 in the loop | done | neutral; real int8 forward built (1.19×), default |
| C4 | code-QAD (per-weight / low-rank) | dead | ~80 % noise at 300k rows; low-rank directions don't transfer |
| C5 | code-changing healing (PV-tuning: shadow weights → re-encode) | dead-ish | same gradient-noise problem as C4 |
| C6 | per-output bias vectors | backlog | small format addition |
| C7 | table-QAD on packed CPTC | later (V4) | — |
| C9 | lc1 codebook at K ≤ 2 + table-QAD (format, kernels, quantizer header) | table-QAD dead (w224 null, 11-24) | value-remap probe: ≤ 0.5 %, within noise; lc1 alone stays at the bottom of the list |
| C8 | asym ⇄ heal / block coordinate descent (re-encode with healed scales, heal again) | later (V4) | always end on a heal |

## D. MSE family (local hidden-state targets)

| # | idea | status | result / note |
|---|---|---|---|
| D1 | front-half block-MSE stage → KL | **dead** (11-23) | MF25-2M 0.853, MD25-2M w/o blow-ups 0.878 = the plain-KL plateau (~0.85); was:  MD25 0.868 vs 0.883 at equal KL rows; MD25-2M, MF25 (1M stage) → 2M running |
| D2 | whole-model block-MSE stage → KL | **dead** (11-23) | MW25-2M 0.851 (1.011 [0.998, 1.033]); MSE alone 0.935 @ 1M but only 46 % of docs improve; was:  MW25-s1 → MW25-2M; layers 32–55 MSE signal as strong as the front (74 / 67 %) |
| D3 | joint KL + λ·MSE | **dead-ish** (11-23) | MJ25-2M 0.852; training KL tracks plain KL ±0.3 %; was:  0.976 [0.930, 1.001] @ 1M; MJ25-2M (joint 1M → KL 2M) running |
| D4 | λ 0.3× / 3× | running (last word on D3) | — |
| D5 | λ annealing (high → 0 by 1M) | dropped unless D4 surprises | gate: D1–D3 beat the 2M control |
| D6 | more teacher taps (> 1M rows) | dropped unless D4 surprises | ~80 GB / 1.5 h per 1M rows; front-only / scored-only to stretch |
| D7 | sensitivity-weighted block MSE (H_out-weighted) | dropped unless D4 surprises | gate as D5 |
| D8 | cross-block: MSE-only joint, CBQ windows, sequential | backlog | joint KL + MSE covers most of it |
| D9 | gradient-noise diagnostics | done | KL front half at chance (5 %), MSE 60–85 %; KL uses up the back half's signal (12 → 7.5 %) |

## G. Long context (new 11-22)

| # | idea | status | result / note |
|---|---|---|---|
| G1 | long-context judge golden-jlong (74 qad-pool docs, whole up to 247k, positions ≥ 8k) | done | 2.5 2M 0.876 [0.736, 0.948] (71/74 better); 3.0 ~1.00 (3 long wildchat chats uniformly 1.5–3× worse) |
| G2 | lift the training cap 8k → 32k (jlong docs excluded / 8k-capped) | next | gate: peak 30.56 GB, no slowdown; 25.5 % of calib's scored positions are beyond 8k |
| G3 | layer-split filled pipeline on 2 cards for 64k–256k training | brief next session (Opus medium) | 512-token micro-segments with the graph kept across halves |
| G4 | more calib rows: fit + search (~1.4–1.8M) and deep rows (~0.9M) | with G2 | |

## E. Blow-up documents

| # | idea | status | result / note |
|---|---|---|---|
| E1 | characterise + robust scoring | done | 4 of 166 docs; robust columns in the readout |
| E2 | fragility tests (released sizes, random noise, family reverts) | done | document fragility, not QAD (the unhealed 2.0 / 2.5 / 3.5 derail them too) |
| E3 | per-row loss cap | backlog, low | overlaps A14 / A15 |

## F. Final recipe and per size

| # | idea | status | note |
|---|---|---|---|
| F1 | recipe per size: 2.5 3e-4 constant long; 3.0 3e-4 + linear decay; 4.0 1e-4 | proposed | pending the 4M curves and D1–D3 at 2M |
| F2 | keep-best checkpoint + one verify read per finalist | planned | verify only for finalists |
| F3 | **new: is 4.0 worth QAD?** | new (decision) | j217: never > ~1 %; lr 1e-4 also derails by 2M (1.127); the ~5 % was on the retired j214. Per-doc clipping (A14) may change it |
| F4 | **new: other sizes** | next (user) | 2.0 / 2.25 after the 2.5 recipe settles: one 0.5M lr screen each, then one long run |
