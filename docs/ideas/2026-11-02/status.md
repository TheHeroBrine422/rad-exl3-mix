# Idea status: 2026-11-02 ranked list and faraway list vs. experiments

Audit as of 2026-11-05. Read-only on `docs/`; this file is the only write. Times are the LOG's own labels (CST = US Central).

**Inputs.** `ranked.md` (104 merged IDs, A1–J9; the K process items are not counted), `../2026-11-02-faraway/faraway.md` (63 IDs), `../../experiments/INDEX.md` (rows through #99), `../../experiments/LOG.md` (read in full), `../../experiments/BACKLOG.md`, `../../design/v3.md`, `../../ROADMAP.md`. `docs/experiments/archive/` is not in this checkout, so it was not read.

## How status was assigned

- **tested**: an INDEX or LOG experiment measured the idea. The verdict is the pass rule the LOG or ranked.md states: `pass`, `fail`, or `inconclusive` (the last when the log says so or the rule is not cleanly met). Tested results are kept to 12 words or fewer.
- **partly**: a named, related experiment covers a named part; the rest is untested.
- **queued**: a BACKLOG item, or a running or planned INDEX row, covers it. Where the match is by content and approximate, the row says so.
- **dropped**: an explicit decision against, in INDEX, LOG, BACKLOG or v3, or ranked §5's rejection. Where the only verdict is a lens-level kill with no experiment behind it, the row says `lens-only`.
- **open**: no trace.
- Not counted as tests: lens reports and faraway's own M-measurements (`ideas/README.md`: "Nothing here is a result"). They are cited where they kill an item, and marked `lens-only`.
- Matching is by content. Several ranked IDs share one experiment, and one experiment covers several IDs.

**Stale INDEX rows (not edited; the LOG is newer).** #33 reads `running`, but its two-sided arm is a dead end (LOG 11-01). #33-24 reads `running`, but it stopped at 1,920 sequences (LOG 11-05 10:40). #33-28 reads `planned`, but it has a provisional result, −2.8% (LOG 11-05 10:40). #33-29 reads `running`, but it is done with no new arm (LOG 11-05 01:20). E10 (LOG 11-03 08:48) and the two-pass prefill kernels (LOG 11-05 01:55) have no INDEX row.

## Ranked list (104)

### A. 27B diagonal H_out collection

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| A1 | Gen-masked sampled-label loss | tested: pass | #83, #83-02 | K3 0.971, K2 0.950, every seed; Σdy² K3 0.979 |
| A2 | Estimator check (‖x‖²dy², Σdy², Sketch B, leverage) | tested: pass | #33-18, #33-20, #33-29 | ‖x‖²dy² keeps 83% of Sketch B at K3; Σdy² best; leverage untested |
| A3 | Instrument the 27B run (extra sums, traces) | partly | #33-17, #33-24, #83 | halves, snapshots, gen mask, three estimators in one pass; per-sequence traces not added |
| A4 | Window composition (short docs, category mix) | partly | #33-21 | short qad docs neutral (K3 1.000); category-share check not run |
| A5 | Dose below 1K; early 512 decision | tested: inconclusive | #33-18, #33-23 | 512 passes the 80% rule (#33-18) but +2.2% worse at K3 (#33-23) |
| A6 | 27B pilot swap, GM/AM predictor, ldlq_2hess timing | partly | #33-28, #96 | 27B K2.5 swap −2.8% (provisional); predictor only as row-slab ceiling; no timing logged |
| A7 | 27B quant-noise floor (seed arm) | partly | #33-16, #33-28 | 0.8B seed SD ~1% (K3), ~2% (K2); 27B seed arm queued in the #33-28 chain |
| A8 | Fail-loudly asserts (file count, missing `--hout` key) | open | — | no assert added; #33-13 checks only the working path |
| A9 | Analytic SwiGLU diagonal (gate/up) | tested: fail | #84 | gate cosine median 0.86 (bar 0.95) |
| A10 | Gaussian or Rademacher √Fisher probe instead of one-hot label | open | — | — |
| A11 | bf16-GEMM 27B collector | tested: fail | #33-03, #33-14 | bf16 gradients unusable early (p10 row cos 0.02 at layer 20) |
| A12 | Operating-point H_out (collect on the quantized model) | open | — | H_out never collected on a quantized model (#87 covers the H_in analogue only) |
| A13 | Diagonal + rank-r Nyström sketch | partly | #33-12 | full-except-gate/up arm: K3 −9.9% vs diagonal −8.0%, K2 equal; rank-r not built |
| A14 | Full H_out for attention k/v | partly | #33-12 | only inside the mixed arm; not isolated |

### B. H_out inside the quantizer

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| B1 | H_out-aware `refit_scales` (undamped, multi-round) | tested: inconclusive | #85, #85-02 | seed-0 screen K3 0.993 (misses 0.99), K2 0.975; 3 seeds never logged |
| B2 | Tempered diagonal, output damping, block normalisation, floor | partly | #33-27, #33-28, #38 | block-normalised diagonal chosen (27B −2.8%); output damping and tempering not run |
| B3 | Attribution by family, depth, K | tested: pass | #82 | K4 −8.7%, K5 −7.5%; every family but writers (3%) carries it |
| B4 | Output-scale path, re-tested with Sketch-B diagonal | tested: fail | #33-06, #48 | diagonal output scales K2 +10–15%, K3 +4% worse (older diagonal) |
| B5 | Structure beyond diagonal (128-blocks, shrinkage, diag + low rank) | partly | #33-12 | mixed arm ≈ diagonal within noise; 128-block and shrinkage arms not run |
| B6 | Free output permutations (MLP interleave or sort) | open | — | — |
| B7 | Stored output row permutation or wider output Hadamard | open | — | — |
| B8 | Shuffle control (permute `hout_diag` within blocks) | partly | #33-16 | identity H_out (no diagonal) is worse than base; shuffle arm not run |
| B9 | Two-sided proxy in internal choices (σ_in, g_scale) | open | — | — |
| B10 | Block-diagonal fast path in `ldlq_2hess` | open | — | speed item only; not timed |
| B11 | Output-side LDLQ block order | open | — | — |

### C. Other quantizer changes

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| C1 | Activation-denoising LDLQ target | tested: fail | #88 | K3 1.031, K2 1.10: worse at both K |
| C2 | Post-LDLQ tile coordinate descent | tested: fail | #77, #79, #80 | −0.7…−0.8% at K2.5 on 16/16 tensors; bar 2% |
| C3 | Hessian-aware input scales (β exponent) | open | — | — |
| C4 | LDLQ block order | partly | #95-01 | act order inside 128-blocks: median 0.998–1.000; full orders not run |
| C5 | Quantized-prefix (GPTAQ) targets with control variate | tested: fail | #88-02 | all-150-tensor quantized-prefix target: K3 1.53×, K2 2.31× worse; blend not run |
| C6 | Intra-layer sequential (writers on quantized siblings) | tested: pass | #87, #87-02, #87-04, #87-07, #87-08 | 0.8B: K3 0.857, K2 0.824 vs base; K4/K5 hold; stacks on H_out |
| C7 | Scale ↔ code alternation | partly | #95-02 | post-trellis block refits hurt; "joint solve" −0.5…−1.5%, not described in the log |
| C8 | LDLQ-aware g_scale search | open | — | blocked behind #39 (gain scale, never logged) |
| C9 | KL-optimal svh multiplier per level and family | tested: pass | #85, #85-02 | α* K3 0.973, K2 0.911; same on seed 1; swj stack inconclusive |
| C10 | SwiGLU-joint gate/up scale fit | tested: pass | #85-03 | K3 0.969–0.972, K2 0.910–0.914 vs edy, 3 seeds |
| C11 | Per-block rotated-domain scales inside LDLQ; tail codebook | partly | #40, #95-02 | per-group scales dead at low K; post-trellis (row,128) refit worse; in-LDLQ untested |
| C12 | Null-space ridge shrink before the trellis | open | — | — |
| C13 | Gen-weighted H_in with bounded weights | tested: inconclusive | #86 | K3 0.998 (no effect); K2 0.976 every seed, bar 0.97 missed |
| C14 | Tempered or blended gradient-weighted H_in | partly | #33-07 | untempered gradient weighting fails (K2 +44%, K3 +31%); blend not run |
| C15 | H_in factorial diagnosis | partly | #33-10 | H_in swap was the loss (+20%); token, gen, domain, ESS arms not run |

### D. Codebook and trellis format

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| D1 | Per-tile codebook salt | open | — | — |
| D2 | Multi-tile rings / chained trellis | partly | #57 | free-vs-tail-biting gap 4–6%; 16 wrap candidates recover 4–12%; rings not built |
| D3 | Gaussianised mul1 codebook (cubic, i8c) | tested: pass | #78, #78-02, #78-04 | K3 −1.7%, K2 −2.7%, K4 −2.3%; null on top of H_out |
| D4 | Longer trellis state (L18 / L20) | open | — | — |
| D5 | Edge-emitting values from a (16+K)-bit window | open | — | — |
| D6 | Row-slab K driven by diagonal H_out | tested: fail | #96, #29-02 | ceiling 0.036 bit vs 0.10 bar; per-head remainder ~0.04 bpw |
| D7 | Column-split K per k-block | open | — | `lens-only` (faraway M4) puts it near 0.01 bit; no INDEX row |
| D8 | Spike spreading (Householder, 512-pt, two-level Hadamard) | partly | #47 | activation-side random R1 6.6× worse at A8-g128; weight side untested |
| D9 | Within-tile weighted Viterbi | dropped | ranked §5 | judged inert: Γ_b 0.0001–0.0007 bit (`lens-only`, CurveTQ) |

### E. Head and embedding

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| E1 | Low-budget slab heads in knap4 | partly | #34 | headsim budget 3.0: 0.0060 vs uniform K3 0.0079; whole-model swap not run (#34-02 armed) |
| E2 | Two-sided head with the exact Fisher diagonal | partly | #34 | Fisher-ranked slab baseline measured; two-sided within-block feedback not built |
| E3 | Head healing on dumped head inputs | open | — | — |
| E4 | Head statistics from gen rows only | partly | #34 | Fisher ranking stable across pools (top-set overlap 0.93–0.98); gen-only arm not run |
| E5 | Softmax shift t-grid / Fisher-weighted shift | partly | #34 | uniform mean-row shift: head KL −30%, exact; Fisher-weighted shift not run |
| E6 | Low-rank head residual in the C^p metric | open | — | — |
| E7 | Embedding with a downstream column metric | open | — | #35 used H = I and token frequency only |

### F. Allocation, pricing, search

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| F1 | Re-price H_out quants (oursH family) | queued | BACKLOG "Σdy² as the knapsack price" (approximate match) | pricing change waits on the 27B diagonal, which now exists |
| F2 | Two-sided proxy for all options → refit kfit2 | queued | same BACKLOG item (approximate match) | — |
| F3 | Fisher-predicted ΔKL prior | open | — | — |
| F4 | Re-anchor 2.5 sensitivities (fractional-factorial swaps) | open | — | #14 (single-tensor swaps too noisy) motivates it; design not run |
| F5 | Pair-interaction terms in the layer DP | dropped | #16 | Shapley-style interactions skipped: ~15–18 h per 8B model |
| F6 | Lazy-greedy measured-KL descent as a stream kind | partly | #70 | V2 local search and targeted swaps mattered; lazy-greedy kind not run |
| F7 | Cap moves per accepted step; recenter | open | — | — |
| F8 | Posterior-mean selection; Thresholdout | partly | #72 | fit-slice picks ~3% optimistic; posterior rule not run |
| F9 | Separate K for gate vs up | open | — | — |
| F10 | GDN decay half-life as a kfit2 feature | open | — | — |
| F11 | Tail-aware K1 pricing; per-layer K1 caps | partly | #44 | K1 family factor: CV log-RMSE 0.870 → 0.845; caps not run |
| F12 | ⅛-step trunk levels K2.125–2.875 | partly | #36, #90 | ⅛ steps K3–5 ≤0.8% at 4.0 (#36); finer K1.1–1.9 menu −1.3% at 2.25 (#90) |
| F13 | Category- or tail-weighted aggregation | open | — | — |

### G. Evaluation statistics and throughput

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| G1 | Seed-aware z; retire g08-search1; new 0.8B slices | partly | #33-15, #33-16, #93 | seed-noise rule adopted; new 0.8B verify golden planned (#93); retirement not logged |
| G2 | GPU-side top-512 KL, head only on scored rows | partly | #8 | card-1 slowdown traced to the full logits copy; fix not built |
| G3 | Successive halving / early rejection | open | — | — |
| G4 | Prefix cache / layer-major eval / sibling fan-out | open | — | — |
| G5 | Pipelined candidate loading | open | — | — |
| G6 | CUPED / stratified search slices | partly | #58 entry (capstudy) | 2k search cap: ~27% less noise per GPU-second; CUPED not tried |
| G7 | Last-layer suffix simulator | open | — | — |
| G8 | A/A cross-card byte identity | open | — | — |

### H. Healing and QAD

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| H1 | 0.8B healing gates (oracle, depth, gradient cosine) | tested: pass | #89 | oracle ceiling 62% → build S27 (user deferred); last-quarter claim not reproduced |
| H2 | Scale + norm KL healing, frozen codes (S27 stack) | partly | #89, #89-02, #89-03, #89-04 | 0.8B scale healing: K2 0.721, K3 0.886 vs stack; norms ≤0.2%; 27B deferred |
| H3 | Forward-only asymmetric sequential scale refit | partly | #85-03, #87 | joint SwiGLU refit and asym writers pass; sequential scale refit not built |
| H4 | Level → mix transfer of healing | partly | #89 | mixed-K arms logged (0.65–0.74); the ≥70% transfer rule is not evaluated |
| H5 | Depth curriculum, per-layer LR, fp32 early backward | partly | #89 | all-layer healing K2 0.619 beats last-quarter 0.845; curriculum not run |
| H6 | GDN small parameters as compensators | partly | #89-02 | norm gains ≤0.2%; GDN conv / A_log / dt_bias not run |
| H7 | Tail-KL window selection; importance-sampled batches | open | — | cost is "in H2" (depends on healing) |
| H8 | Re-rank finalists after healing | open | — | depends on healing (deferred by user) |
| H9 | Code refresh after healing (re-encode, STE) | open | — | depends on H2 |
| H10 | Small residual adapters / spike LoRA | open | — | depends on H2 |
| H11 | Residual-writer biases (closed form) | open | — | — |
| H12 | Heal with A8 on; tie sibling suh first | partly | #42 | sibling suh differ by 1.5–9% (median); tie cost unmeasured |
| H13 | Long windows / TBPTT and position guard | open | — | — |

### I. Long context, GDN, numerics

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| I1 | Mine the 17 long-trace evals (family × position) | partly | #52 | position-bin KLD ratios flat across quants (2 docs); family × position not mined |
| I2 | Long-position group-sensitivity probe; position-weighted objective | open | — | #52 is the nearest data (2 docs); probe not run |
| I3 | Engine floor and GDN state precision at long positions | partly | #33-26, #60-01 | chunking floor ~0.0007; bf16 vs fp32 KL 3.75e-4, flat; GDN state not isolated |
| I4 | 0.8B long-context H_out gain | open | — | — |
| I5 | fp64 LDL audit | open | — | — |

### J. Kernels and speed

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| J1 | Integer mul1i8 decode | partly | #59, #31 | decode bit-exact at all 64 rates; int8 path timed end to end only |
| J2 | Split-K at decode for N=5120 | open | — | #28 saw prefill under-occupancy at M=512; decode not timed |
| J3 | Fused rotation + int8 prep; gated epilogue | queued | BACKLOG (kernels; IQR section) | fused activation-quant pass listed, not started |
| J4 | T2 chunk sized to the Infinity Cache | open | — | 192 MiB current value (`lens-only` code check); no sweep |
| J5 | Cheaper g128 fold | partly | #28 | group-scale cost baseline −12…−29%; cheaper fold not tried |
| J6 | Decode profile → overhead fusions | open | — | — |
| J7 | DFlash2 port and quantized drafter; MTP | queued | BACKLOG (quantized drafter); v3 §8 | bf16 bundling is in the V3 plan; quantized drafter low priority |
| J8 | Real prefill time table (prep, wdec, fold, epilogue) | partly | #45, #28 | GEMM-core table only; decode placeholder; two-pass timing queued |
| J9 | Tie sibling suh at requant | partly | #42 | siblings do not share suh; tie cost not measured |

## Faraway list (63; D1 answered and not counted)

### A. Cross-tensor prediction inside a layer

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-A1 | Down from up + gate, closed-loop prediction | partly | #81, #81-02, #81-03, #81-04 | CPU proxy ug/D 0.83 at K2.5; control confirms; 0.8B KL running |
| FA-A2 | Online-prediction serving (transposed GEMMs) | queued | BACKLOG (A1 item, kernel step) | kernel cost (ntbench) not run |
| FA-A3 | GDN out_proj from in_proj v | queued | BACKLOG (riders) | — |
| FA-A4 | o_proj from v and q-gate | queued | BACKLOG (riders) | — |
| FA-A5 | Shared neuron direction coded once | open | — | symmetric variant of A1 |
| FA-A6 | q↔k row prediction | queued | BACKLOG (A6/A7 census) | — |
| FA-A7 | GDN z ↔ out_proj / v | queued | BACKLOG (A6/A7 census) | — |
| FA-A8 | up predicted from gate | dropped | `lens-only` (faraway M7) | cosine 0.06–0.12 |
| FA-A9 | Head rows predicted from embedding rows | dropped | `lens-only` (faraway M8) | median cosine 0.012 |
| FA-A10 | Prediction-aware knapsack pricing | queued | BACKLOG (27B queue, option list) | enabler for A1 |
| FA-A11 | Predictor-aware rate for up | dropped | faraway (≈0, "not retried") | `lens-only` estimate |

### B. Transform coding with shared bases

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-B1 | Global residual KLT + variance-sorted blocks | partly | #47 | random-R1 activation test 6.6× worse; data-derived basis untested |
| FA-B2 | Depth-banded residual re-rotation | open | — | depends on B1 |
| FA-B3 | Learned per-tensor structured KLT | open | — | ranked §5 rejects learned rotations under uniform rate, not this objective |
| FA-B4 | Per-block K as column-group sub-tensors | open | — | format; depends on B1 / B3 |
| FA-B5 | Embedding and head in own KLT basis | open | — | pays only through B1 (`lens-only`) |
| FA-B6 | SliceGPT-style residual truncation | dropped | `lens-only` (faraway) | no coordinate earns zero bits |
| FA-B7 | Per-tensor own basis with stored V | dropped | `lens-only` (faraway) | storage 0.6–1.2 bpw |
| FA-B8 | Rank-r subspace split | dropped | `lens-only` (faraway) | 21 MB per tensor |
| FA-B9 | Per-128-block K in Hadamard basis | dropped | `lens-only` (faraway M4) | 0.009–0.013 bit; faraway "≈KILLED" |
| FA-B10 | Per-head V/O rotation for o_proj | open | — | o_proj 1.9% of bytes |

### C. Headroom audit and implementation gaps

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-C1 | Gap-to-ideal audit, 496 tensors | open | — | `lens-only` measurements cover 6 layers |
| FA-C2 | Early-layer attention / GDN-input repair | partly | #38, #96 | attention-input damping (#38); per-head K ceiling 0.27–0.33 on ssm_qkv / attn_qg (#96) |
| FA-C3 | Ideal-model pricing for new formats | open | — | enabler |
| FA-C4 | Non-Gaussian / rotation-free / entropy-coded trunk codecs | dropped | #29 | integer codes ×2–3.5 and Lloyd LUT lose to the trellis |
| FA-C5 | ECTCQ at rest, transcoded at load | open | — | storage-only: counts only once it runs from the stored form (AGENTS.md) |

### D. Decoupling storage from compute

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-D1 | Rules question (container vs runtime) | answered (not counted) | LOG 11-02 17:05 | challenge author: VRAM target; load expansion only as a stopgap |
| FA-D2 | Load-time reconstruct and re-encode | partly | #81-04 | 0.8B KL run uses load-time expansion; re-encode error not measured |
| FA-D3 | Entropy-coded or KLT-domain embedding rows | open | — | storage-only (AGENTS.md rule) |
| FA-D4 | Runtime dense int8 for prefill | partly | LOG 11-05 01:55 (no INDEX row) | host tests pass; device timing queued |

### E. Model-level restructuring

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-E1 | Cross-layer delta / motion-compensated coding | dropped | `lens-only` (faraway M1) | median cosine 0.10–0.16 across adjacent layers |
| FA-E2 | Looped / recursive conversion | dropped | `lens-only` (faraway M1) | plus training cost |
| FA-E3 | Layer merging | dropped | `lens-only` (faraway M1) | — |
| FA-E4 | Layer drop + linear adapter | open | — | #21 single-layer probes give priors only |
| FA-E5 | MLP width pruning + higher K | open | — | — |
| FA-E6 | GDN value-head and attention-head pruning | open | — | — |
| FA-E7 | Within-layer neuron VQ | open | — | census not run (kill if median < 0.2) |
| FA-E8 | Neuron-type split | open | — | subsumed by A1 |
| FA-E9 | MoE-ification (sorted sub-MLPs) | partly | #96 | 128-row slab ceiling 0.036 bit; cluster variant not run |
| FA-E10 | Dead-unit census and exact removal | tested: fail | LOG 11-03 08:48 (no INDEX row) | no down_proj unit below 1e-2 × median; bottom 10% holds 6.7% |

### F. Residual-stream repair

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-F1 | Closed-form low-rank drift correction | open | — | — |
| FA-F2 | Diagonal affine at layer boundaries | queued | BACKLOG (F2 item) | 0.8B, ~2 h |
| FA-F3 | Multi-layer readout corrector | open | — | — |
| FA-F4 | Exact last-layer output Hessian via head Fisher | queued | BACKLOG (F4 item) | — |

### G. Objectives and data

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-G1 | Per-document adaptivity | dropped | faraway (rejected: bytes fixed) | — |
| FA-G2 | Test-time dynamic evaluation | open | — | 0.8B 30-min test not run |
| FA-G3 | bf16 KV / state for a chat preamble | dropped | faraway (rejected unless approved) | AGENTS.md: bf16 KV for all KLD |
| FA-G4 | Pre-quantization conditioning fine-tune | open | — | rented GPUs and a data-rules check |
| FA-G5 | Tail-position-weighted H_in | open | — | lens says near-duplicate of A1 |

### H. Rounding as inference

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-H1 | Soft-decision BCJR + VAMP | open | — | ranked C2 (hard tile CD, −0.8%) is the nearest test |
| FA-H2 | List-Viterbi with cross-tile beam | open | — | — |
| FA-H3 | Annealed stochastic-LDLQ ensembles | partly | #41 | best-of-N seed selection null (spread 0.5%); stochastic-rounding variant not run |
| FA-H4 | LLL before Babai / GPTQ | dropped | `lens-only` (faraway) | trellis codebook is not a lattice |

### J. Symmetries and reparameterisations

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-J1 | SwiGLU up/down scale symmetry | dropped | `lens-only` (faraway: invariance) | #85-03 is a joint refit, not this invariance test |
| FA-J2 | Residual-stream global scale symmetry | dropped | `lens-only` (faraway: inert) | — |
| FA-J3 | Free permutations for A1 | open | — | folds into A1 |

### K. Crazy but cheap

| ID | Idea | Status | Evidence | Result |
|---|---|---|---|---|
| FA-K1 | Implicit neural representation / hypernetwork | open | — | lens expects hopeless; 30-min test not run |
| FA-K2 | Noise filling for sub-1 components | open | — | #90 sub-1 curves exist; dither arm not run |
| FA-K3 | Spectral shrinkage | open | — | — |
| FA-K4 | Seed/LFSR and dynamic-system codes | dropped | faraway (paper: worse than AWQ at 4 bpw) | — |
| FA-K5 | Logit mixture with a small bf16 model | dropped | faraway (0.8B bf16 costs 0.48 bpw) | — |
| FA-K6 | Self-adjoint MLP neurons | open | — | count not done; tail of A1 |
| FA-K7 | Per-block transform side bit | open | — | subsumed by C2 |

## Experiments that map to no ranked ID

Mostly newer than the 11-02 ranked list. Listed for the lead to map if wanted.

- #90, #90-02, #94: sub-1 K pricing and i4 vs i8 (BACKLOG). #94 is invalid (grid cap); rerun pending.
- #91, #98: self-calibration on own sessions (BACKLOG; ROADMAP). #91 phase 1 shows no H_out gain.
- #92: REAL-Q on the 0.8B (planned; the ranked list cites REAL-Q only as a background reference).
- #97: embedding menu at both ends (done; all levels on the hull).
- #99, #34-02: 8.0 floor split and whole-model slab-head check (armed, not run).
- #46-02, #46-03: A4 on the MLP (fail) and per-tensor A4 option (planned).

## Summary

### Ranked list (104)

Overall: **tested 18** (pass 8, fail 7, inconclusive 3) · **partly 39** · **queued 4** · **dropped 2** · **open 41** = 104.

| Theme | Items | Tested: pass | Tested: fail | Tested: inconcl. | Partly | Queued | Dropped | Open |
|---|---|---|---|---|---|---|---|---|
| A. 27B collection | 14 | 2 | 2 | 1 | 6 | 0 | 0 | 3 |
| B. H_out in the quantizer | 11 | 1 | 1 | 1 | 3 | 0 | 0 | 5 |
| C. Other quantizer | 15 | 3 | 3 | 1 | 5 | 0 | 0 | 3 |
| D. Codebook and trellis | 9 | 1 | 1 | 0 | 2 | 0 | 1 | 4 |
| E. Head and embedding | 7 | 0 | 0 | 0 | 4 | 0 | 0 | 3 |
| F. Allocation and search | 13 | 0 | 0 | 0 | 4 | 2 | 1 | 6 |
| G. Eval statistics, throughput | 8 | 0 | 0 | 0 | 3 | 0 | 0 | 5 |
| H. Healing and QAD | 13 | 1 | 0 | 0 | 6 | 0 | 0 | 6 |
| I. Long context, numerics | 5 | 0 | 0 | 0 | 2 | 0 | 0 | 3 |
| J. Kernels and speed | 9 | 0 | 0 | 0 | 4 | 2 | 0 | 3 |
| **Total** | **104** | **8** | **7** | **3** | **39** | **4** | **2** | **41** |

### Faraway list (63)

Overall: **tested 1** (fail) · **partly 7** · **queued 8** · **dropped 18** · **open 28** · **answered 1** (D1, not counted) = 63.

| Section | Items | Tested | Partly | Queued | Dropped | Open | Not counted |
|---|---|---|---|---|---|---|---|
| A. Cross-tensor prediction | 11 | 0 | 1 | 6 | 3 | 1 | 0 |
| B. Shared-basis transform coding | 10 | 0 | 1 | 0 | 4 | 5 | 0 |
| C. Headroom and gaps | 5 | 0 | 1 | 0 | 1 | 3 | 0 |
| D. Storage vs compute | 4 | 0 | 2 | 0 | 0 | 1 | 1 |
| E. Model restructuring | 10 | 1 | 1 | 0 | 3 | 5 | 0 |
| F. Residual-stream repair | 4 | 0 | 0 | 2 | 0 | 2 | 0 |
| G. Objectives and data | 5 | 0 | 0 | 0 | 2 | 3 | 0 |
| H. Rounding as inference | 4 | 0 | 1 | 0 | 1 | 2 | 0 |
| J. Symmetries | 3 | 0 | 0 | 0 | 2 | 1 | 0 |
| K. Crazy but cheap | 7 | 0 | 0 | 0 | 2 | 5 | 0 |
| **Total** | **63** | **1** | **7** | **8** | **18** | **28** | **1** |

### Ten most promising open ideas (ranked list, E column)

Method: open ranked IDs that carry a KL gain in the E column, ordered by the midpoint of the E range (most negative first), ties broken by P. The E and P values are ranked.md's own estimates, not measurements. Excluded: speed or throughput items with no KL column entry (G3–G5, G7, G8, J4, J6) and items with no gain (A8, A10, B10, H13, I4, I5).

| # | ID | Idea | E @2.5 | P | Dependency and cheapest first step |
|---|---|---|---|---|---|
| 1 | F4 | Re-anchor 2.5 sensitivities with fractional-factorial swaps | −1 to −3% | 0.4 | none; ~1.5 card-h on the GPU box (32 evals) |
| 2 | H9 | Code refresh after healing (re-encode, STE) | −1 to −3% beyond H2 | 0.2 | **depends on H2 (healing, deferred)** |
| 3 | C3 | Hessian-aware input scales, β ∈ {0, ¼, ½} | 0 to −3% | 0.25 | none; 0.8B, 3 seeds, fixed codes (~1.5 h) |
| 4 | I2 | Long-position group-sensitivity probe; position-weighted objective | 0 to −3% | 0.25 | none; ~1 card-h probe; #52 is the nearest data |
| 5 | E3 | Head healing on dumped head inputs (+ τ, final-norm gain) | −0.5 to −2% | 0.5 | none; one dump eval (~1 h) plus minutes on GPU |
| 6 | F3 | Multi-layer readout corrector for logits | −0.5 to −2% | 0.3 | none; needs L60–63 state dumps |
| 7 | H8 | Re-rank finalists after healing | 0 to −2% | 0.4 | **depends on healing (deferred)** |
| 8 | D1 | Per-tile codebook salt (1–2 bits per tile) | −0.5 to −1.5% | 0.35 | none; CPU day on real LDLQ tiles |
| 9 | D4 | Longer trellis state L18 / L20 | −0.5 to −1.5% | 0.3 | none; quantizer time ×4–16 |
| 10 | H10 | Small residual adapters / spike LoRA, priced in bytes | 0 to −2% net | 0.25 | **depends on H2 (deferred)** |

Three of the ten (H8, H9, H10) wait on the healing work the user deferred. Without them the list runs F4, C3, I2, E3, F3, D1, D4, A12, D7, H11. Places 6–13 all lie between midpoints −0.9 and −1.25, so the tail of the list is sensitive to the midpoint choice; H7 (−0.75, P 0.35) and C8 (−0.75, P 0.3, blocked behind #39) are the next candidates and both also wait on healing or #39.

### Notes for the lead

- K1 (write the 27B pass rule first) is done: `v3/yaqa/H27.md`, written before data (#33-28). K2 (update v3 §2, §6b, §7) is partly done: §7 and the §2 YAQA row are updated (11-05), but §6b has only a 2026-11-04 annotation and its body still says output weighting "hurt in every form we tried" (v3.md line 433). K3 (ledger entries for 27B qad use and healing) and K4 (requant scope) were not checked; the ledger is outside the evidence set.
- Lens-only kills (faraway M1, M4, M6, M7, M8) rest on the faraway agent's own CPU measurements, not on INDEX experiments. Re-check before treating them as settled.
- The 27B swap test (#33-28, −2.8%) is the only 27B result for H_out. Its seed arm is still queued, so A7 and the 27B confirmation of C6 and C9/C10 are open.
