# Experiment index

One row per experiment; the details are in `LOG.md` (search for the number, e.g. `#58-03`). Status words and numbering:
`README.md`. Bulk-numbered on 2026-11-02 from the old trackers (#1–#62, roughly by date) and `docs/design/v2.md` (V2
wins, #63–#73). Positive and negative results both get rows: the status column says which.

`started` / `updated` = dates of the first and the latest LOG entry (MM-DD, 2026); `≤10-30` = V2 era (#1–#24, #63–#73),
no exact dates. `result` = the answer in one line (or what is being measured, while running).

**runtime** (user, 2026-11-03): what the result costs at inference, beyond picking a different K. `none` = quantizer /
allocation / healing only: same format and kernels, the codes or scales are just better · `~0` = touched but
negligible (embedding lookup) · `P` = prefill compute (`gemm_us`, scored by the board) · `D` = decode (not scored,
but the user cares and may test it) · `P+D` = both (`?` = expected, not yet timed) · `speed` = the experiment is
itself a speed measurement · `n/a` = no model change (eval, data, tooling).


| # | title | area | runtime | status | started | updated | result |
|---|---|---|---|---|---|---|---|
| 1 | Two quantizer processes per card | quant speed | n/a | negative | ≤10-30 | ≤10-30 | +14% throughput only (GPU already at 100%), fragmentation risk: one process per card |
| 2 | Bigger tile batches in the quantizer | quant speed | n/a | null | ≤10-30 | ≤10-30 | trellis cost per tile is flat from 64 to 8,704 tiles per launch |
| 3 | Sharing work across K values in one run | quant speed | n/a | dropped | ≤10-30 | ≤10-30 | the shared part (Hessian prep, LDL, Hadamard) is ≤ 15% of a run: not worth restructuring |
| 4 | K7's slow trellis kernel | quant speed | n/a | dropped | ≤10-30 | ≤10-30 | not fixed (needed an exllamav3 rebuild); qtf2 covers K7 (#37) |
| 5 | Head quantization OOM at K5/K6 | quant speed | n/a | done | ≤10-30 | ≤10-30 | fixed: `refit_scales` rewritten in place, chunked; a head quant still needs ~22 GB host RAM |
| 6 | Bigger prefill steps in the eval | eval speed | n/a | null | ≤10-30 | ≤10-30 | 4,096-token chunks take the same 118 s for the same KLD: the EXL3 GEMMs are the bottleneck (92% of GPU time) |
| 7 | rocprofv3 on radiance | eval speed | n/a | dropped | ≤10-30 | ≤10-30 | unusable (copy tracing breaks the load, kernel tracing faults the replay); use radiance's `--profile-ops` |
| 8 | Card 1 evals ~25% slower (x4 link) | eval speed | n/a | done | ≤10-30 | ≤10-30 | cause: the full-vocab f32 logits copy over card 1's x4 link; an async copy would fix it (not done) |
| 9 | TP2 for whole-corpus runs | eval speed | n/a | negative | ≤10-30 | ≤10-30 | slow over card 1's x4 link: TP2 only for documents that don't fit one card |
| 10 | First knapsack: band sensitivities, linear model, Lagrangian | allocation | none | superseded | ≤10-30 | ≤10-30 | by kfit2 / knap4 (#45) |
| 11 | Symmetric depth-compounding term | allocation | none | negative | ≤10-30 | ≤10-30 | a global link on the total, not a depth effect; replaced by kfit2's directional model (CV log-RMSE 0.28 → 0.11–0.13) |
| 12 | Lagrangian multiple-choice-knapsack sweep | allocation | none | superseded | ≤10-30 | ≤10-30 | exact only on hull points, lands below targets, can't take the directional term: replaced by the exact DP (#69) |
| 13 | Treating all quantizer families alike | allocation | none | negative | ≤10-30 | ≤10-30 | turboderp's plain exports cost 1.15–1.23× the KLD of SC ones per unit of proxy error: one factor per family |
| 14 | Measuring single-tensor swaps end to end | allocation | none | negative | ≤10-30 | ≤10-30 | paired SE ~1.5% vs a single tensor's 0.1–0.5% effect: use group swaps / whole mixes instead |
| 15 | Noise-injection sensitivity calibration | allocation | none | negative | ≤10-30 | ≤10-30 | not cheaper than real swaps (0.2 s reflink mixes) and needs a bf16 forward that doesn't fit one card |
| 16 | Shapley-value interactions (IMPQ, SLQ) | allocation | none | dropped | ≤10-30 | ≤10-30 | literature: interactions ~1% for ~15–18 h per 8B model; skipped |
| 17 | RCO (Riemannian constrained optimisation) | allocation | none | dropped | ≤10-30 | ≤10-30 | needs a full differentiable 27B forward + backward, 800–3,200 passes per size, for 0–4% over measured search: skipped |
| 18 | Gradient boosting / black-box surrogates | allocation | none | negative | ≤10-30 | ≤10-30 | unusable: the exact solver needs a per-tensor-decomposable model; black-box BO needs ~10k evals |
| 19 | Per-layer edge offsets in kfit2 (KFIT_EDGE) | allocation | none | negative | ≤10-30 | 10-31 | no ranking gain; code kept behind `KFIT_EDGE` |
| 20 | Residual re-ranker on top of kfit2 | allocation | none | negative | ≤10-30 | 10-31 | no better than kfit2 (ridge 0.157, XGBoost 0.127 vs 0.118 log-RMSE); retry with V3's larger dataset |
| 21 | Layer 0 as a sensitivity spike | allocation | none | null | ≤10-30 | ≤10-30 | layer 0 ffn_down at K2 is nearly free (+0.00006); the spike is the last layer (L63 ffn_gate_up K2 +0.0065) |
| 22 | GPTQ damping for our quants | quantizer | none | superseded | ≤10-30 | 10-31 | deferred to the V3 requant; by #38 |
| 23 | 2-bit Lloyd embedding | embedding | ~0 | negative | ≤10-30 | ≤10-30 | clearly worse than 3-bit at normal sizes; offered only at the 1.2–1.5 bpw research floor |
| 24 | Head above K6 | head | none | null | ≤10-30 | ≤10-30 | K7 / K8 heads have 5–16× less proxy error than K6, but the KLD gain is within noise |
| 25 | Hardware facts: iu8/iu4 peaks, scale-group cost, no fp4 (alias **E0**) | kernels | speed | done | 10-30 | 10-30 | compute is integer (iu8 ~323 TOPS, iu4), no fp4 / fp8; K-axis scale groups are expensive → per-token / per-channel scales |
| 26 | Activation-quant cost by fake-quant hook (alias **E1/E1b**) | activations | P | done | 10-30 | 10-31 | A8 g128 +0.00005 at K6: fine; A4 all linears +0.149: no |
| 26-01 | A8 variants: per-token, g128, g32, per family | activations | P | done | 10-30 | 10-30 | A8 at K6: g128 +0.00005, g256–g1024 +0.00015–0.00018, per-token +0.00047 |
| 26-02 | A4 per family / depth band, g128, clipping | activations | P | done | 10-30 | 10-30 | A4 g128 is 5× better than per-row but still costs ~+0.023–0.025 per big family at K6 → #46 |
| 27 | Per-tensor activation-quant proxy, 337 tensors × 32 configs (alias **E1c**, AQ-proxy) | activations | P | done | 10-30 | 10-31 | per-tensor activation-quant costs for 337 tensors × 32 configs, calibrated to ±30%: the knapsack's activation costs |
| 28 | int8 / int4 GEMM at real tiling, `igemm.hip` (alias **E2**) | kernels | speed | done | 10-30 | 10-30 | iu8 ~255 TOPS, iu4 ~490 (1.9× / 3.7× f16) |
| 29 | Weight codes, quality per bit on the Hessian proxy (alias **E3**) | weight codes | P | done | 10-30 | 10-31 | integer codes ×2–3.5 the proxy error of the trellis: the trellis stays |
| 29-01 | Trellis decoding to int8 / int4 codebooks (alias **E3b**, tsweep2) | weight codes | P | done | 10-30 | 10-31 | → i8 codebook (#43) |
| 29-02 | Row-slab K ceiling (alias **E3c**) | weight codes | P+D | done | 10-30 | 10-31 | ceiling ~0.05 bit on the big tensors at 128-row slabs; the outlier tensors carry most of it: low priority |
| 30 | New-format solver knap3 + 12-mix validation (aliases **K3**, **V0**) | allocation | none | superseded | 10-30 | 10-31 | by knap4 (#45) |
| 31 | Real int8 prefill kernels (alias **E4**); NFQ int8 crash | kernels | speed | blocked | 10-30 | 11-01 | crash blocks A8 timing |
| 31-01 | End-to-end prototype at ~4 bpw (alias **E4b**) | kernels | speed | planned | 10-30 | 10-30 | after #31 |
| 32 | Vocab-tiered embedding and head | embedding | D | negative | 10-31 | 10-31 | tiered embedding KLD 0.0140 vs uniform e4 0.0081; rare-token rows matter more than frequency predicts: dropped |
| 33 | YAQA output-side Hessians (two-sided LDLQ) | quantizer | none | running | 10-31 | 11-02 | full YAQA worse; calib H_in + Sketch B H_out on our settings: 0.8B K3 −9.7%, K2 −5.6% vs base (#33-11); diagonal H_out enough, ~4K seqs (#33-12); 27B collection next |
| 33-01 | Gate 1: TP2 collection throughput | quantizer | none | done | 10-31 | 10-31 | 5 s per 2048-token sequence |
| 33-02 | Gate 2: bf16 backward, sampled rows, 200 seqs | quantizer | none | negative | 10-31 | 10-31 | +23% (K2) / +28% (K3) |
| 33-03 | bf16 vs fp32 backward noise | quantizer | none | done | 10-31 | 10-31 | bf16 gradients unusable in early layers |
| 33-04 | Gate 2b: fp32 backward, sampled rows | quantizer | none | negative | 10-31 | 10-31 | +17% / +19%; trend toward one-sided |
| 33-05 | Gate 2c: full-token fp32 H_out | quantizer | none | negative | 10-31 | 10-31 | +30% / +21% |
| 33-06 | Gate 2d: diagonal output scales (alias **D1** = its confound check) | quantizer | none | negative | 10-31 | 11-01 | K2 +10–15%, K3 +4% |
| 33-07 | Gate 2e: gradient-weighted H_in | quantizer | none | negative | 11-01 | 11-01 | K2 +44%, K3 +31% |
| 33-08 | Sketch B (paper estimator) on the 0.8B proxy, 1K–8K sequences | quantizer | none | negative | 11-02 | 11-02 | 8K: K3 +7%, K2 +15% vs paper LDLQ (+14% / +20% vs base); 16K stopped |
| 33-09 | Sketch B factors: structure and noise (CPU) | quantizer | none | done | 11-02 | 11-02 | H_out diag precise; full H_out noisy early / residual writers; YAQA arms also swapped H_in |
| 33-10 | Factorial: Sketch B H_in vs H_out, 0.8B K3 | quantizer | none | done | 11-02 | 11-02 | H_in swap +20%; calib H_in + H_out −9.9% vs ldlq, −4.3% vs base (z −5.4); diag only −1.7% vs base |
| 33-11 | H_out at K2 and stacked on base's settings, 0.8B | quantizer | none | done | 11-02 | 11-02 | bhout vs base: K3 −9.7% (z −12.4), K2 −5.6% (z −6.3); switch bar met on 0.8B |
| 33-12 | H_out sizing: mixed / diagonal / dose, 0.8B | quantizer | none | done | 11-02 | 11-02 | diagonal only: K3 −8.0%, K2 −8.0% vs base; full needs ~4K seqs |
| 33-13 | 27B diagonal H_out collection + 0.8B diagonal dose | quantizer | none | running | 11-02 | 11-02 | 0.8B: 1K seqs keep 85% (K3) / 99% (K2) of the gain; 27B run at 1K pending (both cards, ~8.5 h) |
| 33-14 | Split-precision bf16 GEMMs for the fp32 collector | quantizer | none | negative | 11-02 | 11-02 | 0.8B: x2 diag error 1.2–5.5e-3 median (rule ≤ 1e-3), 10× better than bf16; collector stays fp32 |
| 33-15 | Split precision judged by KL, 0.8B | quantizer | none | negative | 11-02 | 11-02 | x2 K2 +2.8%, but bf16 did better: KL test shows ~±2–3% quant-to-quant noise, not precision; collector stays fp32 |
| 33-16 | Seed replicates + identity H_out, 0.8B | quantizer | none | done | 11-02 | 11-02 | 3 seeds: diag H_out K3 −8.1%, K2 −5.4% vs base; seed SD ~1% / ~2%; identity H_out worse |
| 33-17 | 27B diagonal collector: split halves, resume test on 0.8B | quantizer | none | done | 11-02 | 11-02 | resume bit-exact; stopping rule: 27B halves median ≥ 0.995, p5 ≥ 0.964 |
| 33-18 | Estimators (Sketch B / xw / dy) + dose 128–1024, 0.8B, 3 seeds | quantizer | none | done | 11-02 | 11-02 | xw and Σdy² work as well as Sketch B (dy K3 −8.8%); 512 seqs suffice |
| 33-19 | H_out on own agentic traces (pisess) vs qad windows, 0.8B | quantizer | none | done | 11-02 | 11-02 | pisess worse (K3 +3.5%), mix no better → collection stays on qad |
| 33-20 | Sketch A refined weights Σ(xᵀH_in x)dy², 0.8B, 3 seeds | quantizer | none | done | 11-02 | 11-02 | as good as Sketch B, worse than Σdy² (K3 +2.0%) → Σdy² stays primary |
| 33-21 | Short qad documents in the collection (YAQA_SHORT), 0.8B, 3 seeds | quantizer | none | done | 11-02 | 11-02 | neutral; 640 vs 1,024 long windows +1.5% → 27B collects long windows only, to 2,048 |
| 33-22 | Σdy² dose past 1,024 and repeated windows (fresh labels), 0.8B | quantizer | none | null | 11-02 | 11-03 | no gain past 1,024 (dy1350 +1.3% K3); 27B: 512–1,024 |
| 33-23 | Σdy² 512 vs 1,024 windows | quantizer | none | done | 11-03 | 11-03 | 512 is +2.2% worse at K3 (≈2 SE): go ≥ 1,024 on the 27B |
| 33-24 | 27B diagonal H_out collection (Σdy², gen-masked, 2,048 seqs, fp32, both cards) | quantizer | none | running | 11-03 | 11-03 | gate PASS 36.6 s/seq, peak 28.4 GiB; ~21 h; swap test rule `v3/yaqa/H27.md` |
| 33-26 | Chunked-prefill noise floor, 0.8B bf16 vs own golden (chunk 0 / 2048 / 512 / 64) | eval | n/a | done | 11-04 | 11-04 | 2048 = whole (0); 512 → 0.00071 KLD, 64 → 0.00069, top-1 98.8%: chunking alone is ~0.0007, far below Code's 0.025 |
| 34 | LM head: Fisher sensitivity, slab head, mean-row, head curve | head | D | running | 10-31 | 11-01 | mean-row −30% done; slab head's whole-model check not run |
| 35 | Trellis embedding: vs Lloyd, menu sweeps, scale groups, frequency weighting, run5/run6 | embedding | ~0 | done | 10-29 | 11-02 | menu unchanged; run6 null (K1 −0.2%) |
| 36 | Fractional K: CPU sweeps, menu ends, high-K half steps, quarter-step GPU quantizer | weight codes | P+D | done | 10-31 | 11-01 | on the curve; → qtf2 (#37) |
| 37 | qtf2 fractional quantizer and its speed pass (v1–v10) | quantizer | P+D | done | 11-01 | 11-02 | finalgate PASS, 51 instances |
| 38 | Damping vs K (dampsweep, damp2, damp3) | quantizer | none | done | 10-31 | 11-01 | input 0.005 (0.05 below K1), rest 0.025 |
| 39 | Global gain scale vs K | quantizer | none | blocked | 10-31 | 10-31 | best factor rises with K; GPU gaintest result never logged; on iceman |
| 40 | Per-group weight scales at low K (Hessian-weighted) | quantizer | P+D | negative | 10-31 | 10-31 | raw gain 1.3–1.8% at equal K; after the scale overhead groups lose 8–20% at equal bits, K1–2 |
| 41 | Best-of-N rotation seeds | quantizer | none | null | 10-31 | 10-31 | seed spread 0.5% |
| 42 | Shared-input siblings sharing `suh` | quantizer | P | done | 10-31 | 10-31 | they do not share it |
| 43 | Rounded-codebook quantizers (exl3cb, int8 grid) | weight codes | P | done | 11-01 | 11-01 | pass, ×1.000 K1–6 |
| 44 | Sub-2-bit pricing on turboderp's tensors | allocation | none | done | 10-31 | 10-31 | proxy error per bit ~2.0 (K1→K2 2.11); V2's model under-predicts K1-heavy swaps by 20–36%: K1 and each sub-2 level need their own factor |
| 45 | knap4 on real options and the measured speed table | allocation | none | done | 11-01 | 11-01 | V3 option table |
| 46 | A4 worth building at any size? | activations | P | negative | 11-01 | 11-01 | no, at any size (knap4, matmul-core table); the A4 timing slot was never run |
| 47 | R1 residual-stream rotation for int8/int4 activations | activations | P | negative | 10-31 | 10-31 | A8-g128 error 6.6× worse |
| 48 | Output scales by upstream's auto rule | quantizer | none | negative | 11-01 | 11-01 | +2.5% (z 2.1) |
| 49 | radiance PR #362 (prefill softmax per row) on our eval | eval | n/a | done | 10-31 | 10-31 | KLD −0.12% (z ≈ −2.2): real but ~10× below decision noise; taken with the 1.15.x port |
| 50 | Proxy edge over turboderp, in-sample share (alias **D2**) | allocation | none | done | 11-01 | 11-01 | ~9% in-sample, rest real |
| 51 | Re-run the 12 V0 mixes with the current fit (alias **D3**) | allocation | none | blocked | 11-01 | 11-01 | on iceman and the ladder |
| 52 | KLD vs position for every quant (alias **B4**) | eval | n/a | done | 11-01 | 11-01 | rises alike for all |
| 53 | Board `bytes` = tensor payload (alias **B10**) | eval | n/a | done | 11-01 | 11-01 | board bytes = tensor payload (file minus ~10 MB header / alignment): real bpw = payload × 8 / 26,895,998,464 |
| 54 | Our K1 decode bit-exact with exllamav3 (alias **B19**) | weight codes | n/a | done | 11-01 | 11-01 | bit-exact |
| 55 | Document-reuse bias from fit-slice reuse | eval | n/a | null | 11-01 | 11-01 | no measurable flattering |
| 56 | Held-out provenance of every golden | eval | n/a | done | 11-01 | 11-01 | 317 verify docs never used, but none > 16k tokens or agentic: a clean ver3 would be unrepresentative |
| 57 | GPU search's 4–6% gap = tail-biting constraint | weight codes | n/a | done | 11-01 | 11-01 | not a flaw |
| 58 | Full uniform requant ladder (ladder2, K2.5 … ) | quantizer | none | blocked | 11-01 | 11-02 | K2.5, K3 done; K3.5/K4 lost in the iceman crash |
| 59 | exl3i encoding, importer, kernel plugin, serving path | kernels | speed | running | 11-01 | 11-02 | GPU selftests pass; mutation check 6/10; d0.sh not run |
| 60 | 2B / 0.8B proxy pipeline on the 4070 | proxy | n/a | done | 11-02 | 11-02 | goldens, evalq, score check PASS |
| 60-01 | bf16 vs fp32 inference on the 2B | proxy | n/a | done | 11-02 | 11-02 | KL 3.75e-4, flat by position; goldens stay bf16 |
| 60-02 | Prefix run vs whole document | proxy | n/a | done | 11-02 | 11-02 | cuBLAS shape rounding; run docs whole, unbatched |
| 60-03 | Our quantizer on CUDA (qtf2 fixes, qtf2check) | proxy | n/a | done | 11-02 | 11-02 | bit-identical; HIP identity gate before syncing |
| 60-04 | 2B calibration study (`CALIBRATION.md`) | proxy | n/a | dropped | 11-02 | 11-02 | stopped by the user, resumable |
| 61 | xcorp v1: 96M-token proxy corpus from the upstream sources | data | n/a | done | 11-02 | 11-02 | proxy research only |
| 62 | Research-idea run over 448 recent papers | research | n/a | done | 11-02 | 11-02 | 10 lenses + critic, ~950 ideas → 120 ranked; published `docs/ideas/` |
| 63 | Our own EXL3 quants on the challenge Hessians (vs turboderp's tensors) | quantizer | none | done | ≤10-30 | ≤10-30 | proxy error 0.75–0.80× turboderp's at K1–K4; design/v2.md §1, §7 |
| 64 | Lloyd–Max embedding under a 128-point Hadamard, 1–8 bits (`embed_tq`) | embedding | ~0 | done | ≤10-30 | ≤10-30 | used in V1 and V2; design/v2.md §1 |
| 65 | Mixing turboderp's tensors where ≥ 5 bpw | allocation | none | done | ≤10-30 | ≤10-30 | used at 3.5 and 5.5–9.0; design/v2.md §1 |
| 66 | bf16 options (33 tensors at 9.0, embedding at 8.0/9.0); bf16 head broke the model | allocation | P+D | done | ≤10-30 | ≤10-30 | used at 9.0 (33 bf16 tensors) and for the embedding at 8.0 / 9.0; a bf16 head broke the model (KLD 17); design/v2.md §1 |
| 67 | Hessian proxy error per (tensor, option) from the plugin's own decoder (`hproxy`) | allocation | none | done | ≤10-30 | ≤10-30 | ~8,100 rows; design/v2.md §2 |
| 68 | Sensitivity model kfit2 with a depth-compounding term | allocation | none | done | ≤10-30 | ≤10-30 | the compounding term mattered; design/v2.md §3, §7 |
| 69 | Exact DP solver over 1 MiB byte buckets, layer by layer | allocation | none | done | ≤10-30 | ≤10-30 | exact for kfit2's model: per-layer DP over 1 MiB buckets, nearest rounding, exact-size check; design/v2.md §4 |
| 70 | Streaming search: refit, Thompson, local search, exploration, targeted swaps (~1,450 evals) | search | none | done | ≤10-30 | ≤10-30 | local search mattered; flat past ~step 500 at 2.75–4.5; design/v2.md §5 |
| 71 | Targeted swaps to fix the K1/K2 misfit | allocation | none | done | ≤10-30 | ≤10-30 | mattered; design/v2.md §7 |
| 72 | Fixed calib pools (fit / search / qad / verify) and held-out final picks (ver2) | eval | n/a | done | ≤10-30 | ≤10-30 | fit-slice picks were ~3% optimistic; design/v2.md §6, §7 |
| 73 | Long-context check on the verify pool's two ~250k traces, whole | eval | n/a | done | ≤10-30 | ≤10-30 | V2's edge over turboderp holds from the first 8k tokens out to 250k; design/v2.md §6 |
| 74 | radiance 1.15.x plugin port (ABI 32) and KLD parity | kernels | n/a | done | 10-31 | 11-01 | 1.15.1 works; KLD 0.027122 vs 0.027122 |
| 75 | tdverify: turboderp downloads vs our packs | eval | n/a | done | 11-01 | 11-01 | all 16 byte-exact, 0 problems |
| 76 | pisess v1: own pi sessions as an agentic-trace corpus | data | n/a | done | 11-02 | 11-02 | 539 docs, 18.5M tokens, redacted; training / statistics only |
| 77 | Tile coordinate descent after LDLQ (ranked C2), CPU proxy | quantizer | none | done | 11-02 | 11-02 | 16/16: −0.7% at K2.5, every tensor better (real, small); replicate #80, stacking #79, then GPU version + KL test? |
| 78 | Gaussianized mul1 codebooks (ranked D3): CPU proxy, then 0.8B KL (`i8c`) | weight codes | P+D? | done | 11-02 | 11-02 | **adopted** (user): K3 −1.7%, K2 −2.7%, K4 −2.3%; 27B check at the trunk requant |
| 78-02 | i8c confirmation: seeds 4–5 at K2 / K3 | weight codes | P+D? | done | 11-02 | 11-02 | K3 z −3.7 (6 seeds), K2 z −2.5 (5); user called the win before it |
| 78-03 | i8c + diagonal H_out vs diagonal H_out (stacking), 0.8B | weight codes | P+D? | inconclusive | 11-03 | 11-03 | K3 1.000/1.015/1.005: no gain on top of H_out; K2 mixed; seeds 3–5 in #78-04 |
| 78-04 | i8c + edy vs edy, seeds 3–5 | weight codes | P+D? | null | 11-03 | 11-03 | 6 seeds: K3 mean log −0.007 ± 0.007, K2 −0.002: i8c adds nothing on top of H_out |
| 79 | CD on the cubic codebook (stacking), CPU proxy, Thor | quantizer | none (CD) · P+D? (i8c) | done | 11-02 | 11-02 | stacks: cd3/ldlq 0.992 at K2.5 and K3 (16/16); combined vs mul1 −2.1% / −2.6% |
| 80 | Row-sample replicate of #77 / #78 (E3_ROWSEED=1), CPU proxy, colo | quantizer | none (CD) · P+D? (i8c) | done | 11-03 | 11-03 | cd3 0.993, cubic 0.989 / 0.985, every tensor (16 / 16) |
| 81 | Faraway A1: down_proj as residual after decoded up / gate prediction, CPU trellis proxy | weight codes | P+D | done | 11-02 | 11-03 | ug/D 0.83, up-only 0.87; control #81-02 confirms it is real |
| 81-02 | A1 control: permuted predictors | weight codes | P+D | done | 11-03 | 11-03 | ugperm / uperm ÷ D 0.999–1.001 vs ug 0.72–0.94: the reduction is real prediction |
| 81-03 | A1 on the other 48 layers (K2.5, K2, K3), then all 64 at K4 / K1.5 | weight codes | P+D | running | 11-03 | 11-03 | running: A1 gain on every layer at K2.5 / K2 / K3, then K4 / K1.5 (scheduler `v3/e3/a1/a1sched.py`) |
| 82 | Where the H_out gain comes from (family / depth third; K4 / K5), 0.8B (w10) | quantizer | none | done | 11-02 | 11-03 | gdn_in 39%, mlp_in 44%, attn_in 13%, writers 3%; additive; holds at K4/K5 |
| 83 | H_out collection loss on scored positions only (gen mask), 0.8B (w01) | quantizer | none | done | 11-02 | 11-03 | K3 0.971, K2 0.950, every seed; use for 27B |
| 83-02 | Gen-masked loss for the Σdy² diagonal | quantizer | none | done | 11-03 | 11-03 | K3 0.979 (every seed), K2 0.96: use for the 27B collection |
| 84 | Gate/up share of the H_out gain (noGU; A9 failed its CPU gate), 0.8B (w03) | quantizer | none | done | 11-02 | 11-03 | A9 cos gate fail; gate/up share K3 small |
| 85 | Fixed-code scale refits (B1 H_out-weighted, C9 per-family α, C10), 0.8B (w02) | quantizer | none | done | 11-02 | 11-03 | C9 K3 0.973 K2 0.911; C10 K3 0.969 K2 0.914; overlap untested |
| 85-02 | Scale refits: B1 full, C9 on C10, both on edy | quantizer | none | inconclusive | 11-03 | 11-03 | α* on top of swj broken (K3 0.122 vs 0.099); `full` stack in the wrong order (w11 B2): redo |
| 85-03 | swj (C10 joint gate/up scale refit) alone on edy H_out, 3 seeds | quantizer | none | done | 11-04 | 11-04 | PASS: K3 0.969 / 0.972 / 0.970, K2 0.910 / 0.914 / 0.912 vs edy |
| 86 | Gen-weighted input Hessians (C13), 0.8B (w07) | quantizer | none | inconclusive | 11-02 | 11-03 | K3 0.998 (no effect); K2 0.976, every seed z −4…−5 but under its 0.97 bar: more K2 seeds would settle it |
| 87 | Sibling-sequential Hessians for down / out_proj / o_proj (C6), 0.8B (w07) | quantizer | none | done | 11-02 | 11-03 | asym K3 0.857, K2 0.824, every seed; stacking with H_out next (large) |
| 87-02 | #87 asym stacked on edy H_out, 0.8B | quantizer | none | done | 11-03 | 11-03 | K3 0.880, K2 0.845 vs edy (3 seeds, every seed): stacks on H_out |
| 87-03 | ρ screen for #87 asym (fit slice) | quantizer | none | done | 11-03 | 11-03 | fit screen: ρ 0 best (K3 0.947, K2 0.911 vs ρ 0.5); → #87-05 |
| 87-04 | #87 asym at K4 / K5 | quantizer | none | done | 11-03 | 11-03 | K4 0.868, K5 0.888 vs base (2 seeds, every seed): holds at high K |
| 87-05 | ρ 0 (full clean-output target) on search, plain and on edy; ρ −0.25 screen | quantizer | none | done | 11-03 | 11-03 | ρ 0 beats ρ 0.5: plain K3 0.945 / K2 0.916, on edy 0.964 / 0.924, every seed; ρ −0.25 worse |
| 87-06 | #87 for the 27B: layer-streamed sibling moments + writer requant (w12, `v3/sib27/`) | quantizer | none | done | 11-03 | 11-03 | 0.8B GPU test PASS (moments bit-exact, e2e 0.992 vs w07); 27B run needs iceman + ρ 0 |
| 87-07 | #87 code review and sibling-transfer test (w11, `proxy2b/exp/w11-review/`) | quantizer | none | done | 11-03 | 11-03 | review clean; correction does not transfer across sibling seeds (K3 1.049, K2 1.026): writers last |
| 87-08 | Order edy → swj → sibling moments → writers asym ρ 0, vs edy → writers ρ 0 | quantizer | none | done | 11-04 | 11-04 | PASS: swj pays on top, K3 0.985 (0.993 / 0.980 / 0.981), K2 0.971; whole stack vs edy K3 0.836, K2 0.758 |
| 88 | Activation-denoising LDLQ target (C1, level-matched noise), 0.8B (w07) | quantizer | none | negative | 11-02 | 11-03 | K3 1.031, K2 1.10 |
| 88-02 | Asym target for all 150 tensors, own-seed quantized upstream (one-shot GPTAQ-like) | quantizer | none | negative | 11-03 | 11-03 | K3 1.53×, K2 2.31× worse: only the writers with sibling inputs benefit |
| 89 | Frozen-code scale healing (H1, H4), 0.8B (w06) | healing | none | done | 11-02 | 11-03 | K2 all-layer 0.62, K3 0.83; S27 needs the user |
| 89-02 | Healing on the edy + asym stack | healing | none | done | 11-03 | 11-03 | with norms: vs edy + asym K3 0.873 / 0.858, K2 0.665 / 0.662; norms ≤ 0.2% of it |
| 89-03 | Healing on edy → swj → asym ρ 0 | healing | none | done | 11-04 | 11-04 | PASS: K3 0.886 / 0.891, K2 0.721 / 0.708 vs its stack; vs base K3 0.674 / 0.668, K2 0.507 / 0.500 |

Next free number: **90**.
