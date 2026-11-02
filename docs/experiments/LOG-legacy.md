# Experiment log: legacy entries (2026-10-29 to 2026-11-02)

Rewritten from `archive/tracker-2026-11-02.md` (topic sections E0–E4b and the Log) and `archive/deadends-2026-11-02.md` into the template in `README.md`. Newest first. The archive files are unchanged. Conventions and open items are in `## Rewrite notes` at the end.

### 2026-11-02 ~04:25 CST · #60-03 · Our quantizer on CUDA (qtf2 builds, qtf2check PASS)
- **Status:** done · **Machine:** 4070 (CUDA sm_89) · **Code:** exllamav3 fork (built in place), qtf2, `qtf2check.py`
- **Result:** qtf2 builds after two CUDA-only fixes (include order; half2 bit casts via memcpy; HIP code unchanged). `qtf2check.py` PASS on CUDA: 30 rate × codebook rows bit-identical to the fork's kernel, 1/8-step round trips. 2B input Hessians over all calib (34.7 min, long docs as 8k windows); 0.8B (14.9 min). First 2B quant, uniform K3 i8 trunk: KLD 0.0685, top-1 88.9% on g2b-search1 (173 search-pool docs, 166k gen rows; about 2x the 27B's uniform K3). Whole-trunk 2B quant ~2.5 min, its eval 23 s.
- **Decision / next:** rerun the HIP identity gate before syncing to iceman.

### 2026-11-02 ~04:20 CST · #60-02 · 2B: prefix run vs whole document
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/lencheck.py`
- **Question:** why does a document's prefix run alone differ from the whole document?
- **Result:** not a bug. cuBLAS picks its matmul kernel by matrix size; kernels sum in different orders, and the bf16 output rounding then differs. One bf16 linear (6144x2048): 27% of outputs differ between M = 512 and the same rows at M = 3519; none at M = 64 or 2048. Layer 0 differs by ~1 bf16 ulp (3.4e-3 rel); 24 layers amplify it to ~7% hidden / KL ~3e-3 on the first 64 rows. Deterministic per shape (repeat = 0), same on the pure-torch path.
- **Decision / next:** golden and evals run each document whole, same cap, unbatched (mkgold / evalq; self-KL -2.8e-7). Never chunk or batch differently between them.

### 2026-11-02 ~04:05 CST · #61 · xcorp v1: 96.1M-token proxy corpus from the upstream sources
- **Status:** done · **Machine:** colo (`/data/xcorp/v1`) · **Code:** `xcorp/`
- **Result:** the manifest's 58 sources at pinned revisions, with categories, kinds and caps, 12x the per-split budgets, rendered as the challenge (Qwen3.8 chat template: xhigh preamble, tools block, reasoning kept on every turn, structured tool calls). Not the 8 long agentic traces (unpublished). Head of each stream skipped (multi-file: second half of the files; single-file: 1.1x the records it read); a line-level check against calib + public rejected 2,205 documents. Shortfall: hermes-fc 0.8M of 1.9M (source exhausted). Build ~8 min.
- **Decision / next:** proxy research only. The private split is unpublished, so overlap with it cannot be checked; check the challenge's rules before anything trained on xcorp ships.

### 2026-11-02 ~04:00 CST · #60-01 · 2B proxy: bf16 vs fp32 inference
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/halves.py`
- **Setup:** model in two halves so fp32 fits the 4070; golden g2b-fit1 = 87 fit-pool docs, 2k cap, 86.9k gen rows, bf16, fla + causal-conv1d.
- **Result:** control: the split run in bf16 reproduces the golden (KL -4e-7, top-1 1.0000). fp32 vs the bf16 golden: KL 3.75e-4, p99 1.8e-3, top-1 98.4%, flat by position (3.9e-4 at 0-128, 3.6e-4 at 1k-2k: per-layer rounding, not state build-up). GDN kernel-path differences on the same rows ~1.5e-4.
- **Decision / next:** goldens stay bf16 (the challenge's reference is bf16; quants run on the same path, so the rounding is shared, not noise). Matters only at high bpw, where 2B KLDs approach ~1e-3: there, check a screening result with an fp32 halves run.

### 2026-11-02 ~02:05 CST · #35 · Embedding run6 (scale search 0.5..1.0): null
- **Status:** null · **Machine:** colo (run6 per the dead-end entry)
- **Result:** K1 rel 0.25700 -> 0.25642 (-0.2%, edge rows 0%); K1.25 0.18289 -> 0.18287. The scale search was never what limits the K1 / K1.25 embedding.
- **Decision / next:** menu unchanged.

### 2026-11-02 ~02:00 CST · #59 · exl3i serving path works on the GPU, up to the full model
- **Status:** running (GPU selftests pass; `d0.sh` not run) · **Machine:** iceman GPU · **Code:** radiance-hero `qc-exl3-v3` 4edad1f..340165b
- **Result:** encoding `exl3i` (scheme exl3i, transform `<cb>/h128/k<rate>`, a u32 record TABLE plane of 16-row groups); C oracle `common/exl3i_decode.c` (= the reference decoder); importer `exl3quant`'s `exl3i` quantiser (real K2.5 / K3 tensors: H-weighted error = ourq's logged proxy_err, ratio 1.000, stacked gate_up included); kernel library `exl3i` (op `exl3i_dense`, exl3r4d's kernel with the decode as a rate policy). Selftest `exl3i_gpu_test`: decode bit-exact at all 64 rates x 2 codebooks; 256 GEMMs over every rate; 48 shape / dtype cases; integer K = exl3r4d bytes; 19 refusals. Mutation check: 6 of 10 mutants run (all caught) before iceman went down. `qwen35_exl3` routes exl3i weights to `exl3i_dense` (host test).
- **Decision / next:** `d0.sh` (a level's trunk in V2 3.00, real KLD) not run yet.

### 2026-11-02 ~01:40 CST · #35 · Embedding run5 was a no-op
- **Status:** null (no-op)
- **Result:** run5 identical to run4 (K1 rel 0.25700, edge 0.2017). The edge rows sit at the LOW end: K1 826 / 4096 rows at factor 0.85 (the smallest), none above 0.95; K1.25 all at 0.95. run5 widened the top (to 1.75).
- **Decision / next:** run6 searches 0.5..1.0 (colo).

### 2026-11-02 — · #37 · Speed pass 2: profile and output-identical speedups (gated)
- **Status:** running · **Machine:** card (K2.5 layers 3-4, ladder settings) · **Code:** qtf2 (`v3/e3/exl3frac2`), `ourq.py`
- **Result:** profile: kernel 73% of wall, refit 11%, Cholesky / rotations / Hessians ~10%; the host thread waits on the GPU 86% of its samples (CPU is not a limit). Kernel time split (timing-only builds): traceback 3-5%, history writes 11-27% (K5.5: 45%). Tiles per call barely matters (64 tiles: 45k/s, 320: 50k/s). qtf2 v5 (packed history, one 16/32-bit store per thread per position) +5-11% at K1.x-2.x, +3-4% at K3.x; LDS-only step barrier ~+1%. v6 (packed minimum in the uncached path): K5.5 14.1k -> 31.1k tiles/s (x2.2); K6.5 / 7.5 barely. quantize_exl3: regularized-domain proxy and refit's e_before (verbose only) skipped; ourq caches the last Hessian (the sweep reloaded it 4x). e2e K2.5 L3-4: 52 tensors + 13 proxy rows byte-identical to the old code, 63 -> 55.5 s a layer (-12%).
- **Not done:** reusing K-independent work across levels (~7%, needs a tensor-major ladder); refit via Cholesky instead of LU (faster, not bit-identical); KA 6-7 lane-cooperative search; sub-1 0-bit-step fusion. Sub-1 (KA 0): 11.3k tiles/s at the ladder's batch before v7 (costs in global memory): ~3.9 card-h a level, ~27 for all seven.
- **Checks:** qtf2v5check (v6 vs v4 every instance), qtf2int6check, qtint5check, e2e5, speed2deploy.

### 2026-11-02 — · #37 · Speed pass 2: integer K1-4 on qtf2 (MASK 0)
- **Status:** done · **Machine:** card
- **Result:** bit-identical to both integer kernels; x1.9 / 2.1 / 1.9 / 1.3 (K1 67k, K2 81k, K3 85k, K4 93k tiles/s vs 36k / 39k / 44k / 69k). K5-8 stay on the integer kernel (LDS barrier there: +1-6%).

### 2026-11-02 — · #37 · Speed pass 2: v7, step 0 peeled
- **Status:** running · **Machine:** card
- **Result:** `first` made a compile-time constant. With the 4-way width switch the compiler had kept an `if (first)` branch on every candidate (55 KB of code, ~1,100 branches vs 15 KB / 14 at MASK 0). K1.x +70-80%, K2.x +25%, K3-4.x +50-65%, K5.5 41k, K6.5 17k, sub-1 +50%.

### 2026-11-02 — · #37 · Speed pass 2: race found and fixed (v8)
- **Status:** running (v8 gate passed; speed pass continues) · **Machine:** card
- **Result:** the LDS-only step barrier (v5) made KA 1 output nondeterministic (`qtf2race.py`: first run after load differed in v6 / v7; random runs in a byte-history build). The cost arrays are reached through generic pointers (flat loads / stores), which an LDS-scoped fence does not wait for. The earlier e2e passes were luck. Every barrier is `__syncthreads()` again (the barrier was worth ~1%); the same patch on the integer kernel was reverted before use. v8 also picks packing per rate (KA 1 and integer KA <= 3; byte history for mixed KA 2-5).
- **Checks:** v8 gate: exact vs v4 arithmetic at all 34 instances, 0 vs the fork, integer identical, race 34 x 4 runs: 0.

### 2026-11-02 — · #37 · Speed pass 2, later results
- **Status:** running · **Machine:** card
- **Result:** multi-level ourq (`-K a,b,c`) byte-identical to separate runs (K2.25 / 2.5 / 3: 52 tensors + proxy rows + stamps), only -5% time (259 vs 272 s): the K-independent work is small. Integer K6 / K7 on qtf2: identical, x2.0 / x3.0 (i8); K5 slower on qtf2 (30k vs 45k tiles/s), so it stays on the integer kernel. Two ourq per card on v8: identical outputs, throughput x1.14.

### 2026-11-02 — · #37 · qtf2 v9 (QTF2_HIOPT: inlined forward lambdas)
- **Status:** done (final build chosen) · **Machine:** card
- **Result:** identical at every KA 5-7 instance; levels out at ~25k (i8) / ~35k (mul1) tiles/s. Mixed KA 6: 15-18k -> 25k; KA 7.5: 4.3k -> 24k (i8) / 5.6k -> 33k (mul1). Integer K6 / K7 and KA 5 slower with it (unroll capped at 16 for KOUT >= 6).
- **Decision / next:** final build: HIOPT only for mixed KA 6-7 (own compile units, `qtf2_inst_a6f` / `a7f`). K5 integer stays on the integer kernel.

### 2026-11-02 — · #38 · damp3: damping held out across K0.5-7
- **Status:** running
- **Question:** does the best damping spread change with K, and integer vs fractional? (user)
- **Pass rule:** grid and rule fixed before the run in `v3/e3/damp3/damp3report.py`. Grid 0.025 / 0.01 / 0.005 / 0.0025 / 0.001 / 0.0003 / 0.0001, quantized on the split-half H_A, scored on H_B.
- **Result:** old ladder data (K2-4.5): winners integer 51 / 42 / 7% (0.001 / 0.005 / 0.025), fractional 55 / 37 / 8%; neighbour-level agreement ~53% for int-int, frac-frac and int-frac alike: no sign of a type effect. The 0.005-vs-0.001 pick is mostly noise.

### 2026-11-02 — · #34 · Head mean-row subtraction: head-only KL -30%
- **Status:** done (dry run; adopted for every head donor) · **Code:** `headremake.sh` step 1
- **Setup:** exact headsim, uniform heads, 32,768 held-out head inputs.
- **Result:** K3 0.00794 -> 0.00555, K5 0.000492 -> 0.000346 (geo ratio 0.70). Exact (softmax-invariant), free. Damping with mean-row: K3 best sigma 0.005 (-2.6% vs default), K5 best default (0.01 / 0.005 ~+1%): they disagree.
- **Pass rule:** written beforehand: every remade donor runs all three damping values and keeps the lowest headsim KL.
- **Decision / next:** adopted for every head donor. headsim's donor grouping (HS_MAXK) reproduced the earlier pricing exactly.
- **Files:** `v3/yaqa/data/head-settings.json`

### 2026-11-02 — · #35 · Embedding K1 / K1.25: wider per-group scale search (run5, run6) — null
- **Status:** null · **Machine:** colo (`/data/work/embt/rows5`, `rows6`)
- **Setup:** run5 (factors to 1.75) and run6 (0.5..1.0) on 4096 rows, trellis, group 5120.
- **Result:** K1 rel 0.25700 -> 0.25642 (-0.2%); K1.25 unchanged. At K1 a fifth of the rows sat on the low edge of the old search, but moving the edge bought nothing: the error at K1 is the trellis's, not the scale's.
- **Decision / next:** menu unchanged.

### 2026-11-01 ~23:20 CST · #43 · Rounded-codebook quantizers pass; row-slab head works; knap4 vs V2 menu
- **Status:** done · **Machine:** card (GPU) · **Code:** `cbsmoke.py`, `headrun2.sh`, exl3cb
- **Result:**
  - Rounded codebook: int8 on grid, x1.000 K1–6, x1.024 K7, x1.347 K8. int4 on its 16 levels: x1.001 K1, x1.007 K2, x1.020 K2.5, x1.059 K3, x2.5 K4.
  - Row-slab head (#34; exact head-only KL on 32,768 held-out head inputs, paired vs uniform, 95% CI): K5 budget −40% (−48…−31), K6 −36% (−42…−29), K4 −9.6% (−23…+5, n.s.); 3 slabs = per-block; without the K3 floor 3–12x worse; rare-token guard KL x1.2–1.5 at K5–6. Head ≈ 1.5–3% of total KLD: ~0.5–1% at 3.5–5 bpw.
  - knap4 (V3 menu, quality only) vs V2 menu (#45): 1.75 −25%, 2.0 −21%, 2.5 −15%, 3.0 −10%, 3.5 −13%, 4.0 −10%, 4.5 −11%, 5.0 −6%. At 2.5: steps −9.6%, trellis embedding −5.4%, damping ~0, quarter heads 0. At 2.5 bpw nothing goes below K1.5; at 2.0 the embedding hits its K1.5 floor (extend to K1). Speed: 3.0 at x1.53 prefill for +0.6% KLD; 4.0 for +2.5%.
- **Decision / next:** levels settings = int8 codebook + per-tensor damping sweep on inputs. Gate 2e crashed twice more (profiler OOM, then expandable_segments at load); restarted at 23:05 without either.

### 2026-11-01 ~23:05 CST · #57 · GPU search's "4–6% gap" is the tail-biting constraint, not a flaw
- **Status:** done · **Machine:** CPU emulation (`f16gap.sh`) · **Code:** `f16gap.sh`, `viterbi_tiles_tb` (exllamav3 heuristic, NTRY 1)
- **Result:** fp16 costs change nothing (random tiles 1.000; real tensors with LDLQ median 1.000 at K2–6). The CPU reference Viterbi has a free start state, which EXL3's circular tiles cannot encode. A CPU tail-biting search reproduces the GPU exactly (free/TB 0.948 K2, 0.962 K4 vs GPU 0.949 / 0.960). Trying 16 wrap candidates recovers 12% / 8% / 4% of the gap at K1/2/3 (0.8% / 0.4% / 0.16% error): exllamav3's heuristic is near-optimal.
- **Decision / next:** not worth a kernel change. All earlier CPU trellis sweeps (E3b, fracsweep, gscale, damping) used the free start, so their absolute errors are 4–6% optimistic; ratios between arms are unaffected.

### 2026-11-01 ~21:20 CST · #36 · Quarter-step quantizer works on the GPU (`fracsmoke.py`)
- **Status:** running (smoke-test verdict fixed; levels unblocked) · **Machine:** card 0 · **Code:** `fracsmoke.py`
- **Result:** 8192 Gaussian tiles on card 0: pack/unpack round trips and decode exact at every menu K; quarter / half steps on the curve (0.993–1.003 to K4.75). The first verdict said FAIL only because the CPU-vs-GPU check expected agreement within 3%: the GPU search is 4–6% worse than the exact CPU Viterbi at every K, integer K (exllamav3's own kernel) included, and fractional K sits within 0.4% of its integer neighbours' offset. Check fixed.
- **Decision / next:** the levels (chain0) are unblocked. Overnight CPU chains queued: fine gain sweep on Thor after the seed test; damping at 6 more depths (K3/5/7) on colo after the damping sweep. The 4–6% gap itself is open.

### 2026-11-01 ~17:45 CST · #37 · qtf2 v1 (`v3/e3/exl3frac2`): bit-identical to the fork's fractional kernel
- **Status:** running (v2 gate) · **Machine:** card · **Code:** `v3/e3/exl3frac2`, gate `qtf2check.py`
- **Result:** bit-identical to the fork's fractional kernel on all 15 old rates in both codebooks (0 index / value differences, tie-heavy and extreme tiles included). New rates (1/8 steps K0.125-2.875; sub-1 with the salted 0-bit read) pass the window-structure and pack round-trip checks. Speed: x1.27 at KA 2-4, x2.1 at K5.5, x1.03 at KA 1: still ~2x slower than the integer kernel (K3 43k vs 18.6k tiles/s as written; the order looks reversed, see Rewrite notes).
- **Decision / next:** v2 caches every codebook value a thread needs in registers (as the integer K2-5 kernel does) and keeps KA 1's costs in one LDS buffer (register-staged). Gate running.

### 2026-11-01 ~17:45 CST · #34 · Head donors at half steps work alone; K4.5 / K5.5 segfault fixed
- **Status:** running · **Code:** ourq (refit_scales)
- **Result:** head donors at half steps work alone (K3.5 25.6 GB peak). K4.5 / K5.5 segfaulted in refit_scales: a kernel's code object could not load at full VRAM (HSA_STATUS_ERROR_OUT_OF_RESOURCES). Fix in ourq (empty cache + preload the refit kernels): K4.5–6.5 then fine. K7.5 hung (50 min): skipped.

### 2026-11-01 ~17:45 CST · #34 · Slab-head curve is smooth except at the floor rule
- **Status:** running · **Code:** `headslab.py HS_FLOOR_ALL`, `floorcurve.sh`
- **Result:** with half-step donors every 1/8-bit budget step gains 7–10% head KL, but the fixed tail floor round(budget − 1.5) jumps K3 -> K4 at budget 5 (−63%) and K5 -> K6 at 7 (−45%). Between 4.25 and 4.875 the slab plans are worse than uniform half-step heads. Quarter-step donors not needed for smoothness.
- **Decision / next:** one plan per floor; headsim chooses.

### 2026-11-01 ~17:45 CST · #34 · Floor search fixes the slab head
- **Status:** running (whole-model check still to do) · **Code:** `fcurve-A/B.json`
- **Result:** with half-step donors and the tail floor chosen per budget by headsim, the curve is smooth (5–24% per 1/8 bit) and 43–54% below a uniform head at every budget 3.5–7.5 (exact head-only KL, 32,768 held-out head inputs). The best floor sits ~0.5–1 bit below the budget.
- **Decision / next:** whole-model check still to do (critique C3.5).

### 2026-11-01 ~17:45 CST · #74 · radiance 1.15.1 port works (`qc-exl3-115`)
- **Status:** done · **Code:** fork branch `qc-exl3-115` (ABI 32 pins + layout-fingerprint export)
- **Result:** all 4 plugins build; qwen35_exl3 synthetic tests pass. V2 3.00 on golden-ver1: stock 1.15.1 `--kld-ref` vs the frozen 1.13 recorder, gen-masked: 0.027122 vs 0.027122 (+0.002%, SE 0.001%; max per-row difference 0.026). Stock `--kld-ref` scores every row, so mask with gen.u8 (`eval/cmp115.py`), as mergerep does.

### 2026-11-01 ~17:45 CST · #48 · Output scales off on upstream-flagged gate/up: WORSE, +2.5% KLD
- **Status:** negative · **Code:** `v3/e3/osr/`, `v3/e3/osr/result-K3.json`
- **Setup:** K3 mul1; gate + up of the 36 layers with a flagged projection, 52 scales off; swapped into V2 3.00; golden-s2k: 583 search docs, 481k positions.
- **Result:** +2.5% KLD (z 2.1). Upstream's auto rule (ratio > 1.05) misjudges this model, like gate 2d's output-side model.
- **Decision / next:** the requant keeps output scales always on (rule fixed beforehand); the K5 repeat was dropped.

### 2026-11-01 ~17:45 CST · #34 · Low-K heads (`lcurve.json`)
- **Status:** running · **Setup:** donors K2 / K3 / K3.5 / K4, floors >= K2
- **Result:** uniform K2 head 0.0379 (x4.8 K3), but slab plans with a K2 floor are strong (budget 2.125: −48% vs uniform K2; budget 3.0: 0.0060 vs uniform K3 0.0079). Cliffs where the floor switches (2.0 -> 2.125: −48%; 3.0 -> 3.125: −41%) mean the donors are too coarse.
- **Decision / next:** the remake adds K2.25 / 2.5 / 2.75 (+ K1, K1.5); quarters at K2–3 if the cliffs remain.

### 2026-11-01 ~17:45 CST · #37 · qtf2 v3/v4 gate PASS
- **Status:** running · **Machine:** card
- **Result:** packed-min: x3.1–3.5 vs the old kernel at KA 2–4, faster than the integer kernel. Sub-1 table path is procedural with no speed gain: costs in global memory are the limit. `ourq --qtf2` end to end: 52 tensors byte-identical to the old path (K2.5, i8, damping sweep, layers 10–11).

### 2026-11-01 ~17:45 CST · #59 · exl3i oracle and decode plan (`v3/infer`)
- **Status:** running · **Code:** `v3/infer`
- **Result:** plain-C reference decode (fractional, sub-1 salt, mul1 / mul1i8; codebook exact on all 65,536 states) and the generalized per-lane plan of exl3r4d (host emulation: 409,600 lane-position cases, every V3 rate, 0 failures).

### 2026-11-01 ~17:45 CST · #36 · Extra trunk levels don't pay (knap4 `KN_FRACS_EXTRA`)
- **Status:** done (base list stands) · **Code:** `k4levels.sh`
- **Result:** K5.25 / 5.75 ≤ 0.7% (6.0 bpw); 1/8 steps at K3–5 ≤ 0.8% (4.0).
- **Decision / next:** base list stands.

### 2026-11-01 ~15:10 CST · #58 · Full uniform requant planned (user)
- **Status:** planned · **Machine:** card (~90 card-h at today's speed, after a quantizer-kernel pass)
- **Setup (plan):** K0.125–3 at 1/8 steps (sub-1 included as research), quarter steps to 4.75, K5 / 5.5 / 6.
- **Result:** CPU fracsweep below K1: rel error K0.5 0.227, K0.75 0.137, K1 0.089 (2.3–2.5x per bit), so sub-1 is a usable curve. Search slices capped at 2k positions, held-out 8k (`eval/capstudy.py`: ~27% less noise per GPU-second, calls agree 0.993–0.997). Damping sweep kept (per-tensor winners agree only 65–70% across levels). Upstream's output-scale test (`v3/e3/osr/osr.py`, all 64 layers, `v3/e3/data/osr.jsonl`) flags up_proj in 35 of 64 layers (nearly all of L26–63) and gate_proj in 17 (L49–63, L0, L2).
- **Decision / next:** a paired swap test decides the output-scale question before the requant.

### 2026-11-01 ~01:35 CST · #55 · Document-reuse bias test: no measurable flattering from fit-slice reuse
- **Status:** null · **Code:** `eval/biastest.py`; `data/biastest-2026-11-01.out` (existing scores only)
- **Setup:** 76 V2 mixes scored on a fit-pool slice (calA / fit2 / fit3, where V2's models were fitted and the mixes picked) and on ver1; each as a ratio to turboderp's untuned SC reference at the nearest bpw.
- **Result:** bias = ratio on the fit slice / ratio on ver1: median 1.024, weighted mean 1.014 [1.005, 1.023]. By slice: calA 1.027, fit2 1.009, fit3 1.033; by size < 2.5 bpw 0.97 (n = 8), above 1.02–1.03. If anything the mixes look ~2% WORSE on the slices they were fitted and selected on (slice-to-slice spread, not winner's curse).
- **Decision / next:** fit / search reuse costs at most a couple of percent here and is not the problem. The ledger's strict "spent" rule can be relaxed for non-trained documents; QAD-trained ones stay excluded until the same test is run after V4. Caveat: ver1 was the V2 monitor slice (no selection on it; mixes were queued there because of their fit-slice scores).

### 2026-11-01 ~01:25 CST · #46 · A4 timing was invalid (ntbench predates the T2 / NFQ struct fields)
- **Status:** running (rerun queued in cardzero.sh) · **Machine:** card zero · **Code:** `/data/bench/ntbench2` (rebuilt from `/root/ntbench`)
- **Result:** `/root/ntbench` (Oct 29) predates the T2 / NFQ struct fields, so both arms ran the fused NT kernel (identical hashes, no wdec kernels in the profile).
- **Decision / next:** rerun in cardzero.sh with a hash sanity check. The A4 drop (#46, 00:55 CST entry) rests on the matmul-core table, not on this slot.

### 2026-11-01 ~01:25 CST · #34 · Fractional-K heads do not fit the GPU box's RAM
- **Status:** running (integer donors only) · **Machine:** card 1 · **Code:** `headslab.py`
- **Result:** K3.5 / K4.5 OOM-killed; integer heads (~22 GB) fit alone. The slab-head curve uses integer donors K3–K8 (1/8-bit slab budgets still give a continuous curve). Head quants pause card 1's ladder; an OOM kill also took the card-1 K2.75 run once; it resumes per layer. (Superseded by the 17:45 entry: K3.5 works alone there.)

### 2026-11-01 ~01:25 CST · #37 · Ladder profile (py-spy, 150 s of K2.75)
- **Status:** running · **Machine:** card
- **Result:** 84% in ldlq, 77% of wall time waiting on the GPU (trellis search kernel), 9% g_scale search, 4% refit. No host-side speedup is possible; only the frac Viterbi kernel or fewer passes. The input-group damping sweep runs that group 3x: ~+40% time per level for a gain that D2 / damp2 put under 1%.

### 2026-11-01 ~01:20 CST · #53 · B10: the board's `bytes` is the tensor payload, not the file
- **Status:** done · **Code:** `store/sizedef.py` (4 V1 containers; HF files byte-identical to ours)
- **Result:** board vs sum of plane bytes −0.13 / −0.04 / +0.18 / +0.01 MB; vs file size −9.6 to −9.9 MB (header, directory, alignment). Real bpw = payload bytes x 8 / 26,895,998,464; "container bytes" overstates by ~10 MB = 0.003 bpw. radiance 1.15.1's Qwen3.5 plugins do not declare or load the `mtp.` block; whether an MTP drafter shipped in the container would count toward size is not knowable from the board data (none of the entries has one).
- **Decision / next:** ask the challenge author before planning on it.

### 2026-11-01 ~01:05 CST · #52 · B4 closed: KLD rises with position for every quant alike
- **Status:** done · **Code:** `eval/posbins.py`, golden-long2 (2 traces)
- **Result:** ratio to turboderp SC 4.00 per bin 0-2k / 2-8k / 8-32k / 32-64k / 64-128k / 128k+. The reference's own KLD doubles (0.0040 -> ~0.008 past 32k), but the ratios are flat: V2 4.00 0.76–0.84, V2 4.50 0.42–0.51, td SC 3.00 3.46–3.69, V2 3.00 2.5–3.3 (first bin highest). 2 docs only.
- **Decision / next:** no quant compounds error with context faster than another, so short-slice rankings hold at long context; the rise is a property of the text / position.

### 2026-11-01 ~00:55 CST · #45 · knap4 on the measured speed table
- **Status:** done · **Code:** `KN_SPEED=v3/bench/data/speedcal-stock.jsonl` (k4grid default); `KN_A4=a4time.jsonl` swaps in measured paths once A4 has run
- **Result:** also fixed: the LM head was timed at M 4096 (~77 ms, ~5% of the step) though prefill runs it on the last token only; flop shares now exclude it. Measured f16 core ~116 TF (not 135), iu8 ~2.2x. Same picks and KLDs as before at 2.5 / 3.0 / 4.0 (the int8 choice was already saturated). All-int8 speedup estimate x1.65 -> x1.88, at +0.9% (3.0) / +5% (4.0) KLD.
- **Decision / next:** decode is still the placeholder (0.006 ns/weight); A4 decides whether x1.88 holds.

### 2026-11-01 ~00:55 CST · #46 · A4 (int4 activations) not worth building at any size
- **Status:** negative · **Machine:** — (knap4, V3 menu + measured speed table) · **Setup:** sizes 1.75 / 2.0 / 2.5 / 3.0, lambda up to 1e-4
- **Result:** even under the strongest speed pressure the solver puts only 7–14% of FLOPs on A4, buying x1.88 -> x1.91–1.94 prefill for +2…+5% KLD. int8 already takes most of the time out; what is left is decode and the non-GEMM ops.
- **Decision / next:** no int4-codebook quants (they would need their own ladder: post-hoc int4 rounding costs x1.06 at K3, x2.5 at K4) and no iu4 kernel. Revisit only if the A4 timing slot (int8 with decode) comes out far worse than the matmul-core table.

### 2026-11-01 ~00:50 CST · #54 · B19 closed: our K1 decode is bit-exact with exllamav3's
- **Status:** done · **Code:** `v3/e3/b19k1.py`
- **Result:** the plugin's C oracle `exl3_trellis_decode` vs exllamav3's `reconstruct` on turboderp's SC 1.40 tensors: 4 K1, 4 K2, 4 K3 incl. lm_head, 0 differing f16 values. The +12% of our 1.40 vs his chart is not a decode bug.

### 2026-11-01 ~00:47 CST · #50 · D2 done: our proxy edge over turboderp is ~9% in-sample; the rest is real; damping holds
- **Status:** done · **Code:** `v3/yaqa/d2report.py`, data `v3/yaqa/data/d2-score.jsonl`
- **Setup:** split-half input Hessians (layers 3/10/43, 1,200 non-verify windows by document parity), ours quantized on H_A at K4 / K6, scored on H_A / H_B / published H.
- **Result:** ours / turboderp proxy, median of 17 tensors: K4 0.775 in-sample, 0.851 out of sample, 0.863 published; K6 0.774 / 0.852 / 0.862. So ~9% of the advantage is fitting to the Hessian sample at ~1.2M tokens per half; with the published 10M tokens it should be much smaller. Damping: the H_A-best value is also H_B-best for 13/17 (K4) and 14/17 (K6) tensors; the median tensor keeps 0.025 (gain small, as damp2).
- **Decision / next:** ours-vs-ours comparisons (fractional K, damping, codebook) are not affected.

### 2026-11-01 ~00:45 CST · #33-06 · D1 closed by reading the code: no auto-vs-always confound in gate 2d
- **Status:** negative (gate 2d's failure stands; alias D1) · **Machine:** — (code reading, no GPU run) · **Code:** exllamav3 `regularize`, `ourq.py`
- **Result:** output scales on auto differ from "always" only for tensors given an `out_sensitivity` (the gated-MLP path of convert.py); ourq.py never passes one. With a diagonal, the Hessian-derived scales replace the plain ones and output scaling stays on; every gate-2d tensor took them (proxy.jsonl `hout_diag_scales` true for 13/13 in dsamp-K2, dfull-K2, dfull-K3; dsamp-K3 only got 5 tensors).
- **Decision / next:** none needed; no GPU run.

### 2026-11-01 ~00:45 CST · #46 · A4 timing slot written (`v3/bench/a4time.sh`, `a4report.py`)
- **Status:** planned (queued after D2, `a4wait.sh`, card 0)
- **Setup:** ntbench on the NFQ build: f16 T2 vs the int8 path *with* its decode (nfq_act_i8 + nfq_wdec_i8 + nfq_gemm_i8), 5 TP1 shapes, K2–6, M 512 / 2048 / 4096, rocprofv3 decode share at M 2048.

### 2026-11-01 ~00:40 CST · #45 · knap4 wired to real options (quality only; V3 menu: frac + damp; measured speed table)
- **Status:** done (quality only) · **Code:** `KN_EMBT=2`, `KN_HEADSLAB=1`, `KN_LEVELS=/data/v3levels`
- **Result:**
  - `KN_EMBT=2`: trellis embedding priced from chain1's real KLD swaps (golden-ml increments scaled to the pooled corpus by the Lloyd points, 0.333; other K along the CPU curve from the nearest measured point). Result vs the CPU-priced menu (`KN_EMBT=1`): +0.0…+0.8% at 1.75–4.5 bpw, −0.8% at 5, −4.6% at 6. "The CPU-based embedding estimate holds on real KLD." Embedding still sits at its t1.5 floor at ≤ 2 bpw.
  - `KN_HEADSLAB=1` (#34): slab heads at K5 / 5.5 / 6 priced at the conservative end of headsim's interval (x0.69–0.71 head KL; K4 not offered: n.s. and rare-token KL x2.5). Picked at 3.5–5 bpw: −1.3…−1.5% total KLD at 4–5, ~0 at ≤ 3 and at 6. Needs hdslab-K5/5.5/6 containers before validation.
  - `KN_LEVELS=/data/v3levels`: measured fractional levels replace the midpoint interpolation per tensor once the level and both neighbouring integers exist (tested on synthetic data).

### 2026-11-01 00:17 CST · #33-07 · Gate 2e FAILED: gradient-weighted H_in is worse in both K
- **Status:** negative · **Machine:** card · **Setup:** gradient-weighted H_in from the 100-sequence snapshot, blocks 22/23, ver1, paired vs one-sided LDLQ
- **Result:** K2 +44% (z 10.3), K3 +31% (z 11.6).
- **Pass rule (fixed beforehand):** either K positive -> drop. So no blend test (`gate2e-blend.sh` / `hblend.py` were written and tested, unused). All backward-based Hessian ideas are closed (EXPERIMENTS.md). D2 started right after.

### 2026-11-01 ~00:10 CST · #56 · Held-out provenance of every golden
- **Status:** done · **Code:** `eval/pools/provenance.py` -> `eval/pools/provenance-2026-11-01.json`
- **Result:** verify pool: 845 docs; 317 never used by any slice (434k tokens), but none longer than 16k tokens and none agentic / long-agentic / long. Those categories (43.8% of verify tokens) are entirely spent (calB / calBbig / ver1 / ver2 / long2). All 10 verify docs >= 32k are spent; both 250k traces 17 times. Overlaps: ver1 shares 8 docs with calB and 32 with calBbig; ver2 shares 30 / 73. calA / fit2 / fit3 / ml are fit pool (ml 8–12 docs shared with the fit slices).
- **Decision / next:** a ver3 drawn from clean docs only is clean but unrepresentative (no agentic text; code / multilingual 2x over-weighted). Decision needed before building ver3 / ver4 (see TODO).

### 2026-11-01 — · #— · Stop rule for the V3 stream (user)
- **Decision:** a size band's search stops when its held-out (ver3) best-so-far stops decreasing (no gain beyond its paired SE over N ver3 reads, N fixed beforehand). No fixed eval budget (~1,500-3,000 expected). ver3 is read for ~1 in 10 evals only (a candidate that becomes its band's best on the search slice), as in V2.

### 2026-11-01 — · #— · V3 stream budget raised to ~1,200-1,500 evals (user)
- **Decision:** a per-band stop on the ver3 validation curve. Solver-model plan: an uncertainty-aware solve first (bootstrap fits, penalty per option, B11), targeted evals where the fits disagree, partial pooling of factors. The XGBoost / residual re-ranker is retried at the halfway point, with a pass rule fixed beforehand (beat kfit2 on grouped-CV pair ranking by more than the bootstrap noise and on a held-out search slice).

### 2026-11-01 — · #33 · YAQA scratch deleted on the GPU box
- **Status:** housekeeping (no result) · **Machine:** GPU box
- **Result:** deleted qfull, rows32, qdiag, full32 / full64, hout*, n16 / n32, q*, qe: 192 GB; 649 GB free. Kept: headslab / headslab2 (head-slab quants), hs (head Fisher), pylib, scripts, logs.

### 2026-11-01 — · #35 · chain1: real trellis-embedding swaps; ours vs turboderp K6 (results from ~23:50 CT 10-31)
- **Status:** done (trellis embedding and K6 comparison both measured) · **Machine:** card 1 (`chain1.sh`)
- **Setup:** V2 6.00 with golden-ml (multilingual, 67 docs); even-layer K6 swaps into V2 5.00 (fit3, 98 docs).
- **Result:** trellis t2 +0.0047 vs Lloyd 2-bit +0.0092; t3 +0.0012 vs Lloyd 3 +0.0021; t4 +0.00017 vs Lloyd 4 +0.00030; t2.5 +0.0022 ~ Lloyd 3. Trellis costs ~0.5-0.6x Lloyd at equal bits, ~0.5 bit saved at K2-3: confirms the CPU result on real KLD. The fit3 half is empty (no base6 score on golden-fit3). Both sets are on the fit pool, so not held-out numbers. K6 swaps: ours K6 -24.2%, turboderp K6 -23.1%; turboderp vs ours +1.3% (z 1.26, n.s.). Ours and his K6 are equal on real KLD (see #63). The low-damping arm (ours-K6-d005) did not quantize.

### 2026-11-01 — · #75 · tdverify: all 16 turboderp downloads byte-exact against our packs
- **Status:** done · **Result:** 337 tensors each, 0 problems. Each download has 8 MTP trellis tensors that exist only in the downloads.
- **Decision / next:** keep them (or pack them) before deleting.

### 2026-11-01 — · #38 · damp2 (CPU, colo): best lower damping vs 0.025, away from layers 11 / 43 / 63
- **Status:** done · **Machine:** colo · **Setup:** layers 1 / 5 / 19 / 30 / 47 / 58, K3 / 5 / 7 · **Files:** `v3/e3/data/damp2.jsonl`
- **Result:** the win is much smaller away from layers 11 / 43 / 63: attention inputs 0.989–0.996 (first sweep: 0.94–0.97); GDN inputs median 0.992–0.998 but one tensor 0.94; everything else >= 0.993.
- **Decision / next:** the gain is tensor-specific, which is what the per-tensor sweep in the level ladder (`--sigma-in-sweep`) is for. Expect well under 1% total.

### 2026-11-01 — · #— · Sub-1 K moved to V3.5 (user)
- **Decision:** sub-1 K goes to V3.5, before QAD, since sub-2 quants are the natural QAD targets.
- **Also in this entry:** A5 selftest classes added to DESIGN §1c.4; D6 (GDN a/b merge) closed (0.27-0.4% share); B13 `ourq.py` code stamp committed (deploy and smoke test in the morning); B19 K1 bit-compare script queued after the gate 2e verdict.

### 2026-11-01 — · #33-07 · Gate 2e MARGINAL rule (written before the verdict)
- **Status:** running (verdict in the 00:17 CST entry) · **Pass rule:** both K negative (even at z ~ -0.5) -> a blend test alpha*H_grad + (1-alpha)*H_plain (quants + evals only, ~40 min) before any 200/400-sequence collection; either K positive -> drop.
- **Note:** this entry sits after the ~00:55 CT entry in the source, but it must precede the 01:17 CDT verdict. Dated as the source gives it (see Rewrite notes).

### 2026-11-01 — · #48 · Output scales by upstream's auto rule (dead end)
- **Status:** negative · **Setup:** exllamav3's out_scales_ratio test flags up_proj in 35/64 layers and gate_proj in 17; turning their scales off (K3, swapped into V2 3.00, paired on 583 search docs).
- **Result:** KLD 2.5% worse (z 2.1).
- **Decision / next:** output-side local sensitivity models keep failing on this model (cf. YAQA gate 2d). Not adopted.

### 2026-11-01 — · #33 · YAQA output-side Hessians from sampled positions, two-sided LDLQ (dead end)
- **Status:** dropped (every backward-based Hessian variant dropped by 2026-11-01; Sketch B not tested)
- **Result:**
  - (a) bf16 backward, 200 seqs: two-sided +23% (K2) / +28% (K3) worse increment. bf16 gradients are noisy toward early layers (down_proj per-row cosine vs fp32 p10: 0.999 at layer 63, 0.02 at layer 20); the bad rows are ~3x too large and dominate H_out.
  - (b) fp32 backward, 200 seqs: +17% / +19% worse at sigma_reg_out 0.025, +6% / +10% at 0.3 (z ≥ 2). More output regularisation keeps helping, i.e. the trend is toward one-sided.
  - (c) full-token fp32 H_out, 200 seqs: still worse (+30% / +21% at 0.025; +7% / +5% at 0.3); worse than sampled.
  - (d) diagonal output scales (gate 2d): K2 +10–15%, K3 +4% (no auto-vs-always confound: D1, code reading).
  - (e) gradient-weighted input Hessian (gate 2e, 100 seqs, one-sided LDLQ, 2026-11-01): K2 +44% (z 10), K3 +31% (z 12); far worse than the plain Hessian, so the pre-set blend follow-up was skipped.
- **Decision / next:** every backward-based Hessian variant is dropped (2026-11-01). Not tested: the paper's Sketch-B estimator with far more data (64K sequences), which we cannot afford. Code: `v3/yaqa/`, tracker 2026-10-31.

### 2026-10-31 ~22:45 CST · #33 · Upstream exllamav3 already has a YAQA collector (reference v1.6.3)
- **Status:** running (second suspect for gate 2) · **Code:** `util/yaqa_hessians.py`; `convert.py -hess / -h1 / -hreg / -sh`; `--out_scales yaqa`
- **Result:** differences from ours: upstream sums d dᵀ over **every** token on the GPU (layer ranges per run, for memory); we keep 32 sampled positions per sequence (1.6%) as rows on disk. It weights tokens by xᵀ H_in x (Sketch A), we by ‖x‖² (H_in = I); bf16 activations with fp32 sums. Its notes (Qwen3.5-2B): the benefit comes from the output side alone; H_out is "dominated by relatively few high-gradient tokens"; two seeds differ ~20% at 250 full 2048-token rows. So our 6,400 sampled rows likely miss most of the mass. turboderp's SC quants take their GPTQ Hessians from the self-sampled trace (`convert.py --cal_data`), i.e. model-written text only; ours cover every calib token, prompts included.
- **Decision / next:** if gate 2b (fp32, sampled) fails too: full-token sums for blocks 22/23 (≈ 6 GB of fp32 accumulators; no room on the cards beside fp32 activations, so on the GPU box's CPU, ~10 s a sequence) before giving up.

### 2026-10-31 ~22:30 CST · #33-06 · Gate 2d failed: diagonal output scales hurt
- **Status:** negative · **Setup:** one-sided LDLQ with exllamav3's sensitivity_out_scales from diag(H_out), blocks 22/23, vs one-sided.
- **Result:** sampled-H_out diagonal K2 +14.9% (z 7.1); full-token diagonal K2 +10.2% (z 5.9), K3 +4.3% (z 3.0). Output-side Hessian information hurts in every form we tried.
- **Decision / next:** only gradient-weighted H_in (gate 2e) is left of the backward-based ideas.

### 2026-10-31 ~22:30 CST · #42 · Shared-input siblings do not share `suh`
- **Status:** done · **Machine:** CPU (our K6 quant's dumped suh, every layer)
- **Result:** signs agree 100% (one seed per input-Hessian group), but the per-channel input scales differ: gate vs up median |ratio| 1.015 (p99/p1 spread 1.27), GDN qkv vs z 1.036 (1.19), attn q vs k / v 1.092 (1.92); no pair identical. The difference is exllamav3's per-tensor scale refit (`refit_scales`).
- **Decision / next:** on V2's quants each sibling needs its own rotate + int8-quantize prep pass (cheap: ~20 MB of activations at M = 2048), not one shared tile. If shared tiles pay in the time table, the Phase-B requant can tie su across siblings (joint refit), at a quality cost still to be measured.

### 2026-10-31 ~22:20 CST · #33-05 · YAQA gate 2c failed: two-sided YAQA dropped
- **Status:** negative · **Setup:** full-token fp32 H_out (200 seqs, every position) on blocks 22/23; KLD increment vs one-sided.
- **Result:** K2 +30% (sigma_reg_out 0.025) / +7% (0.3); K3 +21% / +5%, all z ≥ 2.9. Full-token is even WORSE than sampled at 0.025 (+15% K2, +10% K3), equal at 0.3. Truncated backward checked exact (0 difference). Sampling was not the problem. Collector profile (seq 1): fwd 8.8 s, bwd 13.8 s, CPU-sum wait 11.3 s with H_in on (the CPU sums, not the GPU, are the bottleneck once H_in is added).
- **Decision / next:** two-sided YAQA dropped. Remaining backward arms: gradient-weighted H_in (gate 2e: crashed on a profiler-path bug before collecting, fixed, not rerun) and diagonal output scales (gate 2d, evals running).

### 2026-10-31 ~22:20 CST · #41 · Best-of-N rotation seeds: dead
- **Status:** null · **Code:** `seedtest.sh` (Thor) · **Setup:** 9 tensors x 4 seeds x 2 row samples, K2–4
- **Result:** seed spread 0.5%; A-vs-B correlation +0.22; best-of-4 picked on A, measured on B: −0.2% / +0.2% / +0.0%.
- **Decision / next:** dead. No seed pick.

### 2026-10-31 ~22:20 CST · #38 · Damping vs K (`dampsweep.sh`)
- **Status:** done · **Machine:** colo · **Setup:** 16 tensors; error vs undamped H; ratio to 0.025
- **Result:** attention inputs (q/k/v) gain with less damping, more at high K: K4 0.970 (0.0025) / 0.961 (0.001), K8 0.938–0.955. GDN inputs, out-projections, FFN: 0.994–1.000 at every K. 0.05 is worse everywhere. Damping does not explain turboderp's K6–8 wins (attention inputs are a small share).
- **Decision / next:** settings 0.005 (or 0.0025) for attention inputs; 0.01 elsewhere is harmless. Depth extension running (colo).

### 2026-10-31 ~22:20 CST · #39 · Gain (global scale) vs K (`gainsweep.sh`)
- **Status:** running (GPU gaintest) · **Machine:** Thor · **Setup:** 16 tensors, RMS x f, f = 0.8–1.8
- **Result:** the best factor rises with K in every family: ~1.1 to K5, 1.2 at K6, 1.3 at K7–8. Out-projections (attn_output, ssm_out) more: 1.2 (K5), 1.3 (K6), 1.4 (K7), 1.5 (K8; range to 1.8). A fixed 1.1 costs x1.05 (K6), x1.2 (K7), x2 (K8) error; for out-projections x1.2 (K6), x1.7 (K7), x4 (K8).
- **Decision / next:** this is the CPU tool's RMS scale, not exllamav3's own g_scale search. The GPU gaintest (running) shows whether exllamav3's choice already tracks this. If it doesn't, it is a strong candidate for the high-K gap.

### 2026-10-31 ~22:10 CST · #36 · Fractional-K trellis works on real tensors (`v3/e3/fracsweep`)
- **Status:** done (on the curve; → qtf2, #37) · **Machine:** CPU box A · **Setup:** 14 tensors of layers 10 / 11 / 42 / 43, every family; 128 sampled rows, same rows at every K (`fracreport.py`)
- **Result:** measured proxy error over the log-linear interpolation of the integer neighbours, median: K1.25 0.990, K1.5 0.988, K1.75 0.990, K2.25 0.999, K2.5 1.000, K2.75 1.002, K3.5 0.998, K4.5 0.996 (all tensors within 0.95–1.02). Integer slopes match the hproxy table (K1→2 2.12, K2→3 1.99).
- **Decision / next:** the shift-pattern trellis costs nothing over the integer ones, so the −5…−7% menu estimate stands. Sub-1 (K0.5 / K0.75) can't be judged this way (K0 = dropped is not a curve point); they sit well below the line and need real swaps.

### 2026-10-31 ~21:50 CST · #44 · Sub-2-bit pricing on turboderp's tensors (CPU, V2 data)
- **Status:** done · **Setup:** ours, all 337 tensors; V2's fitted model
- **Result:** proxy error per bit (ours, median log2 ratio): K1→K2 2.11, K2→K3 2.00, then 1.97 … 1.88. V2's fitted model (q = 1.004, one factor for K1–K3) under-predicts the three K1-dominated band swaps by 20% in-sample / 36% CV (K2 5%, K3–K8 ±1–4%). A refit with its own K1 factor: K1 1.35 vs K2–K3 1.09 per unit proxy error; CV log-RMSE 0.870 -> 0.845, pair ranking 3,115 -> 3,119 / 3,786. Its solutions use far fewer K1 tensors (1.5 bpw 180 -> 105, 1.75 106 -> 52, 2.0 54 -> 21). The K1→K2 KLD slope is ~2.3–2.5 bits per bit, as turboderp notes for this model (~2.35).
- **Also:** what V2 takes from turboderp at high bpw: mostly the tensors his recipes raised to K7 / K8 (6.0: 37 of 49; 8.0: 36 of 42). The fit prices our K7–K8 at 1.58x KLD per unit proxy error (K4–K6 1.25x, his ~1.0x). His per-tensor S (`kld_table.json`) vs our fitted K4 cost: Spearman 0.71 overall; within families weak. Not worth a per-tensor prior.
- **Decision / next:** V3: K1 and each sub-2 level get their own factor and must be measured; K0.25 / K0.75 only if K0.5 earns picks. Thin evidence: 3 K1 swaps. Final kfit2 model dump: `data/v3/kfit2-model-final.json`.

### 2026-10-31 ~21:00 CST · #49 · radiance PR #362 (prefill softmax rescaled per row) on our eval
- **Status:** done · **Code:** `eval/pr362/run.sh` (golden-ver1, V2 3.00)
- **Result:** KLD −0.12% (unpacked) / −0.13% (packed), z ≈ −2.2; 6.00 −0.5% ± 1.0%. Packed vs unpacked differs by +0.007% today, −0.001% with the fix.
- **Decision / next:** real but ~10x below the ±1.5% decision noise: no V2 action; take it with the 1.15.x port (V3 re-baseline then).

### 2026-10-31 ~21:00 CST · #33-03 · YAQA: bf16 backward is too noisy
- **Status:** done · **Code:** `noise.sh`, `cmp.py` (same 8 sequences, positions and labels; rows kept in fp32)
- **Result:** per-row cosine bf16 vs fp32 for down_proj: layer 63 median 1.000 (p10 0.999), layer 40 0.998 (p10 0.66), layer 20 0.988 (p10 0.02), layer 2 0.92 (p10 < 0). The bad rows are 2.7–3.2x too large, so they dominate H_out. H_out diagonal cosine 0.88 at layers 48–63, 0.36 at 0–15. The bf16 forward itself diverges on 2/8 sequences (seq 7 loss/token 1.03 vs 0.83). fp32 mode: 30 s per sequence (5.3 s bf16) after a ~15 min warm-up, 28.4 GiB peak per card.
- **Decision / next:** likely cause of gate 2's failure (blocks 22/23, diag cosine ~0.5).

### 2026-10-31 ~21:00 CST · #35 · Embedding menu, token-frequency weighted (`v3/e3/embfreq.py`)
- **Status:** done · **Setup:** the sweep's saved per-row errors x search + qad token counts
- **Result:** the error curve is unchanged (x1.40 per quarter step; frequency-weighted error +1–3%). Per-row scale still wins to K4 (G1024 x1.011 / 1.006 / 1.003 at K2/3/4, frequency-weighted); from K5 G1024 is 0.5–1% better frequency-weighted. Caveat: 48% of the 4096 random rows never occur and the rest carry 1.2% of the tokens, so the frequency weighting rests on few rows. Group-scale recheck final (16 tensors): same as before.
- **Decision / next:** per-row only (a 1% embedding-error difference at K ≥ 5 is far below anything measurable in KLD).

### 2026-10-31 ~20:40 CST · #35 · Embedding menu sweep done (`v3/e3/emb/run3.sh`, 4096 rows)
- **Status:** done · **Machine:** Thor · **Files:** `v3/e3/data/embmenu.jsonl`
- **Result:** one scale per row wins at equal bits at every K: G1024 is x1.015 worse at K1.5–2.5, x1.008 at K4, x1.003 at K5, even only at K5.75; G512 x1.01–1.02 worse throughout. The quarter-step curve is smooth (x1.40 error per quarter step at every K, i.e. ~3.9x per bit). The best scale factor reaches the search edge (1.35) for 0.6–3% of rows at K5–6 only.
- **Decision / next:** embedding menu = quarter steps, one bf16 scale per row, no group options.

### 2026-10-31 ~20:40 CST · #40 · Per-group weight scales stay dead at low K, Hessian-weighted (`v3/e3/gscale.py`)
- **Status:** negative · **Machine:** colo (12 of 16 tensors so far)
- **Result:** the raw gain at equal K is only 1.3–1.8% (ratio 0.982–0.987 at K1–2; at K0.5 groups are even slightly worse, 1.008–1.035). After the 8/G-bit scale overhead groups lose: x1.08 (G128) / x1.17–1.20 (G64) worse at equal bits, K1–2. Confirms the Oct-30 amgm null result where it was untested (Hessian-weighted, K ≤ 2).
- **Decision / next:** dead at low K.

### 2026-10-31 ~20:20 CST · #— · Overnight queue (TP2 to ~13:00 CT Nov 1), all on iceman, ordered by locks
- **Decision / queue:**
  1. gate 2c (running) -> `gate2e.sh`: blocks 22/23 again with gradient-weighted H_in (`collect.py YAQA_HIN`); if 2c passed, also 400 sequences with snapshots at 100/200/400 and split halves (`hhalf.py`) = the sample-count test.
  2. quarter-step smoke test, gate 2d, head rerun (card 0 / both).
  3. `v3/levels/chain0.sh` (card 0): gain sweep (layers 3/10/43, K2/4/6) and high-K damping sweep (K6/8), then waits for `/data/v3levels/settings.env` (default `--sigma-reg-in 0.005` at 2:30 AM CT) and builds the levels K2.25…K4.75 (exports, trunk only).
  4. `v3/levels/chain1.sh` (card 1): real trellis embedding K2–4 vs Lloyd e2–e5 on V2 6.00 (fit3 + ml); ours vs turboderp K6 (even layers on an ours-K5 base), fit3.
- **Also:** store index done: 53 packs; his plain6.00 / 5.00 packs are uniform K6 / K5; 3–6 of his tensors per pack in some sizes are byte-identical to another of his packs (counted once).

### 2026-10-31 ~20:00 CST · #36 · High-K half steps land on the curve (`fracsweep-hi.sh`)
- **Status:** done · **Machine:** colo · **Setup:** 18 tensors of layers 0 / 10 (GDN), 11 / 63 (attention), 43 (MLP); scale search 0.85–1.75; `v3/e3/data/fracsweep-hi.jsonl`
- **Result:** error / log-linear interpolation of the integer neighbours, median [range]: K5.5 0.997 [0.993, 1.042], K6.5 1.003 [0.980, 1.010], K7.5 1.006 [0.994, 1.021]. Integer steps stay steep: ×3.77 / ×3.67 / ×3.81 error per bit at K5→6 / 6→7 / 7→8. Best factor rises with K (1.15 to K6, 1.3 from K6.5–7, some 1.5 at K7.5–8). K0.25: proxy rel err median 0.36 [0.05, 0.59] (1.0 = dropped): judge only by real swaps. 3 jobs failed on a wrong tensor list (layer 43 is full attention, has no GDN tensors); not rerun, every family is covered.
- **Decision / next:** the K5–8 half steps of the menu are sound.

### 2026-10-31 ~20:00 CST · #34 · LM head sensitivity (Fisher diagonal, `v3/yaqa/headsens.py`)
- **Status:** running · **Setup:** exact Fisher diagonal, sum_t p(1−p), from the calib golden's top-512
- **Result:** ~400 of 248,320 rows hold 50% of the mass; 16–19k hold 90%; 68–90k hold 99%. The four pools agree (verify included).
- **Decision / next:** supports a row-slab head; slab test next (needs `ourq.py --save-wq` head quants).

### 2026-10-31 ~19:50 CST · #33-02 · YAQA gate 2 failed with the bf16 pilot
- **Status:** negative · **Setup:** 200 sequences, blocks 22+23 swapped into the V2 6.00 release, golden-ver1; base 0.000703
- **Result:** KLD increment over base: K2 one-sided +0.00290, two-sided +0.00357 (+23%); K3 one-sided +0.00067, two-sided +0.00086 (+28%). Suspects: bf16 noise (confirmed in 33-03), 6,400 rows for the 17,408-wide H_out, output regularisation 0.025.
- **Decision / next:** rerun in fp32 with sigma_reg_out 0.025 and 0.3 (`gate2b.sh`).

### 2026-10-31 ~19:45 CST · #— · Mining merge acted on (`local/mine/merged-2026-10-31.md`, via `mergep.sh`)
- **Decision:** DESIGN-v3 corrected (YAQA status and costs: 3.7M qad tokens, ~0.2 TB full H_out; head K priced by KLD + bytes; his S not a prior) and the merge's checks added to §6. Queued on iceman behind gate 2c:
  - `gate2d.sh` (merge D1): diagonal-only output weighting, one-sided LDLQ with exllamav3's sensitivity output scales from diag(H_out) (`ourq.py --hout-diag`), blocks 22/23 K2 / K3, paired vs one-sided.
  - `headrun2.sh` (merge A2, replaces the invalid headrun): Fisher ranking from the qad pool only (top-set overlap 0.93–0.98; fit-pool Fisher captured in the top 1,280 rows 0.610 vs 0.613), floor K3, budgets K4 / K5 / K6, judged by `headsim.py` (exact head-only KL on the 32,768 calib-A head inputs; paired vs uniform, 512-position block bootstrap). Whole-model check for the finalist only, with the fixed plugin.

### 2026-10-31 ~19:15 CST · #34 · Head slab test INVALID (plumbing bug); rerun pending
- **Status:** blocked (rerun waits for YAQA gate 2c and the cards) · **Machine:** card
- **Result:** every spliced variant (u3 / u4, slabs, Lagrange) scored KLD ~15 with top-1 0.0002 (V2 3.00 itself: 0.027). The donor heads are right (rows vs the bf16 head: K4 rel err 0.076, cos 0.997; K3 0.15), so quantization and the un-permute work. Cause: `qwen35_exl3`'s fallback for an unquantized head used `gemm_nt`, whose output is the activation dtype (bf16), written into the f32 logits buffer; never exercised before (V1 / V2 heads are always EXL3). Fix: `logits_gemm` as in the bf16 arch (radiance-hero 9a156d0, local).
- **Decision / next:** rerun = dev build of the plugin + the 14 evals (~20 min, both cards) after YAQA gate 2c. Plan numbers (proxy) are unaffected.

### 2026-10-31 ~19:10 CST · #33-05 · YAQA gate 2c started (`v3/yaqa/gate2c.sh`)
- **Status:** running · **Setup:** full-token fp32 H_out for blocks 22/23 on the pilot's 200 windows and labels; backward truncated at layer 22 (gradients at/above unchanged; checked against an untruncated 2-sequence run), CPU sums async. Then two-sided K2 / K3 at sigma_reg_out 0.025 / 0.3 vs one-sided and vs gate 2b's sampled two-sided.
- **Result (build):** first fracsmoke build failed (torch's ROCM_HOME = pip runtime without hipBLAS / thrust headers); now ROCM_HOME=/opt/rocm/core-10.1 as for the exllamav3 build.

### 2026-10-31 ~19:00 CST · #36 · Quarter-step quantizer instances written (exllamav3 branch `rxm-frac-quarter`)
- **Status:** running (not yet built) · **Code:** worktree `/workspace/forks/exllamav3-frac`; `v3/e3/exl3frac`; `ourq.py --frac-ext SRC`; `v3/e3/fracrun.sh`; `fracsmoke.py`
- **Result:** 15 instances for the V3 menu (KA 1–4 x 0x8888 / 0xAAAA / 0xEEEE, KA 5–7 x 0xAAAA), one compile unit per KA; `frac_k` derives the canonical even mask for any K = KA + n/16 (n even); exllamav3's own decode refuses non-0xAAAA rates instead of misdecoding. The live exllamav3 is a prebuilt 159 MB .so (no build tree), so the fractional kernel is built as a side extension and swapped in (prebuilt pack / unpack already take any mask).
- **Decision / next:** `fracrun.sh` (iceman, queued behind the head test) builds it and runs `fracsmoke.py`: every menu K on 8192 Gaussian tiles: on-curve vs integer neighbours, GPU vs CPU Viterbi error, pack/unpack round trip, decode check.

### 2026-10-31 ~19:00 CST · #33-01 · YAQA gate 1 passed (`v3/yaqa/collect.py`)
- **Status:** done · **Code:** `collect.py`, `fold.py`, `ourq.py --hout DIR`, `gate2.sh`
- **Result:** bf16 text model streamed tensor by tensor onto both cards (layers split 32 / 32; the transformers loader OOMs host RAM on the 30 GB box); fla 0.6.0 chunked gated-delta, gradient checkpointing, chunked sampled-label CE (labels drawn from the model's own distribution, as YAQA's Fisher). Per sequence records r_t = ‖x_t‖ dy_t at 32 random positions for all 496 trunk linears (≈233 MB bf16). 5.0 s per 2048-token sequence including writes; peak 26.9 GiB per card. The qad pool (1,819 windows, 3.7M tokens) is ~2.7 h and ~425 GB of rows.
- **Decision / next:** gate 2 (`gate2.sh`): 200-sequence pilot H_out (6,400 rows: rank-deficient for the 17,408-wide gate / up, which leans on sigma_reg_out), blocks 22+23 at K2 / K3, one- vs two-sided, swapped into V2 6.00, golden-ver1.

### 2026-10-31 ~18:55 CST · #36 · Fractional-K at the menu's ends (`fracsweep-hi.sh`, 21 tensors)
- **Status:** running (~1.5 h) · **Machine:** colo
- **Setup:** 21 tensors = the 14 of fracsweep + layers 0 and 63: K5–8 half steps and K0.25, K4.5 / 5 overlapping.
- **Result (while setting up):** `trellis.py`'s scale search {1, 1.15, 1.3} hit its top edge at K7.5 and its bottom edge (1.0) for most tensors at K ≤ 2 in the earlier sweep, so those low-K proxy numbers may be slightly pessimistic (relative K steps less affected; real quants search their own scale). Now `E3_FACTORS`, here 0.85–1.75.

### 2026-10-31 ~18:45 CST · #35 · Embedding menu sweep (`v3/e3/emb/run3.sh`, 4096 sampled rows, 47 runs)
- **Status:** running (~1 h) · **Machine:** Thor
- **Result:** quarter steps K1.5–6 x {one scale per row, G1024}, + G512 from K4; per-row error / energy / chosen factor saved (tail rows, later token-frequency reweighting). Smoke test: at K5.75 9% of groups took the top factor 1.25, so the scale search is now 0.85–1.35.

### 2026-10-31 ~18:35 CST · #35 · Embedding scale group size (Thor, 1024 rows, K2 / 3 / 4)
- **Status:** running (menu sweep pending) · **Machine:** Thor
- **Result:** error vs G128 = 1.016–1.018 at G256, 1.02–1.025 at G512, 1.022–1.031 at G1024, 1.023–1.040 at one scale per row (G5120). Scale overhead 0.125 -> 0.003 bit/value; on the trellis curve (~4x error per bit) 0.12 bit is worth ~18% error, so per-row wins by ~15% at equal bits. Per-row vs G1024: 0.013 bit (~1.8%) saved against +0.1 / +0.2 / +0.9% error at K2 / 3 / 4.
- **Decision / next:** menu: per-row scale, G1024 where it is on the hull; G1024 likely wins only around K5–6 (the menu sweep settles it).

### 2026-10-31 ~18:25 CST · #33-04 · YAQA gate 2b failed too (fp32 backward)
- **Status:** negative · **Setup:** fp32 backward, 200-sequence sampled-row pilot, blocks 22+23 into V2 6.00, golden-ver1, paired vs the same one-sided quants.
- **Result:** KLD increment over base, two-sided vs one-sided: K2 +17% (sigma_reg_out 0.025) / +6% (0.3); K3 +19% / +10% (all z ≥ 2). bf16 gate 2 was +23% / +28%: fp32 helped, and more output regularisation helps further, but the trend runs toward one-sided (infinite regularisation), i.e. our sampled H_out still hurts.
- **Decision / next:** remaining suspect: 32 sampled positions a sequence (1.6%) miss the few high-gradient tokens that dominate H_out (upstream's note). Last test before dropping YAQA: full-token sums for blocks 22/23 (`collect.py YAQA_FULL=22,23`, fp32, CPU accumulators), ~1.5 h with quant + eval. Not run tonight (head test has the cards).

### 2026-10-31 ~18:05 CST · #35 · Trellis embedding vs V2's Lloyd tables (`v3/e3/embtrellis.py`)
- **Status:** running (menu sweep to follow) · **Machine:** Thor · **Setup:** 1024 random rows, H = I, same fwht128 + per-(row,128) scale · **Files:** `v3/e3/data/embtrellis.jsonl`
- **Result:** relative error at equal bits, trellis / Lloyd: 0.70 (1 bit), 0.56 (2), 0.51 (3), 0.53 (4), 0.51 (5). Trellis K matches Lloyd at K + 0.46–0.49 bits for K ≥ 2 (DESIGN's "trellis-3 ≈ Lloyd-4" was too optimistic: trellis-3 ≈ Lloyd-3.5). ~0.023 bpw saved at equal error, ~1% KLD at ~3 bpw by V2's fitted embedding costs.
- **Decision / next:** quarter steps adopted.

### 2026-10-31 ~04:30 CST · #32 · Vocab-tiered embedding dropped
- **Status:** negative · **Setup:** early data (embedding tiers, multilingual guard slice)
- **Result:** 30k@3 + rest@1: KLD 0.0140 vs uniform e4 0.0081 (bf16 embedding 0.0080). 30k@3 + rest@2 (2.25 b/value) 0.0094 vs uniform e3 (3.125) 0.0086. On the multilingual guard slice 30k@3 + rest@2 is +40% over bf16 (uniform e4 +4%). Rare-token rows matter more than frequency predicts.
- **Decision / next:** est. gain 0–3% at medium-low confidence, not worth the risk to international users. Remaining plans finish for the record; results go into EXPERIMENTS.md as a dead end.

### 2026-10-31 ~04:00 CST · #— · IQR-V3 published (competitor)
- **Status:** n/a · **Source:** whitepaper and HF `StillDeadcode/qwen3.8-27b-iqr-v3`; kernels in radiance 1.15.0, ABI 32. Board numbers unchanged (already scored).
- **Result:**
  - Convergence on EXL3's design: computed trellis code (3 ALU ops from a 16-bit state, like EXL3's mul1), L = 16, one f32 scale a row (EXL3: svh). Their code values are int8 (q(s) in [-126, 126]), so the Viterbi searches against the exact integers the int8 tile computes. Our E3 found mul1 rounded to int8 costs x1.25 at K7 / x4.5 at K8; an int8-native code removes that at the source.
  - Prefill fold: one scale a row (applied at the store) and a per-128 block factor shared by all rows: their int8 tile folds like FP8: 1.00–1.06x FP8 time vs 1.19–1.21x with per-(row,128) scales.
  - Head-fed projections (attn_output, ssm_out, 8% of trunk) need a larger scale gain from 5 bits (1.36–1.6 vs 1.16–1.24) and still lose to block scales at 6–7 bits.
  - GPTQ damping 0.05 -> 0.01: −3.5% KL on 8 layers at 3 bits. exllamav3 adds sigma_reg 0.025 x *mean* diag(H) to every channel, ≈ 22% of the median channel's curvature with massive-activation channels (mean ≈ 9x median): the same over-damping.
  - Their GEMM tables (decode GEMV vs FP8 per width, prefill per width) are V3's speed reference.
- **Decision / next:** check our K5+ quants of attn_output / ssm_out. Cheap test: one layer's tensors at 0.025 / 0.005 / 0.0025, hproxy error. If it holds, a requant of ours K2–6 (~6 h card time) is a V2.1 / V3 quality step.

### 2026-10-31 ~04:00 CST · #74 · radiance 1.15.0 / ABI 32 port: plugins must be rebuilt and re-validated
- **Status:** planned · **Code:** exl3r4d, exl3dense, qwen35_exl3 (PR #292 on 1.13.0)
- **Decision / next:** rebuild and re-validate (byte-identical KLD) before V2 ships; V2 should target 1.15.0.

### 2026-10-31 ~03:10 CST · #32 · Vocab-tiered head, step 1 (simulation; `quantize/vtsim.py`)
- **Status:** running (embedding part next) · **Code:** `mix/headdec`
- **Setup:** ranking from token counts on search + qad pools (7.16M tokens; top 30k ids = 94.4% of tokens, 51% of ids unseen). Head inputs: 32,768 rows of the head's input on calib-A (fit pool, disjoint from the ranking), all-K6 model, via the E1b hook's `EXL3_AQ_DUMP_ONLY=336` (dump run KLD 0.00101 = the plain run). Our head decoded at K1–8. Validate first: uniform-K sim deltas vs the measured whole-model head swaps (sc4.00-e4 base: K8 0.00764, K7 +0.00002, K6 +0.00009, K5 +0.0004, K4 +0.0016, K3 +0.0066; 0.047 bpw per K step).
- **Result:** head result (`data/vtsim-head-2026-10-31.json`): small. The sim matches the measured swaps once logits are f32 (the first run used f16 logits: a ~0.0002 KL floor). Uniform head-only KL: K3 0.0074, K4 0.0018, K5 0.00047, K6 0.00012, K8 0.000012. Tiering saves only 0.01–0.03 bpw at equal KL, or cuts head KL 30–45% at equal bytes (K4 budget: 0.0018 -> 0.0010, top-1 97.7 -> 99.3%). Against the head the mixes now pick (2.0 K3, 2.5–3 K4, 4 K5, 4.5–5 K6, 6 K8) that is ~1.5–3% of total KLD. The tail cannot go below K3 (tail K2 under 100k@K4: KL 0.0067 vs 0.0018).
- **Decision / next:** not worth a head kernel / format change now (a 2-tier head stays a backlog item, ~2% gain). Embedding next: measured swaps on the same base e1 +0.015, e2 +0.0033, e3 +0.0006, e4 0, e5+ noise; `quantize/embtier.py` (per-row tier embq) on a bf16-embedding copy of the base, 3 validation + 8 tier plans queued as `vt-emb-*` on calib-A. Validation: uniform 2-bit sim 0.01096 vs the real e2 container 0.01131 on calib-A (fake-quant ≈ real).

### 2026-10-31 ~01:50 CST · #19 · V2 side-experiments for V3: per-layer edge offsets and a learned residual re-ranker
- **Status:** negative (both) · **Code:** KFIT_EDGE (kfit2); `search/rr/rr.py`
- **Result:** per-layer edge offsets in the fit: no ranking gain. Learned residual re-ranker (ridge / XGBoost no better than kfit2; GP too slow). Both logged in EXPERIMENTS.md.
- **Decision / next:** a re-ranker is worth retrying in V3 with more data. Edge offsets only if V3's new terms (activation x depth) need per-layer freedom.

### 2026-10-31 ~01:50 CST · #21 · 47 single-layer probes: the sensitivity spike is the last layer
- **Status:** null (layer 0 hypothesis wrong) · **Setup:** 47 single-layer probes
- **Result:** the sensitivity spike is the last layer (L63 ffn_gate_up), not layer 0.

### 2026-10-31 ~01:50 CST · #— · V3 backlog from the probes: K0 / K0.5 / K1.5 / K2.5 and half steps 1–8
- **Decision / next:** backlog: fractional trellis levels (K0, K0.5, K1.5, K2.5 and half-steps 1–8). Not started.

### 2026-10-31 — · #32 · International-user guard for the vocab-tiered head and embedding (plan)
- **Status:** running (plan) · **Result (plan):** calib is ~94% English / code and only 51% of ids ever occur in it, so a calib-ranked low-bit tail could hurt other languages without the score showing it. Plan: (1) language-neutral ranking `quantize/vtguard.py`: min(calib rank, token id) (keeps 99.1% of calib mass in its top 100k; calib ranking 99.6%); (2) 2-bit tail floor as the default (`vt-emb-g-*` plans; 1-bit tail also measured); (3) guard slice `golden-ml` = all 67 multilingual docs of the fit pool (77k tokens; out of sample for the ranking, which used search + qad); every variant measured on it as `vt-ml-*`, with base-bf16 and e4 refs; (4) calib-ranked / 1-bit-tail plans ship only as an explicit "English/code-tuned" variant with an asterisk, unless the ml slice shows they cost international text about nothing.

### 2026-10-31 — · #29-01 · E3b results: int8 / int4 trellis codebooks (tsweep2) — topic summary
- **Status:** done (int8 codebook free through K6; int4 grid ×1.01 at K2, ×1.07 at K3, ×2.7 at K4) · **Machine:** CPU (`v3/e3/data`)
- **Setup:** tsweep2, all 11 tensors of layers 10 and 43, 256 rows; then the full set (layers 0 / 10 / 31 / 43 / 62 / 63, 33 tensors).
- **Result:** error relative to EXL3's float mul1 codebook (layers 10 / 43, 256 rows):

| codebook | K2 | K3 | K4 | K5 | K6 |
|---|---|---|---|---|---|
| mul1 rounded to **int8** | ×1.000 | ×1.000 | ×1.001 | ×1.002 | ×1.004 |
| Gaussian int8 table | ×1.008 | ×1.008 | ×1.108 | ×0.975 | ×0.969 |
| mul1 on the **int4** grid | ×1.010 | ×1.065 | ×2.72 | ×10.5 | ×39.6 |

  Full set (33 tensors): mul1 rounded to int8, K2–K6 ×1.000–1.004 (range 0.99–1.01); **K7 ×1.25, K8 ×4.5** (9 tensors). Int4 grid: K2 ×1.01, K3 ×1.07, K4 ×2.7. W4A4 keeps trellis quality at K2–K3; only the activation side costs there.
- **Decision / next:** int8 compute keeps all of EXL3's weight quality through K6. At K7–K8, 8-bit weight values cannot hold the code: V3's top end needs f16 compute (or a 9-bit-ish decode) for K ≥ 7 tensors. The earlier "K ≥ 5 requant" estimate was for post-hoc rounding of existing quants; the searched codes say K5–6 are fine at int8 when quantized with the rounded codebook.

### 2026-10-31 — · #41 · Best-of-N rotation seeds (dead end)
- **Status:** null · **Code:** `v3/e3/seedtest.sh`
- **Result:** pick each tensor's rotation sign seed from 4 by proxy. Seed spread 0.5%, not reproducible across row samples (corr +0.22); honest gain ~0.
- **Decision / next:** dead.

### 2026-10-31 — · #40 · Per-group weight scales, finer than EXL3's per-row and per-column scales (dead end)
- **Status:** negative (first test 2026-10-30; retest 2026-10-31)
- **Setup:** first test: EXL3's rotated domain (128-Hadamard both dims, random signs) after per-row / col RMS normalisation, 0.5·log2(AM/GM) of group variances, 23 layers x all projections + head (`quantize/amgm.py`, `logs/amgm.jsonl`). Retest (`v3/e3/gscale.py`): Hessian-weighted, real trellis + LDLQ, K0.5–2, G64 / G128 8-bit scales.
- **Result:** first test 0.000–0.0025 bit/weight (MLP ≈ 0, worst tensor 0.006). Retest: raw gain 1.3–1.8% at equal K, so groups lose 8–20% at equal bits. Dead at low K too.
- **Decision / next:** rotation + row / col scales already flatten the variance. Not tested: heavy tails.

### 2026-10-31 — · #22 · GPTQ damping for our quants (deferred, not a dead end)
- **Status:** superseded (by #38) · **Code:** `quantize/damptest.py`, `data/damptest-2026-10-31.jsonl`
- **Result:** sigma_reg 0.025 -> 0.005 changes per-tensor error (undamped H) by −2 to −2.7% on GDN in_proj_qkv / z, −3 to −6.5% on attention q/k/v (layer 3), −0.5 to −1.2% on o_proj, ~0 on ssm_out and the FFN (70% of weights). Est. ~0.5–1% whole-model KLD.
- **Decision / next:** not worth a V2 requant. Use 0.005 for attention / GDN inputs in the V3 requant.

### 2026-10-31 — · #19 · Per-layer edge offsets in kfit2 (KFIT_EDGE)
- **Status:** negative
- **Setup:** a free, shrunk log-offset per (group, layer) in layers 0–3 / 60–63 on top of the 8-layer depth curve, plus a variant with knots every 6. All 47 single-layer K2 probes in the data.
- **Result:** CV pair ranking 1614 / 1618 vs 1620 / 1898 for the plain model: no gain. The probe data itself (in the plain fit) carries the allocation signal, e.g. the L63 ffn_gate_up spike.
- **Decision / next:** code kept behind the flag.

### 2026-10-31 — · #20 · Residual re-ranker on top of kfit2 (`search/rr/rr.py`)
- **Status:** negative
- **Setup:** learn log(measured / kfit2 out-of-fold prediction) from 102 summary features; re-rank candidates. Grouped 5-fold CV on 519 mixes.
- **Result:** plain kfit2 log-RMSE 0.118 / pair ranking 79.9%; ridge 0.157 / 76.4%; XGBoost (all measurements) 0.212 / 80.4%; XGBoost (mixes only) 0.127 / 80.5% (+7 pairs of 1130: noise); a Gaussian process with ARD did not finish in 16 min. The residual (~±12%) is mostly slice / run noise, not learnable structure at this data size.
- **Decision / next:** revisit with V3's larger dataset (retry rule in BACKLOG).

### 2026-10-31 — · #32 · Vocabulary-tiered embedding (`quantize/embtier.py`; data in the new-format log)
- **Status:** negative · **Setup:** rows ranked by token frequency; top tiers at 3–4 bits, tail at 1–2 bits; fake-quantised on a bf16-embedding copy of sc4.00-e4; calib-A + a multilingual guard slice (67 fit-pool docs).
- **Result:** uniform e2 +38% / e3 +8% / e4 +2% calib-A KLD over a bf16 embedding (multilingual +64 / +20 / +4%). Best tiered, 20k@4 + 30k@3 + rest@2 (2.41 bits a value): +11% / +28%: better than uniform at equal bytes, but against the e3 the mixes use it saves 0.034 bpw for +3% KLD, net ~1–2% at 3–4 bpw, while multilingual text pays more (+28% vs +20%). 1-bit tails are bad everywhere (+29–76%; multilingual +60–145%): rare-token rows matter far more than their frequency. A language-neutral ranking (min(calib rank, token id)) did not help the multilingual slice.
- **Decision / next:** dropped (not worth hurting international text). The head version: simulated, ~2% (backlog).

### 2026-10-31 — · #47 · R1 residual-stream rotation for int8 / int4 activations (dropped)
- **Status:** negative (heading says dropped; measured worse) · **Code:** `v3/e1b/r1test.py`; data `v3/e1b/data/r1.jsonl`
- **Setup:** SpinQuant-style R1: one random orthogonal 5120² Q folded into the residual stream, the readers' norm gains (1 + w) folded into their weights. CPU proxy on the K6 activation dumps, all 208 residual-reader linears, output-error energy vs today's block Hadamard with the quant's own suh.
- **Result:** A8-g128 6.6x worse; A8 per-token 2.1x; A4-g64 5.7x (layers 0–15: 17x / 4.9x / 15x; layers 48–63: 1.3x / 0.66x / 1.3x). R1 does spread the massive-activation channel (top channel's share of input energy 65% -> 0.3% in layers 0–15), but this model's norm gains shrink that channel, and R1 must fold the gains into the weights, so the rotation spreads a much larger outlier over every 128-block. Only per-token A8 in deep layers gains, and per-token is ~5x worse than g128 there to begin with. A learned Q (SpinQuant's Cayley optimisation) would not close a 6.6x gap. Side result: the quant's learned input scales (suh vs signs only) lower the A8-g128 error 5%, per-token 35%.
- **Decision / next:** dropped.

### 2026-10-30 ~21:00 CST · #31 · First real int8 prefill kernels run end to end
- **Status:** running · **Code:** `v3/nf/kern/exl3r4d`, runner `/root/e1b/nfkrun.sh` · **Machine:** card
- **Result:** K6 all-int8: KLD 0.00178, eval 74 s vs 100 s (f16). Detail in the E4 summary.

### 2026-10-30 ~21:00 CST · #31 · Post-hoc int8 rounding of EXL3 quants costs at K ≥ 5; requant planned
- **Status:** running
- **Result:** the +0.0003 over fake-quant is the post-hoc int8 rounding of an existing EXL3 quant, matching the paper bound (K4 +4%, K5 +17%, K6 +60% weight error).
- **Decision / next:** requant planned (K5–8 with the int8-rounded codebook).

### 2026-10-30 ~21:00 CST · #30 · V0 validation: the weight model is off at 2 bpw; activation increments track
- **Status:** running · **Result:** ours-K1 / K2 overrated at 2 bpw; fixed by the r9 refit. Activation increments track.

### 2026-10-30 ~21:00 CST · #30 · K3: the new-format solver (`v3/nf/knap3.py`; topic summary)
- **Status:** running (V0 validation; later superseded by knap4, #45) · **Code:** `v3/nf/knap3.py`, model from `nf/kfit2dump.py`
- **Setup:** options per tensor: f16 (any EXL3 option or bf16); a8: ours-K1..8 trellis -> int8 x group {pt, 1024..64}; a4: ours-K1..3 on the int4 grid x group; shared-input linears share one mode + group. KLD: weights from kfit2's compounding model (551 measurements, dumped from round 9); activations additive, from E1c. Time: E2 TOPS at M = 4096 + a decode pre-pass (placeholder 0.006 ns/weight); non-GEMM work is 8% of today's step. Solve: min KLD + λ·time, bytes strictly under the target; layered DP with stage weights; ~2 s per point.
- **Result:** first frontier (prefill speedup vs today's all-f16 EXL3):

| target | all f16 | ×1.3 | ×1.5 | ×1.65 |
|---|---|---|---|---|
| 3.0 | 0.0256 | 0.0257 (+0.2%) | 0.0258 (+0.6%) | 0.0269 (+5%) |
| 4.0 | 0.00650 | 0.00657 (+1%) | 0.00665 (+2%) | 0.00707 (+9%) |
| 6.0 | 0.00063 | 0.00070 (+11%) | 0.00080 (+27%) | 0.00107 (+70%) |

  Readout: at 3–4 bpw, a8 is nearly free (90–95% of FLOPs at a8, a few sensitive tensors at f16); at 6 bpw the activation cost is a large share, so A16 matters; A4 is almost never chosen. Also at this point: the solver works; the V0 validation shows the weight model off at 2 bpw (fixed by the r9 refit); the bf16 head breaks mixes (disabled); the group size becomes a per-tensor option.
- **Decision / next:** caveats: the decode pre-pass time is a placeholder; g128 / g64 speeds are extrapolated. The KLD model is unvalidated for new-format mixes: it needs the E4 prototype to measure real mixes, then a refit.

### 2026-10-30 ~21:00 CST · #31 · E4: real int8 prefill kernels (`v3/nf/kern/exl3r4d`; topic summary)
- **Status:** running · **Machine:** card (build `/root/nfk`, runner `/root/e1b/nfkrun.sh`)
- **Setup:** kernels (per-token activation scales only so far): `nfq_act_i8` (quantizes the prep kernel's rotated rows to int8 in fragment order); `nfq_wdec_i8` (T2's trellis decode with int8 output, rint(v / 4 k_inv)); `nfq_gemm_i8` (iu8 WMMA core with T2's epilogue). Switched per tensor by `NFQ_MASK` ('8').
- **Result:** first end-to-end run (k0, ours-K6, calib-A, all 337 linears int8): **KLD 0.00178, top-1 98.13%, eval wall 74 s** (f16 100 s; fake-quant A8 per-token 0.00149). Correct first try; the whole eval 26% faster untuned (including load and scoring). The +0.0003 over fake-quant is the post-hoc int8 rounding of an existing EXL3 quant, matching the paper bound (K4 +4%, K5 +17%, K6 +60% weight error); E3b's ×1.004 needs the trellis searched with the int8 codebook.
- **Decision / next:** K ≤ 4 existing quants run as is; K ≥ 5 should be requantized with the int8-rounded codebook: a small change to exllamav3's codebook (same hash, rounded values), a ~6 h requant of K5–8 (or K1–8). Next: group scales in the real kernel, the requant, split-K for small N, and per-op timing (`--profile-ops`) for the real time table.

### 2026-10-30 ~21:00 CST · #31-01 · E4b: end-to-end prototype at ~4 bpw (topic summary; planned, no result yet)
- **Status:** planned · **Date:** the section is undated; placed at E4's date.
- **Setup (plan):** trellis -> int8, W8A8 everywhere except the A16 picks. Measure KLD and prefill / decode speed against ours-K4 EXL3.

### 2026-10-30 ~20:40 CST · #30 · Validation tooling and the first 12 validation evals
- **Status:** running · **Code:** `EXL3_AQ_GROUPS` (per-tensor group string in the hook); knap3 `KN_WRITE` (exports mix + masks); `nf/nfval.sh`
- **Result:** first 12 validation evals running.

### 2026-10-30 ~20:40 CST · #26-01 · Batches 4–7 in: A8 g256–g1024 plateau at +0.00015
- **Status:** done · **Machine:** card
- **Result:** A8 g256 / g512 / g1024 all about +0.00015–0.00018; only g128 and smaller go lower (+0.00005).

### 2026-10-30 ~20:40 CST · #26 · E1 / E1b: activation quantization cost (hook in `v3/e1b`; topic summary)
- **Status:** done (A8 g128 near-free; A4 g128 still a candidate at ≤ 3.5 bpw, pending E2) · **Machine:** card (`e1brun.sh`)
- **Setup:** our K6 EXL3 weights, scored on golden-calA (fit pool, 90.8k positions). Baseline KLD 0.00101, top-1 98.75%. Paired SE ±1.5%. The hook fake-quantizes the rotated GEMM input, one absmax scale per token row unless noted. No-op run byte-identical to the normal K6 score. Map: 337 linears in issue order.
- **Result, A8 (int8):**

| run | KLD | added | top-1 |
|---|---|---|---|
| all linears | 0.00149 | +0.00047 | 98.47% |
| all, **g128** activation scales | 0.00106 | +0.00005 | 98.71% |
| all, g32 | 0.00104 | +0.00003 | 98.68% |
| all, g256 | 0.00118 | +0.00017 | |
| all, g512 | 0.00119 | +0.00018 | |
| all, g1024 | 0.00116 | +0.00015 | |
| all, per-row, clip 0.9 | 0.00221 | +0.00120 | (clipping hurts A8) |
| per family, per-row: attn_in / attn_out / gdn_in / ssm_out | 0.00103 / 0.00100 / 0.00101 / 0.00105 | ≤ +0.00004 | |
| per family, per-row: gate_up / down / head | 0.00112 / 0.00103 / 0.00111 | +0.00011 / +0.00002 / +0.00010 | |
| all but down and head, per-row | 0.00124 | +0.00023 | 98.57% |

  A8 everywhere (per row) costs more than our best 7.5 bpw EXL3 mixes measure in total (0.00026–0.00035). IQR's 8.5 bpw I8A8 entry (0.00046) is consistent with this.

- **Result, A4 (int4):**

| run | KLD | added | top-1 | ~share of linear FLOPs |
|---|---|---|---|---|
| all linears | 0.150 | +0.149 | 83.9% | 100% |
| attn_out | 0.0085 | +0.0075 | 96.5% | ~2% |
| ssm_out | 0.0141 | +0.013 | 95.5% | ~5% |
| attn q/k/v | 0.0237 | +0.023 | 94.3% | ~5% |
| gdn_in (qkv, z) | 0.0241 | +0.023 | 93.9% | ~13% |
| down | 0.0262 | +0.025 | 93.7% | ~14% |
| gate_up | 0.0264 | +0.025 | 93.0% | ~28% |
| head | 0.0333 | +0.032 | 91.0% | ~10% |
| layers 0–15, all linears | 0.0295 | +0.028 | 93.8% | 25% |
| layers 16–31, all linears | 0.0199 | +0.019 | 94.6% | 25% |
| layers 32–47, all linears | 0.0305 | +0.029 | 93.5% | 25% |
| layers 48–63, all linears | 0.0407 | +0.040 | 91.9% | 25% |
| all linears, **g128** scales | 0.0312 | +0.030 | 92.3% | 100% |
| gate_up, g128 | 0.0100 | +0.009 | 95.8% | ~28% |
| gate_up, per-row, clip 0.85 / 0.7 / 0.55 | 0.0183 / 0.0146 / 0.0195 | +0.017 / +0.014 / +0.018 | | ~28% |
| gate_up, g128 + clip 0.85 | 0.0097 | +0.009 | 96.0% | ~28% |
| gate_up, g256 | 0.0127 | +0.0117 | | ~28% |
| gate_up, g256 + clip 0.85 / g512 / g512 + clip 0.85 | 0.0100 / 0.0144 / 0.0111 | +0.009 / +0.013 / +0.010 | | ~28% |
| gate_up g64 / g64 + clip 0.85 | 0.0092 / 0.0115 | +0.008 / +0.010 | | ~28% |
| down g64 / g64 + clip 0.85 | **0.0048** / 0.0052 | **+0.0038** / +0.0042 | | ~14% |
| down g256 / g256 + clip 0.85 | 0.0073 / 0.0064 | +0.006 / +0.005 | | ~14% |
| down g512 / g512 + clip 0.85 | 0.0088 / 0.0075 | +0.008 / +0.006 | | ~14% |
| down, g128 | 0.0070 | +0.006 | 96.8% | ~14% |
| down, per-row, clip 0.85 / 0.7 / 0.55 | 0.0205 / 0.0190 / 0.0365 | +0.019 / +0.018 / +0.035 | | ~14% |
| down, g128 + clip 0.85 | **0.0055** | **+0.0045** | 97.0% | ~14% |

- **Decision / next:** A8 with g128 scales is near-free for quality. A16 stays an option only for the very top of the Large tier, and only if E2 shows g128 iu8 is slow. A4 g128 stays a candidate at ≤ 3.5 bpw, pending E2 (iu4 at g128 measured 484 TOPS in the microbench). Batches 4–7: A8 g256 / g512 / g1024 plateau at about +0.00015–0.00018; A4 clipping helps at large groups and hurts at g64.

### 2026-10-30 ~20:15 CST · #29-01 · E3b full: int8 trellis codebook free through K6; int4 grid free at K2–3
- **Status:** done · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Result:** the int8 trellis codebook is free through K6; the int4 grid is free at K2–3.

### 2026-10-30 ~20:15 CST · #30 · knap3 solver works; first frontier: ×1.5 prefill for +2% KLD at 4 bpw
- **Status:** running · **Code:** `v3/nf/knap3.py`
- **Result:** solver works. First frontier (target 4.0): ×1.5 prefill for +2% KLD (0.00665 vs all-f16 0.00650).

### 2026-10-30 ~20:15 CST · #30 · bf16 head breaks mixes in the EXL3 pipeline (r9 9.0 / 9.5); disabled
- **Status:** running
- **Result:** found that the bf16 head breaks mixes in the EXL3 pipeline (r9 9.0 / 9.5 options).
- **Decision / next:** disabled.

### 2026-10-30 ~20:15 CST · #29-01 · E3b: trellis decoding to int8 (tool; topic summary)
- **Status:** done (tool validated; int8 free through K6 on the full run) · **Machine:** CPU · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Setup:** CPU reimplementation of EXL3 quantization on sampled rows: block LDLQ, 16-bit bitshift trellis over 16x16 tiles, per-row RMS scale. Simplifications: free start state (not tail-biting), no output-scale refit.
- **Result:** validation: layer 10 ssm_z, K4, 64 rows: rel 5.37e-4 vs our real ours-K4 4.16e-4 (×1.29). Slightly worse than exllamav3 but good for relative comparisons.

| tensor | K | float mul1 | mul1 rounded to int8 | int8 cost |
|---|---|---|---|---|
| L10 ssm_z (64 rows) | 4 | 5.37e-4 | 5.43e-4 | **+1%** |

  The paper bound is +4% at K4. tsweep1 covers K2–6 plus a Gaussian int8 table and an int4-grid codebook. Full run: int8 codebook free through K6; int4 grid free at K2–3.
- **Decision / next:** trellis quality survives int8 compute at mid bits.

### 2026-10-30 ~19:50 CST · #27 · E1c per-tensor activation costs done and calibrated (±30%)
- **Status:** done · **Code:** `aqcal.py`
- **Result:** per-tensor activation costs done and calibrated (±30%).
- **Decision / next:** these are the knapsack's per-tensor activation costs. Next: a depth term.

### 2026-10-30 ~19:50 CST · #29-02 · E3c row-slab K: ~0.05 bit on the big tensors → low priority
- **Status:** done · **Code:** `v3/e3/rowslab.py`, `rowslab.jsonl`
- **Result:** ceiling ~0.05 bit on the big tensors at 128-row slabs; the outlier tensors carry most of the gain.
- **Decision / next:** low priority.

### 2026-10-30 ~19:50 CST · #27 · E1c: per-tensor activation costs (proxy; topic summary)
- **Status:** done (calibrated, ±30%) · **Machine:** CPU box A · **Code:** `v3/e1b/aqproxy.py`, `aqcal.py`; data `v3/e1b/data`
- **Setup:** dump of 512 real input rows per linear (calib-A, K6 model). Scoring: every tensor x {A8: per-row, g1024…g32; A4: per-row, g512…g32} x clip, rel_out = Σ‖W·e‖² / Σ‖W·x‖², hook's exact quantizer and rotation; 10,784 rows; ~10 s per tensor.
- **Result:** calibration (refit on 45 runs after batches 4–7): log-RMSE 0.25, factors unchanged within 3%. First fit on 37 E1b KLD runs (KLD increment ≈ Σ_families k_f · Σ rel_out): per-family factors give log-RMSE 0.26 (≈ ±30%) over four orders of magnitude; one global factor gives 1.19. Factors: gate_up 0.020, attn_in 0.015, down 0.0097, attn_out 0.008, gdn_in 0.006, ssm_out 0.005, head 1.07 (the head's logit error counts ~100x more per unit of relative output error). Worst misses: tiny A8 runs near the noise floor. Depth: L48–63 under-predicted 1.7x (late layers more sensitive than the proxy): add a depth term. Group size matters very unevenly: A4 rel_out per-row -> g128 drops x2 on L0 gate_up but x28 on L3 q/k/v.
- **Decision / next:** these are the knapsack's per-tensor activation costs.

### 2026-10-30 ~19:50 CST · #29-02 · E3c: row-slab K (`v3/e3/rowslab.py`; topic summary)
- **Status:** done (low priority) · **Machine:** CPU · **Code:** `v3/e3/rowslab.py` · **Files:** `rowslab.jsonl`
- **Setup:** ceiling of re-allocating bits across output-row slabs at the same total error: ½·log₂(AM/GM) of the slab Hessian norms (input side only). 49 tensors over 9 layers.
- **Result:** param-weighted gain:

| slab | all tensors | gate_up + down only |
|---|---|---|
| per row | 0.29 bit | 0.18 bit |
| 128 rows | 0.12 bit | 0.05 bit |
| 256 rows | 0.11 bit | 0.045 bit |

  Outliers: a few small early tensors carry most of the gain: ssm_out L0 3.2 bit, attn_output L3 2.2, attn_qg L3 0.9, down L0 0.5 (at S128). These are residual writers with "massive activation" channels. For the big tensors ~0.05 bit at 128-row slabs.
- **Decision / next:** not worth kernel complexity now; low priority. Revisit with output-side importance (YAQA H_out), or "a few rows at bf16" for the outlier tensors.

### 2026-10-30 ~19:30 CST · #27 · Activation dump: 512 rows per linear, sent to CPU box A
- **Status:** running · **Machine:** CPU box A
- **Result:** 512 real input rows per linear, 3.3 GB. The dump run's KLD is byte-identical to the baseline. The proxy scorer is running.

### 2026-10-30 ~19:30 CST · #28 · E2 M sweep, and group scales in the real kernel
- **Status:** running · **Code:** `v3/bench/igemm.hip`
- **Result:** the group scales spill at the big tile (119 VGPRs): iu8 g128 73 TOPS at 128×256, unusable. At 128×128 (no spills) −12…−29%.

### 2026-10-30 ~19:30 CST · #28 · E2: int8 / int4 GEMM at real tiling (`igemm.hip`; topic summary)
- **Status:** running (decode pre-pass and scale tricks to do) · **Machine:** card 1 (first run, no tuning) · **Note:** this block has no heading in the source; it sits inside the E1c section.
- **Result, first run (M = 4096 unless noted):**

| shape | iu8 | iu4 | f16 T2 core |
|---|---|---|---|
| gate_up 34816×5120 | 5.60 ms (261 TOPS) | 2.96 ms (494) | ≈10.8 ms |
| down 5120×17408 | 2.91 ms (251) | 1.46 ms (500) | ≈5.4 ms |
| GDN in 10240×5120 | 1.71 ms (251) | 0.89 ms (485) | ≈3.2 ms |
| M = 512, 5120×5120 | 195 TOPS | 379 TOPS | |

  GEMM core: iu8 ≈1.9x our f16, iu4 ≈3.7x; 78–81% / 71–73% of peak. Outputs exact vs a naive reference.

- **Result, M sweep (TOPS):**

| M | gate_up iu8 / iu4 | down iu8 / iu4 |
|---|---|---|
| 512 | 237 / 385 | 189 / 357 |
| 1024 | 253 / 477 | 241 / 384 |
| 2048 | 259 / 493 | 248 / 482 |
| 4096 | 261 / 494 | 251 / 500 |
| 8192 | 264 / 495 | 255 / 510 |

  `down` at M = 512 is under-occupied (too few tiles): needs split-K.

- **Result, group activation scales (gate_up, M = 4096):** at the 128×256 tile the extra float accumulators spill (119 VGPRs): iu8 g128 73 TOPS, unusable. At 128×128 (no spills):

| | per token | g1024 | g512 | g256 |
|---|---|---|---|---|
| iu8 | 254 | 225 (−12%) | 220 (−14%) | 210 (−17%) |
| iu4 | 488 | 390 (−20%) | 375 (−23%) | 346 (−29%) |

- **Decision / next:** to do: the trellis -> int8/int4 fragment decode pre-pass; the activation quantize step fused into the rotation; an M sweep (incl. expert-sized shapes); tuning. Group scales: first version; to try prefetching the group scales and power-of-two scales applied as integer shifts.

### 2026-10-30 ~19:00 CST · #28 · E2 first run: iu8 ~255 TOPS, iu4 ~490 TOPS on real shapes
- **Status:** done · **Code:** `v3/bench/igemm.hip`
- **Result:** iu8 ~255 TOPS, iu4 ~490 TOPS on real shapes (1.9x / 3.7x our f16 GEMM).

### 2026-10-30 ~18:50 CST · #28 · Group-scale speed for iu8 and iu4 measured
- **Status:** done · **Result:** iu8 g512 −3.5%, g128 −18%.

### 2026-10-30 ~18:50 CST · #26-01 · A8 g256 +0.00017; batches 4–7 queued
- **Status:** running · **Result:** A8 g256 +0.00017. Batches 4–7 (g64 / g256 / g512 / g1024, A4 and A8) queued.

### 2026-10-30 ~18:50 CST · #30 · Group size becomes a per-tensor knapsack option
- **Status:** running (design) · **Result:** the group size becomes a per-tensor knapsack option.

### 2026-10-30 ~17:50 CST · #26-01 · A8 with g128 activation scales costs only +0.00005
- **Status:** done · **Result:** A8 with g128 activation scales costs only +0.00005 (per-row: +0.00047). The A8 floor was the scale granularity.

### 2026-10-30 ~17:50 CST · #26-02 · A4 g128: 5x better than per-row
- **Status:** done · **Result:** A4 g128: 5x better than per-row (all 0.031); gate_up g128 +0.009.

### 2026-10-30 ~17:50 CST · #29 · E3 sweep moved to CPU box B (44 jobs at once)
- **Status:** running · **Machine:** CPU box B · **Code:** `sweep2.sh`

### 2026-10-30 ~17:45 CST · #— · CPU work moves to CPU boxes A and B; sweep jobs rewritten
- **Decision / result:** CPU work moves to CPU box B (48 threads) and CPU box A (36 threads); the VM is downsized to 8 vCPU / 16 GB. Sweep jobs rewritten as single-thread processes with row sampling and a vectorized Viterbi: ~12x (trellis) and >100x (GPTQ on big tensors) faster.

### 2026-10-30 ~17:30 CST · #29-01 · E3b trellis tool built and validated
- **Status:** running (full sweep pending) · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Result:** tool validated: ×1.29 vs exllamav3 K4 (layer 10 ssm_z, 64 rows); int8 codebook costs +1% at K4.

### 2026-10-30 ~17:30 CST · #— · Calib Hessians on the VM (47 GB, 257 files)
- **Result:** input Hessians on the VM (47 GB, 257 files).

### 2026-10-30 ~17:00 CST · #— · Tracker created
- **Result:** tracker created.

### 2026-10-30 ~17:00 CST · #26-02 · E1b A4 per-family results in; batch 3 queued
- **Status:** running (batch 3 queued) · **Result:** A4 per family (rows in the E1 summary): attn_out 0.0085, ssm_out 0.0141, attn q/k/v 0.0237, gdn_in 0.0241, down 0.0262, gate_up 0.0264, head 0.0333. Batch 3 queued: A4 g128 and clipping.

### 2026-10-30 ~17:00 CST · #— · Calib Hessians downloading to the VM from HF
- **Result:** calib Hessians downloading to the VM from HF.

### 2026-10-30 ~16:40 CST · #26 · E1b A8-all +0.00047 (Large needs A16), A4-all 0.150
- **Status:** done · **Result:** A8 all linears +0.00047 (per-row; Large needs A16); A4 all linears 0.150.

### 2026-10-30 ~16:40 CST · #29 · E3 partial: int codes ×2–3.5 vs trellis
- **Status:** running · **Result:** plain integer codes ×2–3.5 vs trellis (3 tensors so far).

### 2026-10-30 ~16:40 CST · #29 · E3: weight codes, quality per bit (Hessian proxy; topic summary)
- **Status:** running (provisional decision; 3 tensors so far) · **Machine:** CPU · **Code:** `v3/e3/e3.py`
- **Setup:** metric rel = tr(dW H dWᵀ)/tr(W H Wᵀ), the same as hproxy. Shown: median ratio to our EXL3 ours-K at the same bits. Data: 3 tensors so far (layers 10 / 43).
- **Result:**

| bits | GPTQ, per-channel int | GPTQ g128 -> int8 | Lloyd LUT -> int8 | RTN (rotated) |
|---|---|---|---|---|
| 2 | ×2.3 | ×1.7 | ×1.7 | ×12.7 |
| 3 | ×2.1 | ×2.0 | ×2.0 | ×11.3 |
| 4 | ×2.7 | ×2.3 | ×2.2 | ×13.0 |
| 5 | ×3.1 | ×2.5 | ×3.0 | ×13.8 |
| 6 | ×3.5 | ×3.4 | ×4.5 | ×15.3 |
| 8 | ×3.5 | ×19 | ×12 | ×16.3 |

  Plain integer codes cost about 0.6–0.9 bit against the trellis, and GPTQ is mandatory. Folding group scales into int8 has a floor around 1e-4 (≈ K6). On paper, the EXL3 mul1 codebook rounded to int8 adds ~6e-5 relative error: +4% on K4, +17% on K5, +60% on K6; it dominates K7–8.
- **Decision / next:** provisional: a trellis decoding to int8 for ≤ ~K5. Above that, f16 compute (EXL3 as is).

### 2026-10-30 ~16:20 CST · #26 · E1b hook built in a separate tree; no-op run byte-identical
- **Status:** done · **Code:** `/root/e1b` (separate tree)
- **Result:** the hook's no-op run is byte-identical to the normal K6 score.

### 2026-10-30 — · #25 · E0 iu4 / iu8 peaks and scale-group cost measured; fp4 absent
- **Status:** done · **Machine:** GPU (gfx1201)
- **Result:** measured (details in the E0 summary). fp4 absent.

### 2026-10-30 — · #25 · E0: hardware facts (gfx1201, measured; topic summary)
- **Status:** done · **Machine:** GPU (gfx1201; ISA checked for every number)
- **Result:** measured peaks:

| op | peak |
|---|---|
| iu8 16×16×16 | 323 TOPS |
| fp8 16×16×16 | 321 TOPS |
| iu4 16×16×32 | **685 TOPS** |
| iu4 16×16×16 | 328 TOPS |
| f16 (earlier) | 158 TF |
| fp4 / fp6 | none (gfx1250 only) |

Activation scales applied along K (`wmmagrp.hip`, a tight loop, so the worst case):

| scale group | iu4 TOPS | iu8 TOPS |
|---|---|---|
| 64 | 331 | 220 |
| 128 | 444 | 265 |
| 256 | 536 | 295 |
| 512 | 596 | 312 |
| 1024 | 631 | 323 |
| per token / per channel only (epilogue) | ~685 | ~323 |

- **Decision / next:** compute is integer (iu8 + iu4), not fp8; W4A8 is dropped. Group scales along K are expensive, so the scales are per token and per output channel.

### 2026-10-30 — · #1 · Two quantizer processes per card (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; INDEX bounds it at ≤10-30 · **Code:** `ourq.py` on exllamav3 1.6.3, gfx1201
- **Result:** +14% throughput only (GPU already at 100%), and memory fragmentation risk (head quants OOMed before).
- **Decision / next:** one process per card.

### 2026-10-30 — · #2 · Bigger tile batches in the quantizer (dead end; V2 era)
- **Status:** null · **Date:** undated in source; ≤10-30
- **Result:** per-tile trellis cost is flat from 64 to 8,704 tiles per launch.
- **Decision / next:** nothing to gain.

### 2026-10-30 — · #3 · Sharing work across K values in one run (dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** the K-independent part (Hessian prep, LDL, Hadamard) is ~5 s of ~35 s per layer, so ≤ 15%.
- **Decision / next:** not worth the restructure.

### 2026-10-30 — · #4 · K7's slow trellis kernel (dead end; V2 era)
- **Status:** dropped (not fixed; the INDEX row currently says done, see Rewrite notes) · **Date:** undated in source; ≤10-30
- **Result:** K7 ~40 µs/tile vs ~15 for K4 / K8. Not fixed: needs an exllamav3 extension rebuild. What did work: dropping the diagnostic `block_trace` (79 -> 35 s/layer, bit-identical); `EXL3_QT_OPTIMIZED=1` (identical indices; K1 2.2x, K6 1.26x).

### 2026-10-30 — · #5 · Head quantization OOM at K5 / K6 (dead end; V2 era)
- **Status:** done · **Date:** undated in source; ≤10-30
- **Result:** `refit_scales` allocated full-size temporaries; fixed by an in-place chunked rewrite (`/data/quantize.py.orig` is the original). Head quant still needs ~22 GB host RAM.

### 2026-10-30 — · #6 · Bigger prefill steps in the eval (dead end; V2 era)
- **Status:** null · **Code:** `radiance-kldgold-fast` · **Date:** undated in source; ≤10-30
- **Result:** `--max-num-batched-tokens 4096` is a no-op (chunks are capped at the 2,048 GDN checkpoint interval); `--prefill-span 2` and `--checkpoint-interval 4096` run 4,096-token chunks but take the same 118 s with the same KLD. 92% of GPU time is the EXL3 GEMMs at ~40% of peak.
- **Decision / next:** the fix is the prefill GEMM rewrite.

### 2026-10-30 — · #7 · rocprofv3 on radiance (dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** unusable: memory-copy tracing breaks the load, and kernel tracing makes the recorded-launch replay fault.
- **Decision / next:** use radiance's own `--profile-ops`.

### 2026-10-30 — · #8 · Card 1 evals ~25% slower (152 vs 119 s) (dead end; V2 era)
- **Status:** done (cause found; fix not made) · **Date:** undated in source; ≤10-30
- **Result:** the full-vocab f32 logits copy over its PCIe3 x4 link.
- **Decision / next:** async copy + background scoring would fix it (~1–1.5 h engine work); not done yet.

### 2026-10-30 — · #9 · TP2 for whole-corpus runs (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** card 1's x4 link makes it slow.
- **Decision / next:** TP2 only for documents that do not fit one card.

### 2026-10-30 — · #11 · Symmetric "depth compounding" term in kfit (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30 · **Code:** kfit.py (cost_w·(1 + c·cost of earlier layers))
- **Result:** algebraically a global quadratic link on the total, not a depth effect: it improved CV only by fixing the overall level, and its solver step biased against late layers. Replaced by the directional model (error amplified by downstream error, kfit2.py): CV log-RMSE 0.28 (additive) -> 0.19 (link forms) -> 0.11–0.13 (directional).

### 2026-10-30 — · #12 · Lagrangian multiple-choice-knapsack sweep (dead end; V2 era)
- **Status:** superseded (exact solver instead) · **Date:** undated in source; ≤10-30
- **Result:** exact only on hull points; lands well below targets (hence the old "3.99" targets); cannot handle the directional term. Pitfall in the replacement DP's first version: 4 MiB buckets rounded *up* per tensor left ~0.1 bpw unused per mix -> 1 MiB, nearest rounding, exact-size check.
- **Decision / next:** replaced by an exact byte-bucket DP.

### 2026-10-30 — · #13 · Treating all quantizer families alike (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** turboderp's "plain" exports cost ~1.15–1.23x the KLD of "SC" ones per unit of Hessian proxy error; the proxy is not quantizer-neutral.
- **Decision / next:** fit a factor per family (and per K range for ours).

### 2026-10-30 — · #14 · Measuring single-tensor swaps end to end (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** paired SE on ~100k positions is ~1.5%; a single tensor's effect is ~0.1–0.5%.
- **Decision / next:** use group swaps, whole mixes, or random multi-tensor perturbation designs.

### 2026-10-30 — · #15 · Noise-injection sensitivity calibration (HIGGS / Q-Palette / exllamav3 style; dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** not cheaper for us: real swaps are 0.2 s reflink mixes, and noise injection would need a bf16 forward that does not fit one card.

### 2026-10-30 — · #16 · Shapley-value interactions (IMPQ, SLQ; dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** literature puts interactions at ~1% for ~15–18 h per 8B model. Skipped.

### 2026-10-30 — · #17 · RCO: Riemannian constrained optimisation (arXiv 2605.00649; dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30 · **Files:** `docs/research-rco.md`
- **Result:** needs a differentiable torch forward + backward of the full 27B on ROCm, 800–3,200 passes per size, for 0–4% over measured-loss search. Skipped.
- **Decision / next:** the "one backward pass = in-context sensitivity of every tensor" idea is kept as an optional spike.

### 2026-10-30 — · #18 · Gradient boosting / generic black-box surrogates (XGBoost, full GP BO; dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** not usable: the exact solver needs a per-tensor-decomposable model, and black-box BO over ~340 categorical dims needed ~10k evals in AMQ.

### 2026-10-30 — · #21 · Layer 0 as a sensitivity spike (dead end; V2 era)
- **Status:** null (the hypothesis was wrong) · **Date:** undated in source; ≤10-30
- **Result:** K2 on layer 0 ffn_down costs +0.00006 (≈ free). The spike is at the end: L63 ffn_gate_up K2 +0.0065 (80% of the base's KLD); L60–63 ffn_down +0.0006–0.0013.

### 2026-10-30 — · #23 · 2-bit Lloyd embedding (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** earlier embedding sweep: clearly bad vs 3-bit at normal sizes.
- **Decision / next:** still offered to the optimizer for the research floor (1.2–1.5 bpw) only.

### 2026-10-30 — · #— · Process pitfall: selecting on public
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** never. Public is held out (`docs/METHOD.md` §6). Verify-slice reuse also bakes in noise slowly.

### 2026-10-30 — · #— · Process pitfall: `pkill -f` / `pgrep -f` with a pattern in our own ssh command
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** killed the shell several times. Anchor patterns.

### 2026-10-30 — · #— · Process pitfall: queue claims by `mkdir` alone on the overlay filesystem
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** claims raced: two cards ran one eval. Claims now take a lock.

### 2026-10-30 — · #— · Process pitfall: `golden_pick.py` with `splitlines()`
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** split documents on Unicode line separators. Split on `\n` only.

### 2026-10-30 — · #— · Process pitfall: fp8 KV cache
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** fp8 KV cache moves KLD. All measurements use bf16 KV.

### 2026-10-30 — · #— · Process pitfall: reference curve including dominated containers
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** a reference curve that includes dominated containers (bigger *and* worse, e.g. plain 2.00 / 4.00 / 5.00 / 6.00) made mixes look better than they were. Compare against the Pareto envelope only.

### 2026-10-29 — · #10 · First knapsack (`knap.py`): band sensitivities from big K=2 swaps, linear model, Lagrangian solver (dead end)
- **Status:** superseded (by kfit2 / knap4, #45)
- **Result:** the K=2 swaps overstate slopes ~4x (only relative values usable); a power-law (P) variant blew up and was dropped; a units bug in the head / embedding cost gave every mix a 3-bit head. Fixed version: ~−3.5% vs the reference containers on calib-B on average, and the 2.00 bpw mix was +68% *worse* (non-additivity at low bpw).

### 2026-10-29 — · #35 · Trellis (EXL3-style) token embedding instead of Lloyd (dead end; deferred)
- **Status:** dropped (deferred on 2026-10-29; the trellis embedding was later measured on real KLD and adopted, see #35)
- **Result:** measured 4-bit Lloyd at ≈ +2% KLD at 4.00 bpw vs IQR's trellis embedding ≈ +0.6%: at most ~1.5% total to win, and it needs a row-wise trellis gather kernel.
- **Decision / next:** deferred; only reconsider for vocabulary-tiered embeddings (simulate both first).

### 2026-10-29 — · #24 · Head above K6 (dead end)
- **Status:** null
- **Result:** our K7 / K8 heads have 5x / 16x less proxy error than a 6-bit head, but the KLD gain over a 6-bit head is within noise (~−5% vs the 5-bit head for all of them).
- **Decision / next:** head bits past K6 are wasted except maybe at 8+ bpw.

## Rewrite notes

*Applied 2026-11-02 (main session): (a) both radiance-port entries → #74, tdverify → #75 (new INDEX rows); (b) corrections 1, 3, 4, 5, 6, 7, 8 applied to INDEX (#39 set blocked, not running: its GPU test is on iceman); 2 (#38) and 9 (#37) left as they are: damp3's result set the V3 damping and finalgate closed the speed pass.*

**Conventions applied.** Times: CT and zone-less times are kept as given and labelled CST. Five entries the source gives in CDT were converted (−1 h): 01:55, 01:50, 01:47 and 01:40 CDT on 2026-11-01, and 01:17 CDT (Gate 2e). `~` is kept where the source is approximate, or where the time is inferred from a neighbouring entry. `—` marks entries with no clock time in the source, including every dead-end bullet. Undated V2-era dead ends, and the six process pitfalls, carry the header date 2026-10-30 (INDEX bounds the dead ends at ≤10-30). Two undated 2026-11-01 log bullets (tdverify, damp2) take the date of the entry above them, with `—`. Within a bullet list, a bullet with its own bold label is one entry, and its plain sub-bullets are folded into it; a sub-bullet with its own date takes that date. A dead end whose bullet spans several dates uses the latest one (per-group gain scales 2026-10-31; YAQA sampled positions 2026-11-01). Where a source entry gives no status word, the status was taken from its text (for example, `running` while a gate is still open). Entries are sorted by date, then time, newest first; entries without a clock time follow that day's timed entries. Numbers: every number in the entry bodies appears verbatim in the two archive files, except INDEX cross-references (`#46`, `#33-08`).

**Topic summaries (11).** E0, E1/E1b, E1c, E2, E3, E3b (tool), E3b results, E3c, K3, E4, E4b. E2 has no heading in the source: its block sits inside the E1c section, so it is written as a summary under #28 and dated at its last log result (19:30 CST). E4b has no result; it is placed at E4's date (`~21:00`). E3 is dated 16:40, its last result in the section (INDEX #29 says last 10-31; the 10-31 full set is in the E3b results summary).

#### (a) Entries marked #? (2)

- **2026-10-31 ~04:00 CST · #? · radiance 1.15.0 / ABI 32 port (planned).** Suggested row: `#?` | radiance 1.15.x plugin port (ABI 32, layout fingerprint) and KLD parity | kernels | planned | 2026-10-31 | plugins rebuilt and re-validated (byte-identical KLD) before V2 ships.
- **2026-11-01 ~17:45 CST · #? · radiance 1.15.1 port works (done).** Same row proposed; one row can cover both entries (the 1.15.1 entry has the parity check: 0.027122 vs 0.027122).
- Optional, not numbered: `eval/capstudy.py` (search slices capped at 2k, held-out 8k) sits inside the #58 entry; D6 (GDN a/b merge closed) and B13 (`ourq.py` code stamp) sit inside the sub-1 K entry. tdverify was given #63 by content; no row covers it.

#### (b) INDEX rows whose verdict or status looks wrong (9)

1. **#39 Global gain scale vs K.** INDEX: `null`, 10-31. Source (10-31 ~22:20 CT): "the best factor rises with K in every family … the GPU gaintest (running) shows whether exllamav3's choice already tracks this. If it doesn't, it is a strong candidate for the high-K gap." Suggest `running`, pending the GPU gaintest.
2. **#38 Damping.** INDEX: `done` (11-01), verdict "input 0.005 (0.05 below K1), rest 0.025". Source: 11-02 "**damp3 (running 2026-11-02):** … Grid 0.025 / 0.01 / 0.005 / 0.0025 / 0.001 / 0.0003 / 0.0001" and 10-31 "0.01 elsewhere is harmless". Suggest `running` (damp3), and reconcile "rest 0.025" with "0.01 elsewhere".
3. **#33 YAQA output-side Hessians.** INDEX: `running`. Source: dead end (11-01) "Every backward-based Hessian variant is dropped (2026-11-01)"; log 10-31 ~22:20 CT "two-sided YAQA dropped". The only open arm is #33-08 (Sketch B). Suggest the parent read `negative` (or `dropped`) with 33-08 as the open part; the dead end also says Sketch B was "not tested".
4. **#34 LM head.** INDEX: `done` (11-01, "mean-row −30%"). Source (11-01 17:45 CST): "Floor search fixes the slab head … whole-model check still to do (critique C3.5)"; (11-01 00:40 CST): "Needs hdslab-K5/5.5/6 containers before validation." Suggest `running` for the slab head; the mean-row result stays `done`.
5. **#46 A4 activations.** INDEX: `negative` (11-01, "no, at any size (knap4)"). Source: "The A4 drop above rests on the matmul-core table, not on this slot" (11-01 01:25 CST), and the A4 timing slot is "queued after D2" (11-01 00:45 CST). Suggest keeping `negative` but stating that the timing slot has not been run.
6. **#22 GPTQ damping.** INDEX: `negative` (10-31). Dead-end bullet: "not a dead end, deferred … not worth a V2 requant; use 0.005 for attention / GDN inputs in the V3 requant." Suggest `superseded` by #38.
7. **#4 K7 trellis kernel.** INDEX: `done`. Dead-end bullet: "not fixed: needs an exllamav3 extension rebuild." Suggest `dropped`, or `done` only with a pointer to #37 (qtf2 integer K6 / K7, 11-02).
8. **#19 Per-layer edge offsets.** INDEX last date `≤10-30`. Source: 10-31 ~01:50 CST entry and dead end "(KFIT_EDGE, 2026-10-31)". Suggest last date 10-31.
9. **#37 qtf2 speed pass.** INDEX: `done` (11-02, "finalgate PASS, 51 instances"). Source (11-02 speed pass, lower confidence): "Not done: reusing K-independent work across levels …; refit via Cholesky …; KA 6-7 lane-cooperative search". Check whether the pass is closed.

#### (c) Source entries not placed, or contradictory

1. **Gate 2e MARGINAL rule** (11-01, date only): listed after the ~00:55 CT knap4 entry, but it must precede the 01:17 CDT verdict. Kept as `2026-11-01 —` (sorted to the end of the day).
2. **qtf2 v1** (11-01 17:45): "still ~2x slower than the integer kernel (K3 43k vs 18.6k tiles/s)". The numbers read backwards; copied as written.
3. **Fractional-K heads** (11-01 01:25 CST) say K3.5 / K4.5 OOM and do not fit. The 17:45 CST entry says K3.5 "work[s] alone" (25.6 GB peak). Both kept; the later one supersedes.
4. **Speed pass 2** bullets sit under the 11-01 17:45 CST container but carry 2026-11-02 dates. They are dated 11-02 with `—`.
5. **chain1** (11-01): results are from 10-31 ~23:50 CT; the entry is dated 11-01 with no clock time.
6. Several 10-30 and 10-31 entries have no time zone; written as CST.
7. The 11-01 ~21:20 CT quarter-step entry and some 10-31 entries are out of order in the source; sorted by timestamp.
8. **Trellis token embedding** (dead end, 10-29, "deferred") was measured and adopted later (#35, 10-31 onward). Its status is kept as of its own entry (`dropped`, deferred).
9. **Per-group gain scales**: dead-end dated 10-30, retested 10-31; dated 10-31.
10. **Cross-references inside multi-topic bullets**: the rounded-codebook bullet (#43) also covers #34 and #45; chain1 (#35) covers #63; the knap4 wiring entry (#45) covers #34 and #35; the 11-01 requant bullet (#58) covers #36, #38 and #48.
11. **Dead ends that restate a log entry** (embedding run5/run6, Best-of-N, upstream auto rule, GPTQ) are kept as separate entries, as the brief asks.
12. **Undated process pitfalls** have no INDEX row and no date in the source; placed at 2026-10-30 (≤ V2 era). This is an assumption.
13. **Not converted:** the tracker header, "Plan to the V3 stream (2026-11-01, user)", "Running / queued now", "Backlog" and "Candidate experiments". They are outside the two sources the brief names. The user decisions in the plan appear in the Log entries where the Log records them.
