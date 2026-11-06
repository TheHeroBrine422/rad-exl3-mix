# Backlog: ideas and plans not yet started

Moved verbatim from the old tracker on 2026-11-02 (sections "Plan to the V3 stream", "Backlog", "Candidate experiments"). Items get a number in `INDEX.md` when they start; then strike them here or delete them.

## Plan to the V3 stream (2026-11-01, user)

Goal: start the eval stream soon after the level ladder finishes (late Nov 2 at the earliest). Phases:
1. **Quants:** the level ladder (running; ~3 h a level a card), head curve (running tonight), slab-head and
   trellis-embedding containers, a code-stamp (B13) deploy.
2. **Kernel / eval optimization: 2-4 passes before the stream** (each eval at the stream's pace pays back many
   times): the V3 int8 path (fold kernel, decode pre-pass vs fused, decided by the A4 timing), eval runner overhead,
   packing. Every pass ends with a byte-exact / KLD-identical check against the previous build.
3. **Quantizer optimization pass(es):** only output-identical speedups (profile one level end to end with a stage
   split first; the sigma sweep triples the input-group time). Pays off for any later requant (V3.5 sub-1 K, heads,
   fixes); changing the quantizer under a running ladder is out unless the shards are bit-identical (B13 stamps).
4. **Solver model, before the stream (CPU, needs no cards):**
   - [ ] **Uncertainty-aware solve (B11):** fit the model on bootstrap resamples; each option pays a penalty
     proportional to how much the fits disagree about it. Stays per-tensor additive (exact solver). Test on V2's data:
     does it avoid the V0-style extrapolation picks (K1 at 6 bpw, 31x misses)?
   - [ ] **Partial pooling** of the per-family / per-K factors (hierarchical kfit2): each factor shrunk toward its
     group unless the data supports it; new V3 terms (fractional levels, slab heads, trellis embedding, int8 mode)
     start from a prior, not a free fit.
5. **During the stream:**
   - [ ] **Measure where the model is unsure:** a fixed share of evals on the options the bootstrap fits disagree on
     most (sub-2 levels, new fractional levels, slab heads), not only near-optimal mixes.
   - [ ] **Residual re-ranker retry (XGBoost etc.) at the halfway point,** pass rule fixed now: beat kfit2 on grouped-CV
     pair ranking by more than the bootstrap noise AND on a held-out search slice; else dropped for good.
   - **Stop rule:** a size band stops when its held-out (ver3) best-so-far stops decreasing (no gain beyond its paired
     SE over N candidates, N fixed before the run). No fixed budget (~1,500-3,000 evals expected).
6. **D3** (re-run the 12 V0 mixes with the current fit on a search slice) as soon as the ladder has its first levels:
   the check of the solver's predicted -8...-21%.
7. Final: ver4 once; public only to report.


## Backlog (rough priority within each group)

### Activations
- [ ] A8 per family (batch2): find the tensors that need A16.
- [ ] A4 g128 / clip (batch3). Then Atom-style outlier channels at A8 within a tensor; full-dim / learned
      (SpinQuant) rotations.
- [x] Activation dump + CPU proxy (E1c): per-tensor costs, ±30% vs KLD with family factors. Next: a depth term.
- [ ] A4 by depth band at fine resolution, if bands differ a lot.
- [ ] Per-tensor clip ratios (searched on fit pool).

### Weight codes
- [ ] **Quarter, continuous and row-by-row K** (2026-10-31): a pattern-driven fractional trellis gives any
  K = a/b (quarter steps 2.25/2.75 in the focus range; dense / continuous menus per tensor). Row-by-row K = per row
  slab (rows ranked by sensitivity), cf. E3c's ~0.05-bit ceiling. Estimated gain beyond half steps: quarter ~1.5%,
  continuous < 1% more; row slabs ~0.05 bit on big tensors. Same machinery as half steps; cost is quant runs.
- [ ] **Fractional / sub-1 K for V3** (2026-10-31): K0, K0.5, K1.5, K2.5 (and other half-steps). K1.5 sits between the K1 and K2 tensors the < 2.5 bpw mixes use most. Half-steps help at every K (~5–6% KLD); once the fractional trellis exists, extra Ks are just quant time (~1 h each) -- order 1.5, 2.5, 3.5, 4.5, then 5.5+.
  - **K2.5 etc.:** a trellis shifting 5 bits per 2 weights. Estimate: half-steps cut the knapsack's discretisation loss
    from ~8% to ~2% of KLD (exponential D(R), rounding to ±0.5 vs ±0.25 bit) → ~5–6% lower KLD in 2.6–4.5 bpw,
    worth ~0.1 bpw.
  - **K0.5:** a 1-bit-per-2-weights trellis. Mainly for the < 2 bpw mixes; they put 51–154 tensors at K1.
  - **K0:** drop a tensor / replace it with a low-rank approximation. Candidates: early ffn_down, where K2 costs
    +0.00006 on layer 0. Testable now with probes (zero / low-rank donor containers).
  - **Cost:** quantizer (exllamav3 kernel), decode kernels, container format. Do with the int4-grid codes.
- [x] Small-sample Viterbi check (E3b): tool built, K4 int8 +1%; full K2–6 sweep queued.
- [ ] Trellis whose codes land on the int4 grid: the W4A4 path at 2–3 bpw.
- [x] Row-slab K (E3c): ceiling ~0.05 bit on the big tensors at 128-row slabs; low priority.
- [ ] Quantizer knobs: damping, column order, re-Viterbi. YAQA A/B on a few layers (low expectations).
- [ ] Full-tensor / full-model quants: **GPU only** (LDLQ + Viterbi).

### Kernels
- [ ] E2 timings (above). Then W8A8 with trellis decode in a pre-pass (T2-style) vs fused.
- [ ] iu4 kernel with g128 scales, if batch3 shows g128 makes A4 viable.
- [ ] Activation quantize kernel fused into the rotation (prep) pass.
- [ ] Grouped-GEMM speed at small M per expert (for Qwen Flash, later).

### Optimizer / frontier
- [ ] **Prime V3 from V2** (2026-10-31).
  - V3's fit starts from V2's final kfit2 model as a prior: per-tensor sensitivities, compounding, embedding /
    head costs.
  - Every V2 measurement enters as an all-f16 point (activation error 0). For a8 at K <= 6 the weights are the same
    (E3b ×1.000–1.004).
  - Fit only the new terms: activation factors, the activation × weight interaction, the int4-grid / int8-aware
    options.
  - Measure V3 mixes via the EXL3 stand-in + hook masks until the kernels land. Start from the best measured V2 mixes.
- [ ] **V3 pipeline: streaming instead of rounds** (2026-10-30).
  - **Loop:** fit runs continuously on a CPU box (CPU box A / CPU box B). Each fit re-solves all targets and replaces the
    not-yet-started mix candidates. The cards pull the freshest candidates plus uncertainty swaps (`uncswaps.py`).
  - **Gain:** a misfit is caught after ~10 evals, not a round, and the cards are never idle between rounds.
  - **Needs:**
    - explicit pairing for same-slice A/Bs (the slice offsets are already fitted);
    - fit→mix provenance for pred-vs-meas tracking;
    - Thompson exploration kept.
- [x] ~~**V3: ours-only option set** (2026-10-30).~~ SUPERSEDED (2026-10-31): design/v3.md §1d keeps turboderp's tensors as options at every K. Drop turboderp's containers as options. V3's K≥5 requant
      with the int8-rounded codebook makes every option ours anyway, since turboderp's quants can't be requantized.
      - Keep their existing measurements in the fit, as data on the per-tensor sensitivities. Those don't depend on
        which quantizer made the tensor.
      - Drop their family factors from the option set.
      - Re-check CV and held-out ver1 after the switch.
      - Round 11's ours-only A/B (V2) gives the first evidence.
      - Later (a day or two in): drop turboderp measurements from the fit too, once our own measurements cover every
        family × depth band. Test: refit without them; CV and the ver1 ranking must not get worse.
- [ ] 3D knapsack: DP over bytes × time (or a Lagrangian on time), per-tensor options
      {W·A16, W·A8, W4A4} × K × activation group size {per-token, 1024, 512, 256, 128, 64} (decided 2026-10-30:
      32 and K/N splits dropped; per-token kept as the fastest option). Linears sharing an input
      (q/k/v, gate_up) are one item with one activation format. Per-tensor activation costs come from the
      activation dump + CPU proxy, scaled by the E1b family KLDs. Develop on CPU against estimated time costs until E2 lands.
- [ ] Frontier display: KLD vs prefill speed, one curve per bpw slice.
- [ ] Speed model: prefill from E2, decode from bytes / bandwidth + dequant.

### Embedding / head
- [ ] Vocab-tiered head (simulate first, multilingual guards).
- [ ] Trellis embedding vs Lloyd at tiered widths (E5).

### QAD (not started)
- [ ] Bit-exact torch reference of the kernel's dequant.
- [ ] Trainable scales/norms, STE fake-quant for A8/A4, teacher top-k logits cached from the qad pool.

### Other models
- [ ] Qwen Flash (MoE): W4A4 / A8 per expert by importance (routing frequency, H_out mass).

---

## Candidate experiments (2026-10-31; queue nothing until the YAQA decision)

Scope: sizes below 6.00 bpw only (6.00 is already 0.00076). Stop exploring when a full round finds nothing worth
more than ~1% KLD at the target sizes; then freeze menu + settings and build.

- [ ] **int8 codebook for K ≥ 5** (planned): write, validate one layer (DESIGN §6 A3), then requant K5–8 (~6–7 h).
- [ ] **Definitive A4 (W4A4) test at < 3 bpw** (user, 2026-11-01): the earlier "A4 almost never chosen" came from
      the activation side (E1c fake-quant); weights on the int4 grid were only CPU-measured (E3: x1.01 K2, x1.07 K3).
      1. full-model int4-codebook quants K2 / 2.5 / 3 (`EXL3_MUL1_I4`, exl3cb i4 build);
      2. a ~2.5–3 bpw mix with those tensors + the E1b hook at A4 g64/g128 on them, vs int8 codebook + A8 and vs f16:
         the real W4A4 KLD cost;
      3. time table with iu4 throughput (E2: ~3.7x f16, g128 −30%) + decode pre-pass -> knap3 at 2–3 bpw;
      4. only if knap3 picks A4: a real iu4 GEMM (nfq_gemm_i4) for an end-to-end timing.
- [ ] **Activation cost on a low-K base** (mining B1/B2): E1b on a K2/K3 base, int path gated at M ≥ 512 vs not (~1 h card).
- [ ] **Speed calibration** (mining A4/B7): key shapes at full/half N, stock/210 W, M = 16–511 (minutes, existing bench).
- [ ] **Best-of-N rotation seeds** (new): pick each tensor's sign/rotation seed from 2–4 by proxy; free at inference.
      CPU test of the per-tensor seed spread first (trellis.py with other seeds), expect 1–3% if anything.
- [x] ~~fp32 costs in the GPU trellis search~~ (2026-11-01): explained, nothing to gain. See log.
- [ ] Depends on tonight: YAQA count test / full collection (gate 2c/2e); gradient-weighted H_in if it wins;
      (dropped 2026-11-01: why turboderp wins at K6–8 -- damping and gain ruled out; the rest only matters ≥ 6 bpw,
      and his tensors stay options in the menu); quarter-step heads if the slab head fails; row-weighted trellis (PR #364) if gate 2d helps.
- [ ] **Head as a slab-plan curve in the solver** (planned): ~33 options, slab plans at 1/8-bit budgets 3–7 bits/row
      (headslab.py), each priced by headsim.py (exact head-only KL, validated vs whole-model head swaps); needs the
      int8-codebook head quants (one per K).
- [ ] **Later, during search/convergence: refine the head locally** (user idea, 2026-11-01): once the stream settles
      on a head budget, search finer around it (1/32-bit budgets, slab boundaries moved by single 128-row blocks,
      half-step K levels inside slabs), priced by headsim. Write after the search starts.
- [ ] Blocked / later: sub-1 K and K0 (needs the salted 0-bit read); fused-GEMM K ties (after the time table);
      radiance #328 check at the 1.15 port.

## Added 2026-11-02

- [ ] **Rebase on radiance 1.16.0** (released 2026-11-02; fixes a small prefix-rounding issue that can move our
  KLDs). Like the 1.15 port (#74): rebuild our plugins against it (exl3r4d, exl3dense, qwen35_exl3, exl3i,
  exl3import; check the plugin ABI version), byte-identical / KLD-parity checks against 1.15.1, then decide whether
  the goldens and the reference KLDs must be regenerated (the fix changes the bf16 path too?) — never compare KLDs
  across engine versions without a paired re-run. Timing: before D3 / the V3 eval stream, after the 27B H_out run
  (don't change the engine under a running comparison). Also move the `qc-exl3-v3` branch / upstream PR onto it.

## 27B confirmation queue (nothing proven on the 0.8B / CPU proxy ships without a 27B check)

The 0.8B / 2B proxies and CPU proxy-error tests decide what is worth a 27B run; they do not size the effect. Each
item below needs a paired 27B test with its own pass rule (and the 27B quant-noise floor, ranked A7, measured once
first). Keep this list current: add an item when a proxy test passes, strike it when the 27B test is logged.

Runtime cost (INDEX legend): every row here is quantizer-only (none) except i8c (P+D?, timing after the HIP build)
and A1 (P+D: an extra GEMM per MLP); those two also need a prefill and a decode timing before they ship.
First try to decide each globally (adopt if clearly better at its cost or the cost is negligible, drop if clearly
worse). Only if a variant is likely on the Pareto frontier for some tensors / sizes, or tests cannot tell, does it
become a **per-tensor knapsack option** (user, 2026-11-03): e.g. i8 / i8c per tensor (both quants at each K; both
decode variants compiled), and per layer down_proj plain / up-predicted / up+gate-predicted (A1 is the likely case:
`ugtrue` ≈ `ug` on the CPU proxy, so the gain barely depends on up's K and the option is close to separable from the
gate_up choice). Each option needs its sensitivity in kfit2 and its time in knap4's speed table; **knap4 also needs a
decode table** (bytes and ALU per token) so decode can be a constraint or a reported axis, since the board scores only
prefill.

| proxy result | 27B test needed | depends on |
|---|---|---|
| Diagonal H_out (Σdy² / Sketch B), 0.8B 3 seeds: K3 −8…−9%, K2 −5…−6% (#33-16, #33-18); holds at K4 −8.7%, K5 −7.5% (#82); all families but the writers carry it (#82) | 27B collection to 2,048 sequences (user, 11-03: the 27B may converge differently; #33-22 saw nothing past 1,024 on the 0.8B, so the 27B's own snapshots decide), pilot swap at K3 / K2.5 vs ladder levels | gpubox back, card 1 |
| Estimator choice (dy vs Sketch B vs xw), 512 seqs (#33-18) | collect all three in one 27B pass, compare in the swap test | same run |
| Cubic int8 codebook i8c (#78: 0.8B KL K3 −1.7%, K2 −2.7%, K4 −2.3%; adopted). **But on top of H_out: no gain at K3 (#78-03, 1.000 / 1.015 / 1.005); #78-04 adds seeds 3–5** | requant 2–3 trunk levels with i8c **with and without H_out**, swap test | gpubox; HIP build of the i8c decode; #78-04 |
| Tile coordinate descent (#77: CPU proxy −0.8%, every tensor; replicates on other rows #80, stacks on cubic #79) | only if judged worth its 2–3× quant time | a GPU version + a seed-paired KL test |
| (#33-19 / #33-20 / #33-21 / #33-22 failed or neutral: none ride along.) Gen-masked loss (#83: Sketch B K3 0.971, K2 0.950, every seed) | `YAQA_GENMASK=1` on the 27B collection (needs gen.u8 beside the calib golden on gpubox) | #83-02 (the same on Σdy², the 27B estimator) |
| Sibling-sequential asym target for the 48 writers (#87: K3 −14%, K2 −18%, every seed) | layer-streamed 27B forward with the quantized siblings (moments over ~1M qad tokens), requant the writers of 2 levels, swap test | #87-02 (stacks on H_out?), #87-04 (K4 / K5); gpubox |
| Fixed-code scale refits C9 α\* (#85: K3 −2.7%, K2 −8.9%) and C10 swj (K3 −3.1%, K2 −8.6%) | refit on existing 27B levels (no requant), fit-slice α\*, swap test | #85-02 (do they stack, on edy?); gpubox |
| Frozen-code scale healing (#89: K2 −38%, K3 on H_out −17%) | S27 healing on the 27B: QAD-class stream | **the user's go-ahead**; #89-02 |
| Faraway A1, down_proj from up / gate (#81: proxy error ×0.83, up-only ×0.87) | GPU proxy → 0.8B KL with load expansion → kernel cost | #81-02 control (energy vs error gap) |

## From IQR-V4 (2026-11-04, `docs/research/iqr-v4.md`)

- [ ] **V3 prefill as IQR does it** (decode once into an int8 fragment plane + radiance's i8a8 prefill tile;
  `docs/research/iqr-v4.md` "Speed"): est. ~4,000–4,300 µs a layer vs V2's ~8,050 and our 5,200 target. Steps: plugin
  API vs `r4d_gemm_8a8_prefill.h`, EXL3 decode-to-int8 pass, timing at the board's shapes on gpubox. Replaces the
  fused-A8 kernel route (#31, blocked by the NFQ crash) if it holds. Runtime: this *is* the speed work (P).
- [ ] **f16 two-pass for 6+ bpw** — correction 11-04: exl3r4d's T2 (fork ae6539f) already decodes once to f16
  fragments from 512 rows (gate_up 8.08 -> 6.74 ms); what is left is a better 16-bit tile than T2's own. (Original:
  decode once per step into an f16 plane + radiance's 16-bit GEMM, same values: KLD unchanged, no requant; speeds up V2 6.0–9.0 and V3's f16 tensors). Board hint: EXL3-V2-9.00 (33 bf16 tensors) 7,331
  vs 8.00's 8,124 µs. Est. ~6,000–7,000 µs a layer vs ~8,100. First measure on gpubox (~1 card-h): bf16 GEMM a layer at
  the board's shapes, our decode-to-f16 and decode-to-int8 passes. Runtime: P (and D for the decode kernels).
- [ ] **Self-calibration on own agentic sessions (0.8B first)** (user, 2026-11-04): self-generated continuations of the
  user's sessions (`xcorp/selfgen.py`, a running server at low priority; teacher logits by a bf16 record pass or the
  `qc-kld-serve` recording build, subagent w14). 0.8B test on the 4070 (0.8B generates its own continuations,
  transformers): (a) healing on raw traces vs self-generated vs calib qad; (b) input Hessians (Σxxᵀ) and Σdy² from
  self-generated text vs calib vs blends, same token counts. Scored on calib search **and** an agentic self-generated
  golden. Decide globally (adopt, blend or drop): not a second ladder in the knapsack. Context: turboderp's SC
  ("self-calibrated") EXL3 tensors needed ~3% fewer bits than his plain ones at equal KL (board, 3–4 bpw; #13: plain
  cost 1.15–1.23× the KL per unit of proxy error); own raw traces as H_out data were +3.5% worse on calib (#33-19).
  Note: stock radiance `--calib-corpus` records input Grams, so 27B H_in on any corpus needs no new tool.
- [ ] **Kernel optimizations from IQR-V4** (prefill P1–P6, decode D1–D4, hash codebook #100): list, status and results
  in `docs/design/kernel-opts.md`.
- [ ] **V3 decode kernels after radiance 1.17** (`docs/research/radiance-1.17.md`): block-per-tile GEMV with K-split
  waves + biased-int fold; barrier-free 16-token tile for 2–16 tokens; gated activation quant in gate_up's epilogue.
  Needs a gfx1201 card (extra RDNA4 boxes or gpubox after the collection). Runtime: D (this is the speed work).
- [ ] **Act order inside the trellis** (+ longer tail-biting runs on FFN): IQR −0–2.3% on uniform bases. For EXL3: permute at 16-column block
  granularity (free at runtime; libquant sorts by descending H diagonal, `tcq_order=1`). CPU proxy
  first (trellis2 with columns permuted by descending H diagonal after the Hadamard), then 0.8B KL. Runtime: none.
- [ ] **bf16 (row, 128) scales on attn_output / ssm_out from K5** (0.010 bpw): IQR −1–6% at 4–6 bpw. CPU proxy at
  K5 / K6, then kernel cost. Runtime: P+D (a scale per 128 in the K loop).
- [ ] **Concentration of the 27B Σdy² diagonal** (IQR: per-token gradient energy is heavy-tailed, their Hessian
  reweightings all lost): top-row share per linear from the #33-24 snapshots; read beside the swap test.
- [ ] Check knap4's V3 menu gives L63 ffn_gate_up its extra width (IQR's one exception; our #21 spike).

## Later, low priority (user, 2026-11-05)
- [ ] **Autoresearch on KV quantization** (user idea 2026-11-06; KV work itself is parked, this is the cheap way to
  push it): once radiance #416 merges, an autoresearch-v3 task whose candidates change KV layouts / transforms
  (whole new KV cache setups the agents propose, e.g. KVarN-style variance-normalized layouts, plus new packed
  layouts, rotation variants, V-mean / V-prediction fits, per-layer recipes) and whose judge is
  `rad-autoquant` / the engine's KL mode on a fixed long corpus vs a bf16-cache reference (KL above the measured noise
  floor, ΔNLL, top-1, drafter acceptance as a limit), plus the kernels' byte-exact tests. Easier than weight quant:
  every candidate is a runtime option (no requant), evals are minutes. Needs the 9070 XTs or idle card time, and
  a quick check with Code 000 so it builds on his work rather than around it.
  Idea corpus first (user 2026-11-06): a digest of recent KV-cache work for the agents to draw on, built with our
  paper pipeline (`local/papers/`: search → fetch → triage → digest, as the IQR / SOTA reads): KV quantization
  (KIVI, KVQuant, SKVQ, GEAR, ZipCache, MiKV, rotation-based KV like QuaRot / SpinQuant KV, KVarN-style
  normalization, TurboQuant-style vector quant), eviction / compression / low-rank / latent KV (MLA-style), sparse
  attention (D-LSA, QSA), and Code 000's #416 / #412 design notes; one-page entries with method, numbers, and what
  fits radiance's paged layouts. Shipped with the task as read-only context.
  **Judge goal (user 2026-11-06): runtime cost and tails, not just mean KL / NLL.** Tiered, so expensive checks only
  hit finalists: (1) gate: byte-exact kernel tests; (2) screen (minutes): on a fixed long corpus vs bf16 cache, mean
  KL above the noise floor, **tails** (p99 / p99.9 per-position KL, worst-document KL, KL by position bin — errors
  that compound with context), ΔNLL, top-1; plus **runtime**: bytes / token, decode tok/s at 32k / 128k / 512k,
  prefill tok/s, extra passes (vpred's second attention pass is a cost); (3) finalists (hours): RULER-style probes
  (multi-key needles, variable tracking) at 128k / 512k as a hard gate, drafter (MTP / DFlash2) acceptance as a limit.
  Score = Pareto position over (bytes / token, speed, quality incl. tails), not one number; anything that fails a tail
  or RULER gate is out regardless of mean KL.
  Not planned: autoresearch on the weight-quant research itself (needs far more hardware; the user stays in the loop
  for decisions that complex).
- [ ] **User-side mix builder** (2026-11-06, "interesting", low priority): publish the ladder levels per tensor plus the
  fitted sensitivities / speed table; a small tool runs the knapsack for any target bpw / VRAM and fetches only the
  chosen tensor shards (download ≈ the model's own size; no GPU, no eval). Predicted quality only (no mix healing,
  no search/verify). Cost on our side: hosting the ladder (~1 TB at 73 levels), versioning it with the format.
  Gets better with a well-trained predictor (user): sensitivities fitted across sizes, compute modes (f16 / A8 / A4)
  and text types (calib, agentic, long context: per-objective fits the user picks from), speed tables per card
  (R9700, 9070 XT, …; volunteer runs can supply them), and the search's verify data folded back in so predictions
  track measured KLD.
- [ ] **Lessons from radiance PR #416 (Code 000's KV quant, 2026-11-06)** for our own work: (1) **MTP / DFlash2 drafter
  acceptance** as a release metric: their best-KL recipe cut drafter acceptance 0.63 → 0.41 (decode 178 → 132 tok/s)
  because KL mode doesn't run speculation; V3's last-layer / head choices need an acceptance check (and maybe a
  knapsack constraint). (2) **Measured engine noise floor:** they recompute bf16 in another prefill order (chunk 64)
  and subtract it; do the same for our 27B goldens. (3) **Non-additive per-layer costs** for KV ("one early layer
  costs as much as all"): their solver verifies on the device and rescales; our knap4 assumes additivity across
  tensors, so add a verify-and-rescale round on the finals (#46-03's additivity check is the first data point).
- [ ] **From the 11-05 mining review** (`local/mine/review-2026-11-05-ideas.md`, triaged 11-06; ideas 1 and 8 are
  #102 / #103): idea 2 layer-split replicates (noise floor from half-mixes, no new quants; build the tool, validate
  against the measured seed pairs before trusting it); idea 4 writers' asym as a fixed-code refit (compare with w37's
  requant path); idea 10 0.8B 3-layer-proxy transfer coefficient (4070, ~1 h); ideas 5 / 6 narrow decode planes,
  fused sibling launches without K ties (→ `docs/design/kernel-opts.md`; idea 5 may make i4 K2–3 worth building
  without A4); idea 7 per-block A8 exceptions for A4 (feed #46-03); idea 9 0.8B stream-policy testbed (before the
  27B search); ideas 3 / 12 head-only refit and last-quarter 27B healing (healing-class: ladder phase); ideas 11, 13,
  14 low priority.
- [ ] **QAD / healing research pass** (user 2026-11-06, backlog): (1) literature read (worker): NVIDIA Model Optimizer
  QAD / Flash-Next recipe, EfficientQAT, PV-tuning, BitDistiller, LLM-QAT, QuIP#/QTIP fine-tuning, trellis /
  codebook-aware healing; what transfers to frozen codes. (2) 0.8B loss variants on frozen-code healing: hidden-state
  terms, TV vs KL, boundary losses, norms trained, token budget curve (where does it saturate). (3) a size-transfer
  point between 0.8B and 27B (9B, staged) to estimate the shrink before the first 27B healing run. Rules before data.
- [ ] **Embedding-specific QAD / healing** (user question 2026-11-06; 0.8B first, after the hidden-state test):
  (a) **true QAT on the embedding only:** rows are independent lookups with no input Hessian, so after each update
  only the touched rows are re-encoded (trellis per row) — latent-weight training with re-quantization is feasible
  here, unlike the trunk; (b) per-row importance from the gradient at the embedding output (diagonal Fisher per
  token row, from a 0.8B backward pass; no new 27B Σdy² collection) for importance-weighted re-encoding; (c) rows
  of tokens absent from the training data get no signal: keep them at the static quant (#32: rare rows matter, so
  never let them drift), and make the data broad (#101's xcorp); (d) split each row's error into direction vs scale
  (layer 0 starts with RMSNorm, but the raw embedding also rides the residual stream) and test scale-only healing vs
  direction re-encode. Rule written before data.
- [ ] **Library QAD vs mix QAD** (user question 2026-11-06): train a piece (embedding level, head level, one trunk
  layer's levels) with everything else bf16, so it slots into any mix like a ladder level (amortized over every size;
  knapsack sees healed sensitivities); then optionally a short per-mix pass that compensates neighbours' errors.
  0.8B checks: additivity (healed pieces combined vs the sum of their single gains, as #46-03's ±25% rule) and how
  much a per-mix pass adds on top. First candidate: the embedding menu (idea above).
  Scope (user 2026-11-06): ship-side mostly embedding / head; test trunk block-wise too on the 0.8B (cheap).
  Post-quantization (codes frozen), so it doesn't block the freeze, but it goes before the knapsack / search.
  Caveat: 0.8B, 2B and 4B tie embedding and head (`tie_word_embeddings: true`); Qwen3.5-9B (4096 × 32 layers) and the
  27B don't. Plan: trunk block-wise on the 0.8B; embedding and head on the **9B**, on the 4070 via staging (embedding:
  forward/backward through the first group only, hidden-state target at its end; head: train on precomputed pre-head
  states from a mkgold27-style staged pass; both ~2 GB tables). Needs: 9B bf16 download (~18 GB), its calib golden +
  lm_head H_in (mkgold27 / mkhin27, staged), embedding / head levels from our quantizers. Prior guesses (to check, not
  to plan on): trunk block-wise ½–⅔ of mix healing on the 0.8B (EfficientQAT-style: block-wise stage takes most),
  less on the 27B (block-output methods shrank to ~⅓ there); head a few % of total KLD at ≤ 2.5 bpw; embedding
  smaller, mostly at ≤ 2 bpw.
- [ ] **Hidden-state matching for healing / V4 QAD, 0.8B first** (user 2026-11-06; NVIDIA's Flash-Next QAD uses it,
  ROADMAP V4): add a per-layer loss between student and bf16 teacher residual streams at block ends, next to the token
  loss. Layers: the ends of the 4-layer groups (27B: 7, 15, …, 63, always incl. the last; 0.8B: its group ends), each
  term normalized by the teacher's norm at that layer. Storage: full bf16 at healing scale; past ~20M tokens a fixed
  random projection to ~512 dims per layer (≈ 10× smaller, as Sketch B). 0.8B test: frozen-code healing (#89 recipe)
  with vs without the hidden-state term, and full vs projected targets, K3 / K2 × 2–3 seeds, paired on calib search,
  verify on calib verify; also try the TV loss and end-of-thinking / end-of-response boundary terms. Rule written
  before data. 27B only if the 0.8B gain is clear (healing is deferred to the ladder phase anyway).
- [ ] **Long-context eval for our quants: RULER probes** (user 2026-11-05; same probes as the D-LSA gate,
  `docs/design/lsa.md`). Our KLD goldens cap documents at 8k (32k for the 27B trace golden), so nothing checks the
  quants at the contexts people run (256k–750k, fp8 KV, YaRN). One shared harness: RULER-style tasks (multi-key /
  multi-value needles with distractors, variable tracking, aggregation), exact-match scoring, no LLM judge; synthetic
  text, so no pool rules apply. Use: release check for V3 finals (and a V2 baseline at 3.0 / 4.0), bf16 vs fp8 KV,
  quant vs bf16 where bf16 fits (≤ ~128k on TP2, else quant vs the next-larger quant), and the LSA gate. Card cost is
  prefill-bound (a 512k sample is minutes): a few samples per task and length, reported with binomial intervals.
- [ ] **Kernel autoresearch on other radiance kernels** (user 2026-11-05, side note): once the autoresearch-v3 kernel
  task (w35) works on our A8 path, point it at upstream radiance kernels too (e.g. `libr4d` attention, GDN chunk
  scan, i8a8 / fp8 GEMM tiles, norms, sampler); each problem needs radiance's own tests as the byte-exact gate plus a
  bench. Also the D-LSA kernels once w31 / w32 pass their tests (user 2026-11-05): `attn_paged_list` (decode, then the
  phase-2 sparse prefill) and `lsa_select` (centroid scan + top-k expansion), gated by their tests vs the dense op and
  the LSA CPU oracle, benched at 128k / 512k with fp8 KV. Code 000 also has a working LSA (2026-11-05, unreviewed):
  once public, seed the problem with both implementations (ours and theirs) as starting candidates, same judge, and
  hold our phase-2 prefill until theirs is out. Wins go upstream as PRs. Bench card: the 9070 XT when it arrives. GEPA considered and not used (user).
- [ ] **saber-engine read for V3's own kernels** (`/workspace/references/saber-engine`, Apache-2.0, gfx1201; user
  2026-11-05): one short worker read, like the IQR-V4 kernel read, results into `docs/design/kernel-opts.md`. Ideas,
  not code (weight decode differs: their NVFP4 lookup vs our trellis):
  - **FP4 prefill GEMM** (`crates/saber-kernels/hip/mxfp4_gemm_*`, `mxfp4_atiled_*.h`): decode inside the tile, fp8
    WMMA, since RDNA4 has no native FP4 = the same problem as our P1 (fused decode). Read before writing P1.
  - **Decode GEMV variants** (`mxfp4_gemm_decode_{pf,arw,arnq,silu,smq}.hip`, `saber_device_row_{split,ilp}.hip`):
    row split, prefetch, fused SiLU epilogue: patterns for D1 (split-K GEMV).
  - **Fused norm + quantize:** check whether radiance's `rmsnorm_quant_fp8` fold already covers our A8 activation
    quant (~8% of prefill); the all-reduce part is TP2-only.
  Everything else in saber is engine-level and left to upstream radiance work (user: someone else may do it): tree
  speculative decoding with linear-attention state replay (`p27_tree_*`, `p27w2_gdn_replay*`; likely its biggest
  decode win), attention kernels (beyond the slot-list piece used for D-LSA, `docs/design/lsa.md`), linear-attention
  chunk scan, sampler / head kernels, TP2 all-reduce, prefix caching. We take those when they land upstream.
- [ ] **A calib-like tools / agentic corpus** (d1110 per-category, 2026-11-05): our methods gain most on calib's agentic text
  (H_out −19%, stack −35% at K3) but least on xcorp's tools / agentic text (H_out ~0, stack −18…+4%). Find or make agentic
  data that behaves like calib's (the 27B's own agent sessions, calib's function-calling sources) for screening and
  real-use checks. Not now (user).
- [ ] **Per-head K on ssm_qkv / attn_qg** (#96 side result: the Σdy²-weighted spread sits between GDN heads; ceiling
  ≈ 0.04 bpw overall). Needs a K per 128-row slab in the format and kernels: only with other per-slab work.
- [ ] **Volunteer compute v0 (parked, user 2026-11-05)** (moved to its own repo `/workspace/radquant` on 2026-11-06, history kept; w13 + w13b; tentative name **rad-quant@home**
  for display, **`radquant`** in identifiers: images, packages, CLI, hosts; points maybe "lumens"): queue server (stdlib Python,
  container, 61/61 noop e2e tests, restart / crash / partial-upload safe), worker image spec (ROCm, gfx1201), job
  types quant_layer / quant_head / eval_mix. Resumes when the V3 settings are frozen and the ladder starts. Two uses: ladder quant jobs
  and search evals (`eval_mix`, ~50–100 card-hours a stream pass; public slices only, the private-text share of
  accept / reject stays on our cards; needs local mix assembly from cached levels, not per-candidate downloads; 16 GB
  cards only for the lower sizes). Open:
  Docker Hub namespace; build the worker image and run one real job on an R9700 before publishing; where the
  server's volumes live on the queue host; admin = one read-only status endpoint. Donors (user, 2026-11-05): many R9700s (= our card:
  every search size fits; results not bit-identical across GPUs, so w13's tolerance check applies; cheap spot checks),
  hosts mostly ≥ 32 GB RAM: workers already declare RAM (`--ram-gb`); add a minimum per job type (trunk layer ~24 GB, LM head ~48 GB: its ~22 GB
  peak is tight on a 32 GB desktop). With a handful of R9700 donors ladder + search ≈ 2 days: start during the freeze.
  **Big-box tier** (4x R9700, 128 GB VRAM / 192 GB RAM builds in the community): 27B bf16 reference goldens (public data;
  first candidate, on the critical path), full-model backward passes (Σdy² collections on public data, REAL-Q's Fisher),
  fuller H_out accumulators, long-context checks, later healing / QAD and MoE. Trusted donors only (long jobs are not
  cheaply spot-checked: split-half and per-layer comparisons against our snapshots); never the user's traces; a box
  registers as one multi-GPU worker for these and as single-card workers otherwise.
  **Multi-card worker (v1):** one worker with N GPU slots (per-card processes, one shared cache and RAM budget), gang
  jobs claim all cards after a drain, with a server-side reservation so they do not starve; servers get a
  "cards needed" field per job type. Until then: one container per card.
- [ ] **Quantized DFlash2 drafter and vision tower.** V3 bundles both in bf16 (`docs/design/v3.md` §8; outside the
  bpw count). Quantizing them shrinks the container / VRAM: the drafter's quality shows as acceptance rate (decode
  tok/s), not KLD, so it needs its own metric (acceptance length on held-out text); the vision tower needs an image
  eval (none in our corpora). Runtime: D (drafter acceptance) / VRAM.

## Fuller output Hessians for the 27B (2026-11-04, user question)

- [ ] **"Mixed" H_out** (full output factors except gate / up, which stay diagonal): 0.8B at 8K seqs K3 −9.9% vs
  diagonal −8.0% (≈ +2%), K2 equal (#33-12); full gate / up hurts at K2 (noisy). Cost at 27B: ≥ 4K seqs (2× #33-24 or
  more) and ~25–51 GB of fp32 accumulators (disk-backed, or layer subsets per run / per volunteer worker). Only if the
  27B diagonal swap test (H27.md) shows H_out clearly helps; a good volunteer-compute job (additive sums).

## Collector speed (2026-11-04, user question)

- [ ] **Two-sequence pipeline for the 27B collector** (collect.py): layers split 32 / 32, one sequence at a time leaves
  each card idle ~half of every 37 s sequence. Two sequences in flight (1F1B) ≈ 1.5–1.7×, not 2× (bubbles). Blockers:
  pinned host buffers per in-flight sequence (YAQA_XOFF=bf16 ~9 GB each; 30 GB host), per-sequence hook state, test
  needs two devices (0.8B on gpubox). Worth it only for another collection; not during #33-24.
- [ ] **Less recompute in the backward** (26 s bwd vs 10 s fwd): keep activations of a few layers per card in the
  ~3.5 GB headroom instead of checkpointing them; est. −10–20% per sequence.

## NVIDIA Model Optimizer overlap (2026-11-03, `docs/research/modelopt.md`)

- [x] **Saturation check of the knapsack** (done 11-03: null, errors compound instead; LOG) (CPU ~1 h)
- [ ] **Saturation at 0.5–2 bpw** (user, 11-03): the check above had only 13 mixes below 1.6 bpw (the r9 low-bpw mix
  files live on gpubox), and kfit2 + saturation fitted C ≈ 1.8, i.e. it would bite once whole-model KLD nears ~0.5
  (≲ 1.75 bpw, and V3.5's sub-1 K). Redo with the gpubox mix files, plus 6–10 new measurements at 0.5–1.5 bpw (V3.5 menu).: does c(1 − exp(−Σb)), b = −log(1 − ΔKL/c), predict measured mix
  KL better than ΣΔKL (ladder levels, V2 mixes, kfit2 swaps), esp. at 1.75–2.5 bpw? If yes, re-price the low V3 sizes.
- [ ] **Reference KLD of `nvidia/Qwen3.8-27B-NVFP4`** (gpubox, weight-only dequant first) on the verify slice: place it
  on our frontier next to V2 5.0 / 5.5.
- [ ] **Σdy² as the knapsack price** = their AutoQuantize gradient score: merge w08-hprice once the 27B diagonal exists.
- [ ] **#87 at 27B via a layerwise loop** (their single-GPU PTQ design: boundary activations on disk, one layer live).
- Healing (#89 → S27): their scale-only learning results (more consistent than QAD; lr 1e-4, ≤ 100 steps) are the
  starting settings, when the user approves.

## Faraway lens (2026-11-02, `local/papers/runs/2026-11-02-faraway/ideas/faraway.md`): ideas to test later

- [ ] **A1 · down_proj predicted from its neuron's decoded up + gate rows, residual coded** ("chroma from luma"):
  measured 0.09–0.23 bit on down_proj (mean ~0.16; up↔down |cos| 0.27–0.47), est. −3 to −8% KL at 2.5 bpw, P ~0.4.
  **Constraint (challenge author, 2026-11-02): the target is VRAM, load-time expansion only as a stopgap** → it
  needs the online-prediction kernel (faraway A2: transposed GEMMs re-reading the up / gate tiles, +1 GEMM with up
  only keeps ~75% of the gain; +1–2 GEMMs in the MLP). Weigh the gain against that speed cost. Steps: (1) real-trellis
  proxy test on gpubox's GPU (quantize D vs R = D − αÛ − βĜ at K2 / K3 from the ladder levels; pass ≤ 0.85× at K2 on
  layer 30); (2) 0.8B KL with load-time expansion (allowed for measuring); (3) kernel cost (ntbench); then decide.
  Riders: A3 (GDN out_proj ← v), A4 (o_proj ← v / q-gate), A6 / A7 census (minutes on cpu1).
- [x] **E10 · dead-unit census** (done 11-03: null, LOG) (E[h²] at down / out_proj / o_proj inputs from the calib Hessians; cpu1 minutes).
- [ ] **F2 · boundary diagonal affine repair** (closed form, 0.8B ~2 h). **F4 · exact last-layer output Hessian.**
- Faraway's headroom audit: gate/up within ~0.1 bit of an ideal LDLQ + Gaussian coder; weights near-Gaussian; no
  cross-layer redundancy (kills cross-layer sharing / global-basis ideas). The rest of its list: the report.

## Proxy negatives to reconsider at 27B (scale-dependent mechanisms)

A 0.8B / 2B negative is final only if nothing about the idea depends on scale. Reconsider at 27B (or 9B) when the
mechanism is one of: activation outliers / massive activations (grow with model size: A8 / int8-activation and
rotation choices), depth (64 vs 24 layers: error propagation, sequential / output-aware quantization, healing),
matrix size (5120 × 17408: estimation noise, dead units, VQ / lattice gains), the untied 27B head (0.8B is tied), or
the near-lossless regime (0.8B at K3 has KL ~0.09, like the 27B near 2.0–2.25 bpw: high-bpw-only effects are
under-tested). Add a row when a proxy test fails for a reason on this list.

| proxy negative | scale mechanism | 27B check |
|---|---|---|
| #88 activation-denoising target (C1): K3 +3%, K2 +10% with level-matched noise | depth (64 layers: upstream error grows), error propagation | only if #87 / #88-02 show a depth trend |
| #33-22 more H_out windows past 1,024 do not help (0.8B) | matrix size (5120 × 17408 diagonals are estimated from the same windows: noisier at 27B) | the 27B collection's own 512 / 1,024 snapshots (split-half agreement) |
| #78-03 i8c gives nothing on top of H_out (0.8B K3) | matrix size / near-lossless regime (codebook-shape gains vs error-weighting overlap may differ) | the i8c 27B check with an H_out arm |
- [ ] **o_proj / GDN out_proj head-permutation act order (from #103, 11-06):** their 128-block trace spread (p90/p10
  2.83 / 2.58) is higher than down_proj's, where block act order gave −0.8%; CPU proxy first (`v3/e3/actblk/`).
- [ ] **rad-quant-at-home for big MoE models (user 11-06; V4 era, after V3):** donors with TP2 / TP4 + MoE expert
  streaming generate (1) bf16 golden logits (forward only; verify by KLD vs a verifier's recompute of sampled docs),
  (2) input Hessians (additive over documents: split by doc shard × layer group; upload size is the limit — per-expert
  Hessians of a ~100B MoE are ~0.2+ TB per full set; accumulate many docs per donor, bf16 / compressed storage),
  (3) YAQA output-side Hessians only as diagonals / sketches (full Kronecker factors need backward passes and huge
  uploads; and H_out's gain shrank from 0.8B to 27B, so price it first). Targets: Qwen Flash-Next, GLM Flash 5.3,
  DeepSeek V4.1 Flash.
- [ ] **Larger / broader calibration corpora for larger models (user 11-06):** for future models, not V3.
- [ ] **rad-quant-at-home pulls source data (bf16 shards, published Hessians) from HF, pinned by revision hash
  (user 11-06):** no upload load on our hosts; donors and auditors can check inputs against HF. Our own derived inputs
  (e.g. H_out Hessians if adopted) need an HF dataset of ours (outward-facing: user approves).
  - DeepSeek V4.1 Flash (user 11-06): ~500 GB, ~700B params of which ~200B "engram", released FP8 trunk / FP4 experts
    (no bf16): golden logits / Hessians come from the native checkpoint and KLD is vs it; experts go FP4 → ~2–3 bit.
    Engram, if lookup-table memory: quantize like our embedding (per-row frequency weighting, V2 Lloyd / embtrellis /
    vocab tiers), lives on SSD (user: so extreme quants matter little; mild / none is fine). Expert Hessians are thin per expert on a calib-sized
    corpus (→ larger corpus). Verify the engram structure from the model card / code before designing.
