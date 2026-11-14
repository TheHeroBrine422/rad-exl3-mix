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
- [x] ~~A8 per family (batch2): find the tensors that need A16.~~ CLOSED 11-09 (int8-only decision, user + deadcode:
      no A16 anywhere — LOG 12:17 PM CST; outlier work at A8 stays open below).
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
| Per-channel output bias β = (W − Ŵ) μ on every linear, fixed codes (#120-03: plain 0.8B K3 −1.1%, 3/3 seeds z ≈ −10; K2 seed-dependent 1.000 / 0.975 / 0.992). Runtime P+D (an add per linear unless fused; 0.005 bpw) | first the 0.8B check on the **adopted** stack (writers / readers / all, μ bf16 vs quantized; `v3/e3/settle/w91/exl3i-fields.md` check 1), then one 27B fixed-code pair on two seeds with a dev-build bias + prefill / decode timing | engine: optional `blk.N.<w>.bias` weight in qwen35_exl3 (no exl3i change; post-freeze) |

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
- [ ] **Volunteer compute v0 (parked, user 2026-11-05)** (moved to its own repo `/workspace/rad-quant-at-home` on 2026-11-06, history kept; w13 + w13b; tentative name **rad-quant@home**
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
- [ ] **QAD family as a cost ladder (user + lead 11-07, framing for the QAD research pass):** (1) local closed-form /
  one-pass fits at nonlinear junctions from bf16 inputs (swj, w60's q/k, v/o, GDN candidates): mix-independent =
  library, one pass per (layer, K), donor-computable, priced in the knapsack; (2) library healing / QAD per level
  (block-local training, weakly mix-dependent); (3) mix QAD end to end (cross-layer compensation, per mix). Push as much
  as possible into (1)–(2); measure what (3) still adds on top of the full library stack.
  - **MoE reference (user 11-07): NatScale, "Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE
    Quantization", arXiv 2610.20353.** Per-expert bit allocation under a byte budget: gradient sensitivity per expert,
    power-transformed so its contrast (Jeffreys divergence from uniform) hits τ ≈ 1 nat, × joint gate/up/down error on
    replay inputs × AlphaQ's spectral prior, then LP + rounding. Routed experts only, 2.0 / 2.5 / 3.0 bpw; models include
    Qwen3.8-Flash-Next and DeepSeek-V4-Flash. Best KLD / top-1 in 9/9 settings vs MxMoE / AlphaQ / MODE / GEMQ (Flash-Next
    mid budget KLD 0.249 → 0.194); beats UD and GSQ-RCO at matched size. Calib: 1M tokens, stats reused across budgets.
    For us: their allocator ≈ our knapsack with an expert-level prior; the τ "contrast" knob = how hard sensitivities
    are trusted (our kfit2 calibration analogue). Read in full when V4 MoE work starts.

- **Trained 2^15 LDS codebook (user 11-07: not pursued).** A 32 KiB int8 table in shared memory instead of the computed
  mul1 code. Estimated decode cost (unmeasured): ~0 to −5% at K5+, −5 to −10% at K4, −10 to −30% at K2–3 (random LDS
  reads with bank conflicts, the per-workgroup table fill, occupancy), for an uncertain gain of a few percent (QTIP's
  HYB). Not worth it at the low bpw where it would cost most. Reopen only with a V4 code redesign (see #113's bound).

- **Session archive as training contexts (user 11-07).** `/workspace/pi-sessions` (~27M tokens of agentic trajectories
  in main / subagent / worker sessions, growing a few M a day) as *contexts* for (a) a speculative-decoding draft for
  Qwen3.8-27B and (b) QAD for real-world (non-challenge) releases. Targets always come from the bf16 Qwen itself
  (regenerated continuations / its logits), never from the logged Claude text. Needs: rendering into Qwen's chat and
  tool-call template, a secrets / host scrub, dedup of miner prompts. Allowed for challenge entries too (AGENTS: pi-sessions
  are a training corpus since 11-02, xcorp since 11-04); never for selection or verification, public stays held out.
  **Spec-decode plan (user 11-07): SC-style data.** Split the rendered session documents (`xcorp/sessdocs.py` →
  `sessdedup.py`, `local/sessdocs/`) at the end of each real user turn (after the user message / tool responses, where
  the assistant would answer), and have bf16 Qwen3.8-27B generate the assistant response itself (its own reasoning +
  answer / tool call), possibly in chunks for long contexts (prefix cap, continue from the previous generation). The
  draft model then trains on real Qwen reasoning over real agentic contexts, never on the logged Claude text. Open:
  sampling settings (match deployment), how many responses per context, which turns (all vs the deduped ones), tool
  calls the logged environment never answered (stop at the first tool call, or splice the logged tool result).

### V4 QAD: alternating block-wise / end-to-end (user discussion 2026-11-08)
- **This refines the 11-07 "QAD family as a cost ladder" entry below** (its steps (1) library closed-form fits,
  (2) per-level library healing / block QAD *before the knapsack*, (3) per-mix end-to-end): the notes here are its
  design detail. Step (2) must use clean bf16 inputs to stay mix-independent (#111).
- **Idea:** block coordinate descent instead of full end-to-end QAD (full 27B QAD ≈ 350–450 GB VRAM with Adam;
  ~170–200 GB with 8-bit states + offload: rental only). Round = (1) block-wise: train each layer's weights (codes
  re-searched) to match the bf16 layer output, fed the *student's* inputs (asym for every tensor) and weighted by the
  current model's Σdy² / Fisher (H_out); (2) end-to-end scales / norms on KL (our healing); (3) refresh the weights from
  the healed model, repeat. Fits one R9700 (one layer ≈ 420M params ≈ 5 GB with Adam); ~hours–1 day + one requant +
  one heal per round per size. Our V3 pipeline (H_out + asym + healing) is round 1 in miniature.
- **Test (0.8B, 4070):** full end-to-end QAD fits the 4070 (~10 GB, 8-bit optimizer) = the measured ceiling; compare
  healing only, block-wise + healing ×1, ×2. Ask: how much of the full-QAD gain do 1–2 rounds capture?
- Literature anchors: EfficientQAT (Block-AP → E2E-QP), BRECQ, CBQ. After V3.
- **Hybrid form (preferred, 11-08):** block-wise training learns each tensor's *target* (continuous, nonlinear,
  student inputs, Σdy²-weighted), then our exact trellis search (LDLQ / Viterbi) quantizes against it, then end-to-end
  healing; repeat. = asym generalized (learned nonlinear target, every tensor) while keeping our strong discrete
  search instead of STE through trellis codes. Order for V4: asym ⇄ heal (cheap) → hybrid; 0.8B arms vs the full-QAD
  ceiling: heal only · asym ⇄ heal ×1/×2 · hybrid ×1/×2.
- **Schedule (11-08):** round 0 = asym → heal (cheap; V3 if #105 passes; asym is only a warm start once the hybrid
  runs, since the hybrid covers the writers); round 1+ = hybrid block-wise → heal, repeated while the 0.8B shows gains.
  Always end on a heal (code re-search discards healed scales). Variants to test: **sequential** block-wise (one
  front-to-back sweep on refitted upstream outputs, GPTQ order) and **selective** block-wise (only the most sensitive
  layers by the knapsack's prices; heal the rest).
- **Per level vs per mix (11-08, the #111 lesson):** split QAD by context dependence. *Per level* (once, in the
  ladder, reused by every mix): block-wise training of each tensor at each K on **clean bf16 inputs**, Σdy²-weighted =
  better options for the knapsack (student inputs here would repeat #111's uniform-K mistargeting). *Per mix* (after
  the knapsack): the context-dependent part = asym or a sequential block-wise sweep on student inputs, then heal.
  Re-measure sensitivities after per-level QAD before the knapsack. Interaction with V3 gains: format / knapsack kept;
  swj / C9 mostly absorbed by healing (warm start); H_out / damping / tile CD partly kept; asym absorbed by the hybrid.
  0.8B arm to add: per-level block-wise (clean) → mixed-K mix → heal, vs the per-mix variants.
- **Per-mix stage screen (user 11-08):** before choosing the V3/V4 per-mix post-processing, a 0.8B factorial in a
  mixed-K mix: full 2³ factorial of {swj/C9 re-fit, per-mix asym, sequential block-wise}, each on/off,
  fixed order, always ending with heal (8 arms; main effects + interactions); 2 seeds, low + mid size; repeated rounds
  for the best stack only; winner vs heal-only on one 27B low-bpw mix.
- **Low-rank (LoRA-style) correction (user 11-08):** (1) *merged*: low-rank delta learned in block-wise / healing,
  folded into the tensor's target, then requantized with our trellis search (no runtime cost; = the hybrid's target
  parameterization; part of the gain lost in requant). (2) *runtime adapter* (quantized base + rank-r residual,
  LQER / QERA-style): bytes ~3% of a 2-bit gate_proj at rank 16, plus a small GEMV a tensor at decode (launch overhead
  unless fused). Literature: pays mainly ≤ ~3 bits. Order: (1) first; (2) only if a gap remains at 1.0–2.0 bpw, and then
  as an **equal-bytes test** (it is trained per mix, after the knapsack, so the knapsack cannot price it): knapsack
  at S − adapter bytes → mix → train adapter (with / after heal) vs the plain full-S mix healed the same way; KLD and
  decode speed; ranks / sensitive-tensors-only as extra arms. Only a per-level (mix-independent) adapter could be a
  knapsack option. V4.
  **Prefill cost (estimate 11-08):** FLOPs ~0.4% (rank 16 on gate_proj: r(in+out)/(in·out)); unfused it re-reads X
  (~21 MB at M 2048) and read-modify-writes the output (~140 MB) ≈ 0.25 ms vs ~1.8 ms for the int8 GEMM → **~10–15%
  slower per tensor** (rank-16 GEMMs also run inefficiently); fused (X·A from the main GEMM's loaded X tile, ·B in the
  epilogue) ~1–3%, needs kernel work in the f16 and A8 paths. Viable only fused; the equal-bytes test must measure
  prefill gemm_us as well as decode.
- **Embedding Fisher at h0 (#119, w83, 11-08; GPU ~1–2 card-h):** one backward pass on fit-pool sequences with sampled
  labels (collect.py), recording at the embedding output the full 5120² Σ g gᵀ and Σ‖g‖² + count per input token id.
  Decides the embedding's quantizer metric (CPU `v3/e3/emb2/embldlq.py` with H = F_full; worth −0.3…−4.4% KLD at
  1.0–2.0 bpw depending on the metric, free at runtime) and gives measured per-row sensitivities for a floored row
  allocation (IQR-V4's head rule on the embedding; per-run K table in embed_tq if it passes). KLD swap with the
  multilingual guard before anything ships.
- **Low-budget slab heads in knap5 (#119 / #34, 11-08; CPU minutes):** budgets 2.125–3.5 with the floor ~1 bit below
  (`v3/yaqa/data/lcurve.json`; reject plans whose guard KL exceeds the uniform head's) as options at 1.0–2.0 bpw; today
  only K5–6 slab heads are offered and the head sits at uniform K3 at 2.0. Est. −2…−3% at 2.0 bpw; gated by #34-02.


### Statistics collection for future models (user discussion 2026-11-08; not needed for the 27B: both exist)
- Measured on the 27B: input Hessians (forward only, `mkhin27`, staged, one card) ~240 tok/s → ~12 card-h for
  ~10M tokens; Σdy² diagonal (forward + backward, fp32, gen-masked, #33-24) 1,077 min on both cards ≈ 36 card-h, with
  the layers split across the cards so each card idled ~50%.
- Cheaper for the next model: (1) **one combined run**: the backward pass needs the forward anyway, so accumulate H_in
  = Σxxᵀ from the same forward (extra ≈ the xxᵀ GEMMs; host-RAM accumulators for the 17408² down Hessians, ~1.2 GB each,
  are the constraint: stage them or keep them on the host, 64 GB now); (2) **pipeline the two cards** (micro-batches:
  card 0 runs sequence i+1 while card 1 runs sequence i) instead of plain layer-split: ~2× on Σdy²; (3) install the
  real `causal_conv1d` (both collections fall back to the reference GDN conv); (4) stop Σdy² at the convergence point
  (#33-25 suggests ~half the sequences). Estimate: both statistics for a dense 27B in ~10–12 h on two cards, vs
  ~18 h + ~6 h today.
- **User 11-08: for future models (V4 MoE etc.) the default is H_in only, no Σdy² / H_out**: ~2× the statistics
  time for ~−3% KLD is not worth it; revisit only if the collection gets cheap or a model shows a much larger gain.


But what I will add to my lists of things to try is seeing if a lower quality but lower overhead int4/int8 quant can be viable. But that will need to be post very heavy trellis kernel optimization to confirm it’s as good as it’s gonna get.

- **Lessons from the DeepSeek-V4.1-Flash-1017 calibration set (user, `Hero422/deepseek-v4.1-flash-1017-quant-calib`,
  11-12):** 100M-token corpus with pools sized per use (the 27B's ~10M calib starved its search pool by week 2); a
  published bf16 noise floor shipped with the goldens (matters for large MoEs with sparse attention, where reruns
  flip routing and the floor is ~0.01; dense models like the 27B sit near 1e-5, user 11-12); score model-written rows only
  (`gen.u8`): tool output is fragile even for bf16 (batch layout moved trace KLD 0.46–0.94 on tool rows) — apply the same
  mask strictly in xcorp slices; Horvitz–Thompson expert sampling with the raw dumps kept so expert statistics can be
  rebuilt with any weighting; separate whole long traces for eval only.

- **DeepSeek-V4.1-Flash-1017 public teacher logits (user, 11-12):** `brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits`
  (MIT; rev 68379d81). Full-vocab fp32 logits from the official release (reference `inference/model.py`, MP4 on B200,
  FP4 experts as released; window 0 recomputed bit-identically each session): 128 windows × 8,192 tokens (~1.05M tokens,
  542 GB) + the same tokens as 512 fresh 2,048-token cuts (542 GB). Mix: agentic 19, SWE 12, math 19, science 12, code
  12, chat / IF 18, multilingual 12, long-context 6, safety 6, web 6, legal 6 (Nemotron SFT sets + nebius SWE; 13-gram
  screened against their build data). **Usable as QAD training data** (full-vocab targets from the official reference
  implementation are ideal KD targets, and ~1M tokens is a real amount for scale-only QAD). The trade-off: it is the one
  *independent* eval reference for this model (a third party's harness, the reference for their TrellisMX-K2 / K3
  builds, a trellis quant of the same model); training on it spends that. Decide per use: train on it if teacher data
  is short, or keep it held out as the cross-check of our goldens / noise floor and our numbers vs theirs. Our own 100M
  calib set (teacher top-k computed by us) is the bulk training side either way. Storage: 1.08 TB for ~1M positions;
  keep top-k on our side (full-vocab fp32 is ~0.5 MB a position).
- [ ] **DeepSeek benchmark vs TrellisMX on their public reference (user 11-12: "a nice benchmark").** For whoever runs
  the DeepSeek quant. `brandonmusic` publishes TrellisMX builds of DeepSeek-V4.1-Flash-1017 (K2, K2-Mix, K2-Mix-5GB, K3,
  K4: trellis-coded routed experts, W2A8 E4M3 at K2; Engram tables K4 in host RAM; attention / shared experts / router /
  head kept as released) and scores them on the teacher-logit set above with a fully specified protocol:
  KL(official || build) in FP64 over the full 129,280 vocab at every row; headline = the 512 fresh 2,048-token cuts,
  also the 128 full 8,192-token windows; BCa 95% window-block bootstrap, Wilson top-1. Their K2 (serving formats):
  **cuts 0.1439** [0.1346, 0.1545], top-1 86.82%; **windows 0.2182** [0.1779, 0.2690], top-1 86.41%; experts-only
  0.1429, Engram-only K4 0.0454. Speed on 2× RTX PRO 6000: decode 121 tok/s at batch 1 (299 at 8), prefill ~10k tok/s.
  **Plan:** (1) reference check: score a few windows with our bf16 path against their logits (download those files
  only, ~4 GB a window) → our engine's offset from the official reference; (2) score our DeepSeek quants on the same
  cuts / windows with their protocol (convert their logits to our top-512 format on the fly, or compute the full-vocab
  KL directly while streaming) and plot against their K2 / K2-Mix / K3 / K4 at equal bytes (count the Engram tables and
  every kept-as-released tensor in the size). Keep the set held out (never train / select on it) so the comparison stays
  clean; report with their interval method.
  Context (11-12): count sizes **without the Engram tables** (they live in host RAM / SSD; user 11-12): their K2 is
  ~158 GB for ~560B params ≈ **2.2–2.3 bpw**. KL 0.144 / top-1 86.8% vs the dense 27B's V2 interpolated at 2.25
  (~0.10–0.11 / ~88%): slightly behind V2's curve, before any routing inflation; a competent single-K trellis quant,
  a fair target. Our edge would be per-expert / per-tensor K (knapsack), H_out / swj-style scale fits and healing.
- **Public Qwen3.8-27B teacher data for QAD (searched 11-12):** no full-logit teacher set exists for the 27B. Candidates:
  - `logic65/whittle-distill-qwen3.8-27b`: 25,178 rows, 78M tokens, **top-20 logprobs on 25.4M answer tokens**
    (thinking on, agentic / tool prompts; licence "other": check before use). Closest match to QAD targets; top-20 only
    (our goldens keep top-512), answer tokens only, and reasoning-heavy, so useful as extra *training* data for V3.1 QAD
    if the licence allows, never for selection.
  - `MaxDevv/Qwen3.8-27B-Distill-1M-3.12B-Tokens`: ~1M conversations, 4.8B output tokens generated by the 27B, text only
    (no logprobs): a large on-policy prompt / response pool; we'd compute our own teacher top-k on a slice (box2 / ds8).
  - `phaedawg/qwen3.8-27b-distribution-fidelity-768x2048-v1`: eval-only fidelity suite (768 × 2,048 tokens, teacher
    tensors + head, KL table incl. NVFP4 / other quants): a third-party *eval* cross-check, like the DeepSeek one.
  - text-only 27B trace sets (agentic SFT, terminal-bench, SWE trajectories) exist too; none carry logits.
  Takeaway: for V3.1 our own qad pool + xcorp + session traces with our top-512 goldens remain the main QAD data; whittle
  (if licensable) and a Distill-1M slice are optional add-ons, ledger-recorded as non-calib corpora.

## User ideas (2026-11-12)
- [ ] **Vision blind spot.** Our corpora, Hessians, H_out and goldens are text only, so the quantizer weights directions
  image tokens use by how text uses them. V3: measure only: a vision KLD eval (image + text prompts, bf16 goldens,
  mmproj unchanged) on finished mixes vs V2 / bf16, plus w124's vision e2e test. If vision degrades much more than
  text, add image-token activations to the calibration statistics in V4 (a codes change, so not for V3).
- [ ] **Much larger bf16 goldens (top-512 logits) on the lent cards.** ~4 KB a position: 100M tokens ≈ 400 GB.
  QAD (V3.1): xcorp + session traces + self-generated text, as much as helps (the qad calib pool stays fixed at
  ~5.7M). Search: xcorp parts may grow (allowed for accept / reject); calib parts stay limited by the pools. Build
  after the ladder; size it with w123's noise-vs-cost model.
- [ ] **Self-generated (on-policy) text.** The board scores only model-written rows (`gen.u8`), so text the model
  writes itself is the real target distribution. Generate prompts' answers with a fast quant (IQR-V4 or our V3 once
  built: only the token stream matters), then score every position with the bf16 teacher. Use for QAD training
  (on-policy distillation, `xcorp/selfgen.py` exists) and as an extra xcorp search part; never for final selection
  (verify stays calib).
- [ ] **Intermediate-layer loss in QAD** (the NVIDIA recipe in ROADMAP V4): add per-block hidden-state MSE vs bf16 to
  the final-KL loss. Viable: the teacher forward runs alongside (hidden states are too big to store). Expected to help
  low-bit stability / convergence more than the final KL; test as an arm in the V3.1 pilot (KL only vs KL + block MSE,
  same steps), keep only if it wins on verify.
- **Mining with the local Qwen** (user): after w114, card 0 joins llama-swap and makes the 27B faster; restart the
  mining pass then (`mine/mix/PAUSE` off), Qwen only.

## Codebook speed vs quality, re-time for V4 / a joint IQR + EXL3 format (user 2026-11-13)
- [ ] **Re-time the hash codebook (#100) on w124's new decode kernels.** #100c (11-06): hash ×1.026 / 1.040 / 1.016 KL
  at K2 / 3 / 4 (≈ +0.011–0.025 bpw at equal quality) for +13–18% decode at M1, measured when decode was also grid-bound
  (8–40 WGs). After #164 (KS split, DQ, DOT) decode is bound by the codebook's ALU (54 VALU a 16×16 tile; IQR ~3.75 ops a
  value vs mul1i8 5.8–8), so the hash's decode gain is likely larger now (guess 25–40% at M1, untested). Format change
  (full requant): V4 only. Natural home: deadcode's future **combined IQR-V4 + EXL3-V3 format** (user 11-13) — bring
  this measurement, our trellis / two-sided / knapsack results and #147's layout table (`rad-local/research/iqr-v4-
  kernels.md`) to that discussion.
  **User 11-13:** V3 keeps mul1i8 (no ladder requant); the re-time waits for idle capacity / the joint-format work.
  Known cost: at ~4.1 bpw our decode step is est. ~15–20% behind IQR-V4 S (24.55 ms, user's web UI), more below K4.

## QTIP fine-tuning, checked against our plan (2026-11-12, user asked)
QTIP (arXiv 2406.11235 §3.1.2, §4.2, A.3.5) fine-tunes with QuIP#'s recipe: (1) **within a block, quantize matrices in
order (v, q, k, o, up/gate, down) and tune the not-yet-quantized weights** (plus the continuous parameters) on the
block's output; (2) end-to-end tuning of the continuous parameters (QuIP#: sign vectors as reals, norms); (3) for the
HYB code only, **tune the 2 KiB codebook LUT** (computed codes 1MAD / 3INST are "not tunable"). Paper: computed codes
without FT already beat QuIP# / AQLM *with* FT ("fine-tuning is not a silver bullet"); HYB 2-bit Llama 2 7B W2 6.83 →
5.86 with (1) + (3) together (no split ablation).
- (2) = our V3.1 scale-only QAD (suh / svh = the sign vectors as reals; norms are arm C of #165). Covered.
- (1) = sequential sibling compensation. It assumes the siblings' exact quantization, so it breaks mix-and-match: our
  asym writers (#87 / #105 / #111) are the same idea and fail per level next to siblings of another K (#111). Only
  possible per shipped mix (a per-mix requant): a V4 item with end-to-end code-aware QAD, not V3.
- (3) Our i8 codebook is computed (int8 decode in the kernel), not a LUT: tuning it would mean a LUT gather in the
  inner loop (kernel cost) or a new computed family; codebook-shape changes gave nothing on top of H_out (i8c #78-03 /
  -04). Not pursued; a tunable LUT would be a V4 format question judged with its decode cost.

## swj fit data (2026-11-11, #144 finding)
- [ ] **More swj fit tokens?** #144: fitting on half of the fit half (~232k vs ~464k tokens) lost 5.6% of the held-out
  gain (spread 1.2%), so the fit is still data-limited at today's ~0.93M qad tokens. A doubled qad slice (or qad + xcorp)
  might add a few % of swj's gain at ~2x the pass cost (affordable with `--numerics bf16`). Judge: held-out gain on
  the same odd documents (`--fit-sub` style), then KL on a search slice.

- **Head tile layout split for fractional K (w137 #175 leftover, 11-13):** store each tile's first 128 B aligned (as K4) +
  the fractional bits in a separate coalesced side stream (4 B a tile at K4.125); same codes and bytes, rad-convert
  repack, no requant. Gain: head GEMV 576 → ~629 GB/s at K4.125 ≈ 90 µs ≈ 0.3% e2e decode, only if heads go fractional.
  **Decide after w139 (#177):** do it if the codebook decision changes the container anyway, or if the speed-table redo
  shows the knapsack picking fractional heads. Runtime: decode only.

- **GPU L18 (/ L20) mad encoder → V3.1 requant with QAD (user + lead 11-13):** mad decode is free at any L < 24
  (byte-sum form); proxy L18 −1.0…−1.6 %, L20 −2…−3 % vs mad L16 (LOG 11-13 7:15 PM CST). Needs a GPU Viterbi above
  exllamav3's 16-bit limit (exact or seeded beam), checked against a CPU reference; measure encode time a tensor and
  27B KL at K2.5 / K4 before a requant (~2.5–3× ladder time). Also as a per-tensor option for sensitive tensors.
