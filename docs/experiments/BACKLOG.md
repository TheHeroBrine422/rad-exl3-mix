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

| proxy result | 27B test needed | depends on |
|---|---|---|
| Diagonal H_out (Σdy² / Sketch B), 0.8B 3 seeds: K3 −8…−9%, K2 −5…−6% (#33-16, #33-18) | 512-seq 27B collection (both cards), pilot swap at K3 / K2.5 vs ladder levels | iceman back, card 1 |
| Estimator choice (dy vs Sketch B vs xw), 512 seqs (#33-18) | collect all three in one 27B pass, compare in the swap test | same run |
| Cubic int8 codebook i8c (#78: 0.8B KL K3 −1.7%, K2 −2.7%, K4 −2.3%; adopted) | requant 2–3 trunk levels with i8c, swap test | iceman; HIP build of the i8c decode |
| Tile coordinate descent (#77: CPU proxy −0.9%, consistent) | only if judged worth its 2–3× quant time | ffn_down results, a seed-paired KL test |
| (#33-19 / #33-20 / #33-21 failed or neutral: none ride along.) Gen-masked loss (w01) | ride on the 27B collection if it passes | its 0.8B result |

## Faraway lens (2026-11-02, `local/papers/runs/2026-11-02-faraway/ideas/faraway.md`): ideas to test later

- [ ] **A1 · down_proj predicted from its neuron's decoded up + gate rows, residual coded** ("chroma from luma"):
  measured 0.09–0.23 bit on down_proj (mean ~0.16; up↔down |cos| 0.27–0.47), est. −3 to −8% KL at 2.5 bpw, P ~0.4.
  **Constraint (challenge author, 2026-11-02): the target is VRAM, load-time expansion only as a stopgap** → it
  needs the online-prediction kernel (faraway A2: transposed GEMMs re-reading the up / gate tiles, +1 GEMM with up
  only keeps ~75% of the gain; +1–2 GEMMs in the MLP). Weigh the gain against that speed cost. Steps: (1) real-trellis
  proxy test on iceman's GPU (quantize D vs R = D − αÛ − βĜ at K2 / K3 from the ladder levels; pass ≤ 0.85× at K2 on
  layer 30); (2) 0.8B KL with load-time expansion (allowed for measuring); (3) kernel cost (ntbench); then decide.
  Riders: A3 (GDN out_proj ← v), A4 (o_proj ← v / q-gate), A6 / A7 census (minutes on colo).
- [ ] **E10 · dead-unit census** (E[h²] at down / out_proj / o_proj inputs from the calib Hessians; colo minutes).
- [ ] **F2 · boundary diagonal affine repair** (closed form, 0.8B ~2 h). **F4 · exact last-layer output Hessian.**
- Faraway's headroom audit: gate/up within ~0.1 bit of an ideal LDLQ + Gaussian coder; weights near-Gaussian; no
  cross-layer redundancy (kills cross-layer sharing / global-basis ideas). The rest of its list: the report.
