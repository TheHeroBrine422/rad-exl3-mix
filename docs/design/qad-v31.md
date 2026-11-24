# V3.1 QAD — plan and state (lead, 11-22)

One page so the plan fits in a head. Details: LOG entries 11-21 / 11-22, BACKLOG "QAD …" entries.

## What we know
- Scale-QAD (suh / svh log multipliers, frozen codes, KL to the stored bf16 golden) gains at 2.5 / 3.0 (best: 2.5 lr 3e-4 0.5M 0.892,
  3.0 lr 3e-4 0.953 on j217), and ~5 % at 4.0 only with lr 3e-4; lr 1e-3 drifts at 3.0 and diverges at 4.0.
- **The gradient is ~90–99 % noise.** Only 0.3–1.6 % of scales carry detectable signal, mostly layers 32–63 and attention q / o.
- **The signal is low-rank:** per tensor a rank-~4 subspace that independent data agrees on (cosine ~0.9 vs ~0.05 raw, null-checked),
  for the weight gradient and the scale gradients alike.
- Adam eps = a worse lr knob (keep default). Trust region: never binds (dropped). Blow-up documents: document sensitivity, not
  runaway scales.
- Code-QAD: plain gradient negative; low-rank step worse so far (step size / sign check running, w218).
- Trainer (w220): compiled default; frozen prefix 1.6× (layers 32–63) / 2.15× (48–63), exact; knobs for every idea below.

## Now running
- w217 (box2 cards 0 / 1): lr 3e-4 1M extensions at 2.5 / 3.0, then a 2.5 2M run — does lower lr keep gaining?
- w221 (box2 cards 2 / 3): late-layers-only (32–63, 48–63), tied front half, clip + β2 — does cutting noise help, and does training
  only the back half lose anything?
- w218 (iceman card 0): low-rank code step — sign check + curvature-sized step.

## Next (in order, each gated by the one before)
1. **Projected low-rank steps for the scales** (subspace from a sketch pass, refreshed from the previous window) + **curvature line
   search** for the step size. The direct fix for the noise; would replace lr schedules.
2. Whichever of {lower lr + more rows, projected steps} wins → its 1–2M curve with checkpoints; then the calib-anneal cooldown.
3. Front-half local MSE (diagnostic first) for the layers KL cannot reach.
4. Re-screen N / H / L, A8-in-the-loop at a sane lr.
5. Final recipe: [projected steps or best schedule] → short low-lr plain pass → keep-best floor → full V3.1 run on every size that
   gains, one verify read each.

## Size ladder (user 11-23)
Target sizes, if the gains hold: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, maybe 5.0. Per size: a short starting-lr screen (0.25–0.5M at 2–3 lrs; the
usable lr falls with size: 3e-4 at 2.5 / 3.0, ≤ 1e-4 at 4.0), then one long run with an automatic lr-drop controller (plateau rule on the
paired fixed monitor + a random-walk detector on scale increments; validated first by CPU replay on existing runs). Open for ≥ 3.5:
larger batches at equal rows (same compute per row; less noise per step where the signal is weakest).

## Budget rule (user 11-22)
Run only as long as the question needs: if 0.5M can settle it, stop at 0.5M (a quarter of a 2M run). Go to 2M+ only for questions that are
about length (plateaus, long schedules, the final recipe) or whose effect only shows late.

## Checkpoint rule (user 11-22)
Every training run takes WSD branch checkpoints (a cheap decay forked off the main run) at 0.25M / 0.5M / 1M / 2M (/ 4M) up to
its budget, each read on j217, so every run gives a curve. Whole-run cosine / linear decays: at each checkpoint, fork a short linear decay to 0 over ~50k rows (2–3 steps) and read
that ("short-decay"); the run's own end point is the main number.

## Judging rules
j217 (held-out qad pool) for comparisons; ratio with and without the blow-up documents; flag any document whose KL rises > 10×;
verify only for finalists; never public.

- **EMA: dropped (user 11-24).** Small gain (3.0: 0.965–0.993 at 1–2M), not worth the extra reads and complexity; final runs use raw checkpoints.
