# d1112 · Does xcorp H_in stack with the adopted 0.8B stack? + fractional-K drift (0.8B, 4070 queue)

Worker w44 (2026-11-06). Pre-freeze questions for the trunk quantizer: (1) d1111 adopted the xcorp H_in blend (full
mX, w0.5; amendment 3's verdict, lead 11-06) on the **dygm** base only (one H_out, no swj, no writers asym). Does its
gain survive the full adopted stack, K2 / K3 and K4 / K5? (2) Fractional-K LDLQ drift (w43's item 1b note).
**Rules below written 2026-11-06 ~4:50 AM CST, before any data.**
Runtime cost: none for all items (quantizer / statistics only: same format and kernels).

## Arms

**Stack** `k{K}st` = d1104's `k{K}edyswjasym_r0` recipe (d1109's best arm; REAL-Q failed), re-run on today's code
(both arms of every pair on the same code):
trunk quant **edy** (`q2b.sh 08 K i8`, Σdy² H_out c18 `hdiag-snap1024-dy`, σ_out 0.01, seed offset s) → **swj**
(`swiglufit.py`, fit slice g08-fit1) → **moments** (`moments.py --mode sib`, through the swj weights, qad, 1M tokens,
seed 0) → **G, F** (`mkdenoise.py` λ 1, ρ 0 on the calib H_in) → **writers asym ρ0** (the 48 writers in `only.txt`
requantized with `--hin G --wpre F`, same H_out and seed) → `wqall` (writers + swj weights). No healing.

**Stack + X** `k{K}sx`: the same, with d1111's **hi-X0.5** (calib H_in blended with mX's xcorp H_in at w 0.5, `blend.py
hin`; copied to `/data/yb/d1112/hi-X0.5`) as the H_in **everywhere the stack uses calib's H_in**: the trunk quant
(`--hin`) and the base H of the writers' metric G (mkdenoise's CALIB argument: G = H_X + λ² N + λ(C + Cᵀ) in its
control-variate form). F (the Wiener filter) uses only the subset's own moments (as mkdenoise documents), so it is
the same recipe. This is what a ladder adopting X would run (one H_in definition throughout).
Stage readouts (search only, not deciding): `k{K}ste` / `k{K}sxe` (trunk quant), `k{K}ste_swj` / `k{K}sxe_swj`.

Goldens (as d1111 amendment 3): **search** = `g08-search1` (calib search pool, the deciding slice), **vf** =
`g08-verify1` (verify pool: the overfitting check only), **xc** = `g08-xcorp2`, **ag** = `d1108/g-agent`. Paired per
seed (`pcmp.py`: ratio sx / st, per-document bootstrap z). "geo" = geometric mean over seeds of the per-seed ratios.

## Item 1 · stack × X at K2 / K3

Seeds 0, 1, 2 first; then seeds 3, 4. **The decision is read on seeds 0–4 (n = 5)**; the 3-seed read is an interim
report. Seeds 5, 6 (queue fill) are a robustness check: reported; if they move a geo across a threshold below, the
result is labelled *fragile* (and goes to the 27B check), never flipped silently.

**Rule** (per K ∈ {2, 3}, sx vs st):
- **stacks (adopt-compatible):** at both K, search geo ≤ 0.99 with every seed < 1, **and** verify keeps ≥ 2/3 of the
  search gain (mean log ratio over seeds on vf ≤ ⅔ × the search mean log ratio; as #93 / d1111 amendment 3).
  → X stays in the recipe on the full stack (the 27B confirmation is d1111 amendments 2 / 4's pairs, on bn).
- **doesn't stack:** search geo > 0.995 at either K. → On 0.8B evidence X does not combine with the stack: the 27B pair
  must be run on the full stack (not bn alone) before X ships; until then X is out of the frozen recipe.
- **search-only gain:** the search part passes but verify keeps < 2/3 → not adopt-compatible (selection noise).
- anything else (both K ≤ 0.995 but not the full pass) = **partial**: reported with its numbers; the 27B pairs decide.
- Reported, not deciding: xc and ag ratios on the stack (flag if xc geo > 0.98 at either K, d1111's xcorp bar: X would
  then lose its real-world point on the stack; flag if ag geo > 1.02, amendment 3's agentic cost), stage readouts,
  and a **reproduction check** of `k{K}st` vs d1104's `k{K}edyswjasym_r0` score files (same recipe and seeds, older
  code): expected ~1.00; off by > 2% means the stack moved with code changes since 11-04 (reported; the sx / st pair
  stays valid, both arms are on today's code).
Why the lead's suggested rule was kept as is, with one adaptation: "every seed < 1" gets stricter as seeds are added
(sign-test p = 1/8 at n = 3, 1/32 at n = 5), so the decision is fixed at n = 5 and extra seeds can only label it
fragile.

## Item 2 · transfer to K4 / K5

Same arms, seeds 0, 1, 2 decide; seeds 3, 4 (queue fill) are a robustness check as in item 1. **Rule, per K:**
- **X at that K:** search geo ≤ 0.99, every seed < 1, verify keeps ≥ 2/3 of the gain.
- **harmless:** 0.99 < geo < 1.01. X costs nothing at inference, so one recipe for every K ("decide globally"):
  X may run at that K if item 1 passes.
- **keep X off at that K:** geo ≥ 1.01 with every seed > 1.
- otherwise inconclusive at this noise (reported).

## Item 3 · fractional-K LDLQ drift

Found while building (before data): **`quantize/ourq.py` already interpolates the drift by default** (commit
`d19647a`, 2026-11-02: log-linear between exllamav3's table points {1, 1.5, 2, 2.5, 3} and 1.0 at K4; table points
unchanged). `OURQ_DRIFT_EXACT=1` gives the old exact lookup (drift 1.0 at quarter K). w43's item-1b note ("fractional K
gets no drift correction") holds for exllamav3 alone and for settle.py's proxy (its `DRIFT` dict is the exact
lookup), not for quants made with ourq. So **no patch is needed**: both arms run on unmodified repo code, and the
question becomes whether the shipped default is right.

Arms: the edy trunk quant only (`q2b.sh 08 K i8`, c18 H_out, σ_out 0.01; no swj / writers: the drift only moves the
global scale search, and the downstream steps add time and noise), K ∈ {2.25, 2.75, 1.75} (deciding) and K1.25
(reported), seeds 0, 1, 2: `f{K}i` = default (interpolated; drift 1.0135 / 1.0065 / 1.0265 / 1.0573) vs `f{K}x` =
`OURQ_DRIFT_EXACT=1` (1.0). Paired fi / fx on search. Sanity check: each quant's `proxy.jsonl` "drift" field must show
the intended value, else the arm is invalid.
**Rule:** (a) **interpolation confirmed** if geo(fi / fx) ≤ 0.99 with every seed < 1 at ≥ 2 of the 3 K;
(b) **revert to the exact lookup** (lead's call, it changes ladder stamps) if geo ≥ 1.01 with every seed > 1 at ≥ 2
of the 3 K; (c) otherwise **no measurable effect**: keep the default (no change, no stamps touched).
Coordination with w43 item 1b (not landed when this was written): 1b's m* at quarter K is measured against
exllamav3's scale **without** drift; ourq already multiplies by the interpolated drift. A 1b factor used as
`ourq --gain` at quarter K must therefore be divided by the interpolated drift first, or the drift is counted twice.

## Order, time, disk

Queue order (`run.sh`): prep → item 1 seeds 0–2 → item 2 seeds 0–2 → item 3 → item 1 seeds 3–4 → item 2 seeds 3–4
→ item 1 seeds 5–6 → report. Estimates (from d1104 / d1111 logs: quant ~2 min, swj ~1.8, moments ~5.4, writers ~1.7,
the four goldens ~2.3 min): a stack arm ~13 min (K4 / K5 a bit more), an item-3 quant + eval ~2.5 min.
Item 1 s0–2 ~2.6 h, item 2 s0–2 ~3 h, item 3 ~1.1 h, item 1 s3–4 ~1.7 h, item 2 s3–4 ~2 h, item 1 s5–6 ~1.7 h:
**~12 h** of 4070 time (more if the lead's experiments share gpu.lock).
Disk: an arm's `wq` copies, moments and G / F (~5 GB) are deleted as soon as its four goldens are scored; kept per arm:
ourq shards (~0.2 GB), logs, score files. Footprint ≲ 25 GB (limit 70).

## Files and how to run

- `run.sh` the queue (idempotent: finished arms skipped by their score files; every GPU unit holds `/data/yb/gpu.lock`
  < 15 min; stop: `touch /data/yb/d1112/STOP` or `/data/yb/night.stop`, checked between units). Runs from the
  snapshot `/data/yb/d1112/snap/`. Output `/data/yb/d1112/`: `result.txt` (eval and pair lines), `report.md`.
  `D1112_STEPS` (default `prep i1a i2a i3 i1b i2b i1c report`) selects steps.
- `report.py O` applies the rules above → markdown (also written to `O/report.md` at the end of each item).
- Registry: `/data/queue.d/30-d1112.sh`; watchdog rules: `local/watch.conf` (`d1112`, `d1112out`, `d1112fail`).

## Status

Started 2026-11-06 4:51 AM CST (10:51 UTC) via `resume-queues.sh 4070`. First arm (k3st, 13.5 min) reproduces
d1104 bit for bit: k3ste_swj 0.0872954471 = d1104 k3edy_swj, k3st 0.0752772210 = d1104 k3edyswjasym_r0 (search,
gen-masked mean KLD). ETA with the 4070 to itself (CST): item 1 seeds 0–2 ~7:35 AM, item 2 seeds 0–2 ~10:45 AM,
item 3 ~11:45 AM, item 1 seeds 3–4 (decision read) ~1:35 PM, item 2 seeds 3–4 ~3:45 PM, item 1 seeds 5–6 ~5:30 PM.

## Results

(none yet)
