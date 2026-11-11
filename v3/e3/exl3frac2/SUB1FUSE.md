# w108 · #142 · qtf2 sub-1 speed: fuse the 0-bit steps (KA = 0)

Worker w108, task `local/workers/instr3/w108.md` (user go 11-11). **Rule written 2026-11-11 ~8:55 AM CST (14:55 UTC),
committed alone before any measurement of this experiment** (nothing built or timed yet). Not edited after data;
amendments get a date and go below the rule they change. Nothing here edits shared code in place: the kernel change is
delivered as a patch to `v3/e3/exl3frac2/qtf2.cuh` (+ checks) under `v3/e3/exl3frac2/sub1fuse/`, the lead merges.

**Numbering.** INDEX's highest row is #141 (#134 / #135 are gaps held by others): this takes **#142**.

## What changes

Context: `pipeline/PROFILE.md` §0 (K0.125 level ~3.3 card-h, 90% trellis), LOG #37 ("Not done: sub-1 0-bit-step
fusion"). In the KA = 0 kernel (`qtf2_kernel<0, MASK>`, 512 threads, costs in global memory) every one of the 256
positions reads and writes all 65,536 state costs in global memory with a block barrier. Between two positions whose
MASK bit is set (branching), a sub-1 ring's 16-bit state cannot change: the steps (D(i), D(i+1)) = (1, 0) (fan-out,
reads the 15-bit costs), (0, 0) (pure add) and the (0, 1) min step only extend each state's own cost.

**Step 1 (this experiment):** thread t of the KA = 0 kernel keeps the costs of the states it already owns in every
65,536-wide step (s = 2t + 1024 m + h, m < 64, h < 2; the same states the (0, 1) min step needs for both k) in
registers. (1, 0) reads the global 15-bit costs once, (0, 0) updates registers only, (0, 1) does the min in registers
and writes the 15-bit costs and the history row (as today), (1, 1) steps stay today's global step. Barriers only after
a global write. **The arithmetic per state is unchanged**: the same half FMA chain c ← fma(d, d, c) in the same position
order, the same salted / unsalted codebook reads (LUT or procedural), the same first-step masking, the same strict-less
pick. So (a) below is expected, not hoped for; any difference is a bug to find, not a summation-order effect.
Old behaviour stays available in the same source as a test build (`QTF2_NOFUSE`).

**Step 2 (macro-step trellis with precomputed emission costs): only if step 1's profile shows per-run overhead still
dominating** (rule: after step 1, if a timing-only split shows global cost traffic + barriers > 30% of the KA = 0
kernel time at K0.125, step 2 is attempted; else skipped, with the split reported). Step 2 changes the summation order
(Σ of a run's errors added to the incoming cost), so it can only ship under judge (b).

## Judges (fixed before any data)

Rates: every sub-1 instance **K0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875** (KA 0, masks 8080 / 8888 / a4a4 / aaaa /
dada / eeee / fefe), **both codebooks, i8 first** (i8 = the ladder's; mul1 = the plain build). Reference = **today's
kernel**: `v3/e3/exl3frac2` at HEAD (518a7c3) built unchanged in its own tree, plus the `QTF2_NOFUSE` build of the
new source. Card: dad1's RX 9070 XT (gfx1201, same chip as the R9700; ratios on the same card).

**(a) Identity → ships as a pure speed change (no settings stamp change, no code-stamp change needed).**
`sub1fuse/fusecheck.py`: qtf2check's tile set (seed 0, N = 1024: Gaussian N, Laplace(0.7) N/4, half-integer ties
N/8, 8 zero tiles, 8 tiles of 3.5, 8 tiles ×12; 1,448 tiles) **plus** 1,024 real-weight tiles (regularized,
Hadamard-rotated rows of one w95 K0.125 tensor, if fetched; else a second Gaussian seed ×0.5 / ×2 scale mix),
every KA = 0 instance, both codebooks, **both 0-bit read paths (QTF2_LUT = 1 default and 0)**, launch batches 32 and
128: indices and values must equal the reference's bit for bit (0 differing elements). Also: every KA ≥ 1 instance
identical (regression: that code is untouched), and **determinism**: the new KA = 0 kernel run 4× on the same input
gives identical output each time (race check, as qtf2race.py). Any difference at KA = 0 that survives a bug hunt →
judge (b).

**(b) Only if (a) fails after the bug hunt.** Report the fraction of differing tiles per rate / codebook. Two-sided
held-out proxy (w87 / w98 f2 method, `v3/e3/settle/w87`) on the w95 K0.125 / K0.5 tensors with the trellis calls
routed through the new vs the reference kernel (GPU, same everything else): per tensor rel2_B(new) / rel2_B(ref)
≤ 1.005 and geo mean ≤ 1.000 → ship as a code change (code stamp only), **lead decides**; otherwise not shipped.

**Speed (reported, not a gate; the change is worth shipping at any speedup ≥ 1.10 at K0.125 if (a) holds).**
(1) tiles/s by rate, new vs reference, i8, launch batches 32 / 64 / 128 / 256 (qtf2sub1.py's method: 1,024 Gaussian
tiles, one warm call then one timed), same card, best of 3; (2) if card-0 time allows: one real K0.125 layer through
`rxq job` on gpubox card 0 (dev build, `/root/card0.lock` then `ram.lock`), per-tensor wall vs w95's per-tensor times;
(3) projected card-h for the seven KA = 0 levels (K0.125–0.875) from PROFILE.md's per-level job walls scaled by the
measured trellis-share speedup (K1 is integer KA = 1, unaffected).
