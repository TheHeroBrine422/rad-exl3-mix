# blockx · review ideas 7 and 5: per-block A8 exceptions for A4, and the decode-plane arithmetic (w42, 2026-11-06)

Source: `local/mine/review-2026-11-05-ideas.md` ideas 5 and 7; feeds #46-03 (`README.md`). CPU only: colo, 6
single-thread jobs at nice 10, 12:50–12:58 AM CST (~0.8 CPU-h). Scripts in `blockx/`, data in `blockx/data/`.
**Runtime:** both are about **prefill** (`gemm_us`) only. Idea 7 would add A8 work inside A4 GEMMs; idea 5 changes the
decode plane's format. Neither changes decode (GEMV).

## Verdicts

- **Idea 7 does not rescue A4.** Two A8 blocks per row cut gate_up's A4-g64 error by only **6–11%** in each depth band
  (rule: ≥ 30%). attn_in passes in L0–15 only (32.9%); gdn_in peaks at 27.9%. The same holds at g128, for all three
  block selectors, and at k = 4 for gate_up (12–17%). The A4 error is spread over the 40 blocks, not held by one
  outlier block. (The norm gains shrink the massive channel before gate_up; #47 found the same.) The few tensors
  that do have an outlier block are the q-gate of attn_in (L3/7/11/15: 75–84%) and GDN z (L6–34: 42–56%). They hold 9%
  and 20% of their families' A4 error; not worth a per-tensor option (decide-globally rule).
- **Idea 5, i4 K2–3 levels for the A8 path alone: no, not at the board's M 2048.** A packed int4 plane on the A8
  path saves the measured decode-pass time (wdec8 → wdec4). That is **+5.4% of gate_up's A8 path, +2.7% of down's, and
  ≤ +1.2% or negative everywhere else** (today's wdec4 is *slower* than wdec8 on the small shapes). Even that much
  requires the in-tile int4→int8 unpack to hide behind the WMMAs. If it serializes the way the A4 fold did, every
  shape loses (−2…−10%). Quality allows only K ≤ 2.5 (i4/i8 = 1.011 at K2, 1.018 at K2.5; 1.026 at K2.75 and 1.067 at
  K3 break the 1.02 limit). It pays at M 512 (gate_up +11…+30%, down +3…+22%), which the board does not score.
- **Idea 5, accurate path (int8 plane converted to f16 in the T2 tile):** about **+5.5%** at M 2048 if the
  conversion hides, **−8%** if it serializes, +11% if the GEMM's plane read is serial time too. At M 512: +6…+32%. The
  review's "4–8%" is the middle case. Whether the conversion overlaps the WMMAs decides both variants of idea 5.

## 1 · Idea 7: k A8 blocks inside A4 (`blockx/blockx.py`, `report.py`)

**Data.** E1c's activation dumps, already on colo at `/data/work/aq/dump`. They hold 512 real GEMM inputs per linear
(8 prefill steps × 64 rows, calib-A = fit pool, all-K6 model, 2.5 GB); nothing was copied or regenerated. Operands,
rotation, quantizer and weight-row subsampling are imported from `v3/e1b/aqproxy.py` unchanged. k = 0 reproduces
E1c's `aq.jsonl` exactly (gdn_in L0 g64 rel_out 9.050e-4, gate_up L0 6.174e-3).

**Method.** A4 at g64 / g128 (clip 1, as the kernel). A per-tensor mask puts k ∈ {0, 1, 2, 4} rotated 128-blocks at
A8 (same group). Error is rel_out = Σ‖W e‖² / Σ‖W x‖². gate_up's halves are masked separately, since each has its own
suh. **Selection is held out:** the mask is chosen on steps 0–3 and scored on steps 4–7, and vice versa (2-fold,
sums reported).
Selectors:
- `amax`: mean rotated-block absmax², the idea's rule.
- `inmax`: the same before the rotation.
- `err`: each block's own measured output-error cut on the selection half.

They agree within 1 point everywhere except attn_out / ssm_out with inmax, which is worse.

**Error cut at k = 2, selector amax** (sum over the band's tensors; full tables with k = 1 / 4, the other selectors,
per-tensor median and best in `blockx/data/report.out`):

| family (blocks / row) | L0–15 | L16–31 | L32–47 | L48–63 | k = 4, best band | g128, k = 2, best band |
|---|---|---|---|---|---|---|
| gate_up (40) | 6.2% | 9.5% | 11.0% | 9.7% | 16.9% | 10.5% |
| attn_in (40) | **32.9%** | 20.1% | 15.1% | 9.7% | 38.5% | 30.6% |
| gdn_in (40) | 27.9% | 26.8% | 22.5% | 15.7% | 33.3% | 26.0% |
| down (136) | 1.8% | 1.7% | 1.7% | 2.3% | 4.1% | 2.2% |
| attn_out (48) | 13.2% | 14.4% | 10.9% | 10.4% | 23.3% | 14.4% |
| ssm_out (48) | 9.2% | 9.8% | 9.4% | 10.8% | 19.5% | 10.8% |

Rule (≥ 30% at k ≤ 2 in ≥ 2 bands, on gate_up and on the attn / GDN inputs): gate_up 0 of 4 bands, attn_in 1 of 4,
gdn_in 0 of 4, at g64 and g128 and for every selector. **Fail.** All-A8 removes 99.4–99.7% of the error, so the
error is real; it just isn't concentrated in a few blocks.

**Whole family, E1c proxy** (ΔKLD ≈ k_f · Σ rel_out, `v3/e1b/data/aqfactors.json`, ±30%), A4-g64 on every layer:

| family | A4 | k = 2 | k = 4 |
|---|---|---|---|
| gate_up | 8.1e-3 | 7.4e-3 (−9%) | 6.9e-3 (−15%) |
| attn_in | 3.1e-3 | 2.5e-3 (−18%) | 2.3e-3 (−25%) |
| gdn_in | 2.3e-3 | 1.8e-3 (−23%) | 1.7e-3 (−29%) |
| down | 4.2e-3 | 4.1e-3 (−2%) | 4.1e-3 (−4%) |

**Implied time cost.** k / B of each row's 128-blocks run at A8 instead of A4:
- 2 / 40 = 5% for gate_up, attn_in and gdn_in; 2 / 48 = 4.2% for attn_out and ssm_out; 2 / 136 = 1.5% for down.
- An A8 block costs ~1.3× an A4 block with w34's folds and ~1.9× without, so this is +1.5–4.5% of gate_up's A4 GEMM
  at k = 2.
- Not counted: the exception blocks' int8 weights (an in-tile unpack, or +5% plane bytes as an int8 side plane), their
  int8 activations in actq4, and the per-K-step branch.

For comparison, moving gate_up from A4 to A8 costs ~+29% of its time and removes ~100% of the error. The block
exceptions buy 9% for 1.5–4.5%: no better per unit time than the knapsack's existing A4 / A8 choice.

## 2 · Idea 5: decode-plane bandwidth (`blockx/planes.py`, output `blockx/data/planes.out`)

**Inputs.** w34's card runs (mul1i4 K2 / K3, M 2048 with each pass timed alone; the M 16–4096 sweep for M 512),
`local/workers/instr3/w34-results/`.

**Plane bytes per step** (MB; independent of M). The trellis read is W·K/8 on top.

| shape | f16 | int8 | int4 |
|---|---|---|---|
| gate_up | 357 | 178 | 89 |
| down | 178 | 89 | 45 |
| gdn_qkv | 105 | 52 | 26 |
| attn_qg | 126 | 63 | 31 |
| attn_o, ssm_out, gdn_z | 63 | 31 | 16 |
| attn_k, attn_v | 10 | 5 | 3 |

**Model** (only the decode passes are measured; the rest is an estimate):
- Bandwidth: the large-shape decode passes run at **444–552 GB/s** (mean 494).
- T2's f16 decode pass (never timed alone) ≈ wdec8 + W / BW. That puts it at ~13% of gate_up's f16 path at M 2048,
  38% at M 512. It checks out: f16 path minus this estimate gives GEMM times 5654 / 1405 µs at M 2048 / 512, a ratio
  of 4.02.
- In-tile conversion, if it does **not** overlap the WMMAs (as w34 found for the A4 fold), costs 0.50–0.64 ns per wave
  instruction per SIMD. That rate is calibrated on the fold: no-fold vs g64b1, VOPD on / off.
- Instruction counts:
  - int8→f16 in T2: 8 ops per 16×16 fragment; each fragment feeds MT = 4 WMMAs.
  - int4→int8 in the A8 tile: 3 ops per fragment (and, shift, and); unsigned-B WMMA plus a −7.5·Σx row term like A4's
    R; each fragment feeds 4 WMMAs.
- Scenarios:
  - pess: decode saving minus serialized conversion.
  - mid: decode saving only, conversion hidden.
  - opt: mid plus the halved plane read once at BW.

**i4 tensors under A8, packed int4 plane** (gain in % of that tensor's A8 path; pess / mid / opt):

| shape | wdec8 → wdec4 µs (K2), measured | M 2048 | M 512 |
|---|---|---|---|
| gate_up | 502 → 271 | −2.4 / **+5.4** / +9.6 | +10.7 / +16.9 / +30.0 |
| down | 205 → 146 | −4.9 / +2.7 / +6.8 | +2.5 / +8.8 / +22.4 |
| attn_qg | 133 → 115 | −6.8 / +1.2 / +5.6 | −2.7 / +4.2 / +19.0 |
| gdn_qkv | 78 → 98 | −9.6 / −1.6 / +2.7 | −12.8 / −5.7 / +9.7 |
| gdn_z, attn_o, ssm_out | 53–55 → 58–64 | −8 to −9 / −1 / +3 | −9 to −12 / −2 to −5 / +10 to +12 |
| attn_k, attn_v | 16 → 23 | −9 / −4 / −1 | −14 / −10 / −3 |
| **layer, all tensors i4** | | −4.9 / **+2.8** / +6.9 | +2.9 / +9.2 / +22.8 |
| gate_up + down only | | −3.3 / +4.4 / +8.6 | +8.1 / +14.2 / +27.5 |

**Accurate path, int8 plane converted in T2** (gain in % of the f16 path; same scenarios): at M 2048, −8.1 / +5.5 /
+11.0 on the layer, nearly uniform across shapes. At M 512: +6.1 / +15.8 / +31.7. Per-shape rows are in
`planes.out`. Caveat: T2 would multiply by the integer q and apply 4·K_INV in the epilogue. That is more accurate than
today's rounded f16 plane but **not bit-exact** with it, unlike what the review assumed.

**Answer to "are i4 K2–3 levels worth building even if A4 never ships?"** Not on the A8-path speed argument at the
scored M 2048:
- The ≥ 5% bar is met only by gate_up, only in the middle case, and only just (+5.4%).
- down is +2.7%, and the other shapes lose because wdec4 is slower than wdec8 on them today.
- The quality bar (≤ 1.02) admits K2–2.5 only.

(The user's #90-02 decision already builds the i4 ladder at K1.125–3 unconditionally. That stays as it is; this
result only says the int4-plane A8 kernel variant is not justified by M 2048 speed.)

Three things would change the answer:
1. A microbench showing the unpack hides behind the WMMAs; then gate_up alone is a ~5% case.
2. A real-prefill metric at M ≈ 512, where the gain is large.
3. A4 shipping, since wdec4 already exists for it.

Note: kernel-opts P1 (decode inside the GEMM tile) would remove the plane traffic altogether and with it most of this
gain.

## What only a card can tell

One microbench (~1 card-hour, as the review planned) settles the pess / mid split for both variants: the A8 tile
reading a packed int4 plane with an in-tile unpack, and T2 reading an int8 plane with an in-tile int8→f16 conversion,
vs today's tiles at M 512 / 2048 on gate_up and down. Idea 7 needs no card follow-up.

## Files

- `blockx/blockx.py`: the idea-7 scorer (imports `v3/e1b/aqproxy.py`).
- `blockx/run.sh`: the colo driver (6 jobs, idempotent).
- `blockx/report.py`: the tables; `blockx/planes.py`: the idea-5 arithmetic.
- `blockx/data/blockx.jsonl`: 6,720 lines, 336 tensors × {g64, g128} × (k = 0 + 3 selectors × k ∈ {1, 2, 4}).
- `blockx/data/report.out`, `blockx/data/planes.out`: script outputs.
- Rerun:
  - `python3 blockx/report.py blockx/data/blockx.jsonl --factors ../../e1b/data/aqfactors.json`
  - `python3 blockx/planes.py`
- Colo scratch `/data/work/w13/w42/` (code copies + per-job outputs, 1.5 MB).
