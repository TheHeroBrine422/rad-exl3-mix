# emb2 · Embedding / head representation study for the 1.0–2.0 bpw sizes (#119, w83, 2026-11-08)

Question (user 11-08): is there a representation of the input embedding (248,320 × 5120, 4.7% of the parameters) or
the LM head that beats today's (fwht128 + one scale a row + quarter-step mul1 trellis; slab head) at equal bytes,
especially at the new 1.0–2.0 bpw sizes? Ideas ranked in `IDEAS.md`; rules written before data in `RULES.md` (two
amendments, both before the data they govern). **Runtime cost of everything here: none** (CPU analysis; the only
screened candidate is encode-only).

## Results

### 1. The embedding has no structure left to exploit (all 248,320 rows; `embspec.py` → `specan.py`, `embnn.py`)

| | measured | best possible saving |
|---|---|---|
| shipped trellis vs the iid-Gaussian bound 2^-2R | ×1.03–1.09 (t0.25 ×0.96 – t4 ×1.09) | ≤ 0.03–0.06 bit for any better code (VQ, lattice, LUT, 2^20 trellis) |
| covariance spread (unit-RMS rows), eigenvalue AM/GM | 1.138 (centered 1.127); per-column AM/GM 1.0005 | KLT + water-filling ≤ 0.093 bit, < the 0.165-bit int8 rotation |
| per-128-block energy CV | 0.126 (iid: 0.125) | 0 |
| mean row | 1.0% of the energy | 0.007 bit |
| embedding from the head (least squares, full 5120² map) | 85% of the energy left (rank 256: 91%) | ≤ 0.12 bit before the map |
| nearest other row | median cos² 0.05, p99 0.33; best-8 least squares leaves 86% | ≤ 0.11 bit before pointers |

To second order the table is a white Gaussian matrix with no near-duplicate rows. The shipped coder is at the floor
for an unweighted (H = I) objective. Head (calib `lm_head.input` H): the head rows are extremely anisotropic in the H
metric (eigen AM/GM 7.1), but LDLQ already takes it (pivot AM/GM 1.02 in the signed fwht128 basis); water-filling
beats the LDLQ model by only ×0.84 (0.12 bit), less than a stored rotation.

### 2. Embedding LDLQ against a sensitivity metric (`embldlq.py`, 4,096 rows, EXL3 16×16 tiles): **FAIL**

Two proxies of which embedding errors the model feels: **M0** = layer 0's input projections with the norm gains
folded in, diag(g)·ΣWᵀW·diag(g). It weights the embedding's top-256 principal directions 2.7× the average; Gaussian
bound in M0: 0.31 bit, LDLQ alone ~0.2. **F** = the measured diagonal Fisher of the residual at layer 0: Σdy² of
layers.0 out_proj, 27B `hdiag-dy`, qad pool. F's diagonal has channels at 18× the mean; M0's diagonal is flat. The
arms are H = I (control: it reproduces #35's t1 0.2562 vs 0.2570), H = M0, H = F, and the hedge H = MF = (M0+F)/2
(amendment 2). Ratios are to the control at equal K; bits are read off the control's curve in that metric.

| H | K | I ratio (bits) | M0 ratio (bits) | F ratio (bits) | rate passes |
|---|---|---|---|---|---|
| F | 1 | 1.057 (-0.04) | 1.035 (-0.02) | 0.910 (+0.07) | no |
| F | 1.5 | 1.066 (-0.05) | 1.048 (-0.03) | 0.900 (+0.08) | no |
| F | 2 | 1.069 (-0.05) | 1.057 (-0.04) | 0.896 (+0.08) | no |
| F | 3 | 1.072 (-0.05) | 1.068 (-0.05) | 0.892 (+0.08) | no |
| M0 | 1 | 1.188 (-0.13) | 0.616 (+0.30) | 1.203 (-0.14) | no |
| M0 | 1.5 | 1.198 (-0.13) | 0.667 (+0.26) | 1.219 (-0.15) | no |
| M0 | 2 | 1.205 (-0.13) | 0.710 (+0.24) | 1.228 (-0.15) | no |
| M0 | 3 | 1.211 (-0.14) | 0.738 (+0.21) | 1.236 (-0.16) | no |
| MF | 1 | 1.076 (-0.05) | 0.656 (+0.26) | 0.971 (+0.02) | no |
| MF | 1.5 | 1.079 (-0.06) | 0.720 (+0.21) | 0.961 (+0.03) | no |
| MF | 2 | 1.080 (-0.06) | 0.769 (+0.18) | 0.956 (+0.03) | no |
| MF | 3 | 1.082 (-0.06) | 0.803 (+0.15) | 0.953 (+0.04) | no |

Control (H = I): rel_I 0.2562 / 0.1299 / 0.0649 / 0.0166 at K1 / 1.5 / 2 / 3 (#35: 0.2570 / 0.1300 / 0.0649 / 0.0166).

- **H = M0** saves 0.21–0.30 bit in M0 and loses 0.13–0.16 bit in both I and F.
- **H = F** saves 0.07–0.08 bit in F (×0.89–0.91, under the 10% bar at three of the four rates) and loses 0.02–0.05 bit
  in M0.
- **H = MF** (the hedge, chosen after the first data): the only arm better than the control in **both** sensitivity
  metrics at every rate (M0 ×0.66–0.80 = +0.15–0.26 bit; F ×0.95–0.97 = +0.02–0.04 bit), for −0.05–0.06 bit in I.
  It fails the rule (F gain < 10%).

**Verdict: FAIL (all three arms, `data/emb2/report.out`).** M0 and F disagree about which directions matter, so the
CPU proxy cannot pick an H: this is the pre-registered "self-fulfilling" case. The size of M0's gain (0.2–0.3 bit,
free at runtime) is what makes the true metric worth one GPU measurement (below).

## Translation to model bpw / KLD (knapsack level)

One bit a value on the embedding or the head = **0.0473 bpw**. From the knapsack's own size–KLD curve (knap4 v3e,
`data/e3/sub1/r90knap.md`: d lnKLD / d bpw ≈ 3.2 at 1.25, 2.3 at 1.5, 2.1 at 1.75, 1.9 at 2.0), bytes freed on the
embedding and re-spent on the trunk are worth:

| bits a value saved | 1.0–1.25 bpw | 1.5 | 1.75 | 2.0 |
|---|---|---|---|---|
| 0.03 (H = MF if F were the truth) | −0.5% | −0.3% | −0.3% | −0.3% |
| 0.08 (H = F's gain in F, if F were the truth) | −1.2% (more at 1.0) | −0.9% | −0.8% | −0.7% |
| 0.10 | −1.5% | −1.1% | −1.0% | −0.9% |
| 0.20 (H = MF if M0 were the truth) | −3.0% | −2.2% | −2.0% | −1.8% |
| 0.30 (H = M0's gain in M0, if M0 were the truth) | −4.4% | −3.2% | −2.9% | −2.7% |

So the metric question is worth roughly **−1% to −4% KLD at 1.0–2.0 bpw** (similar per bit at 2.5–3 bpw, where
d lnKLD / d bpw is still ~1.6), for free at runtime, depending on which proxy is right; if plain H = I were right
instead, MF would cost +0.5–0.8% (−0.06 bit). The CPU proxy cannot decide it. No representation change is worth a format or decoder change: every
structural route is capped at ~0.1 bit, and most of that is eaten by side tables.

## What it would take (lead decisions)

1. **(GPU, ~1–2 card-h, decides #1 and enables #2 of IDEAS):** one backward pass on fit-pool sequences with sampled
   labels (collect.py's machinery), recording at the embedding output h0 (a) the full 5120² Σ g gᵀ, the true Fisher
   metric F_full, and (b) Σ‖g_t‖² and the count per input token id, the measured per-row sensitivity. Then on CPU:
   embldlq.py with H = F_full (minutes per arm) and a row-allocation simulation. If either saves ≥ 0.1 bit in F_full,
   run a KLD swap at t1 / t1.5 on a 1.5-bpw base with the multilingual guard, as #32 did. **Ship cost if it passes:**
   LDLQ needs nothing at runtime (embgpu.py / ourq gains an H for the embedding; same format, same `embed_tq`
   decoder). Per-row K needs a per-256-row-run K table in the embedding's trellis decoder (~20 lines in embed_tq) and
   the importer's layout.
2. **(CPU, minutes, independent of this study):** offer **low-budget slab heads** (budgets 2.125–3.5 with the floor
   ~1 bit below, from `v3/yaqa/data/lcurve.json`) to knap5 at 1.0–2.0 bpw. Today `KN_HEADSLAB` offers only K5 / 5.5 /
   6, and the head sits at uniform K3 at 2.0–2.25 (reprice main). headsim at equal bits, vs log-interpolated uniform
   heads: 2.5 / f2 −42% head KL (guard KL ×3: reject); 3.125 / f3 −46% with the guard also lower. A head KL of 0.0079
   at K3 is ≥ 6.6% of the 2.0-bpw KLD (0.12), so this is worth ~−2…−3% at 2.0 bpw before compounding. Gate: the
   pending #34-02 whole-model check at one low budget.
3. **The embedding's price below t1.5 is extrapolated** (KN_EMBT=2 scales the t2 real-KLD point along the CPU error
   curve). At 1.25–1.5 bpw the knapsack sat at its menu floor (t1 / t1.25 in r90knap, before #97 added t0.25–0.75), and
   by the marginal values above it would go lower. The open TODO item "real-KLD t1 / t1.25 swap" (plus t0.5) decides
   more bytes at these sizes than any representation change could.

## Files

- `IDEAS.md` (ranked candidates), `RULES.md` (screen rules + amendments), this README.
- `embspec.py` (one pass: second moments of the embedding / head, f64), `specan.py` (bounds), `m0an.py` (M0 bound),
  `embnn.py` (shared rows), `embldlq.py` + `runldlq.sh` (the LDLQ screen; restartable per arm), `emb2report.py`
  (table + verdict). `embklt.py`: the KLT coder, written before the bound closed it, never run.
- Data `data/emb2/`: `specan.json` / `.log`, `m0an.log`, `embnn.json`, `spec.log`, `ldlq/*.jsonl`, `report.out`.
- Machines: colo `/data/work/w13/w83/`, Thor `/data/work/w83/` (code + logs only; the 845 MB moments file and the
  5120² U / M0 matrices were deleted after logging: `embspec.py` + `specan.py` rebuild them in ~5 min).
- Run: `python3 embspec.py OUT/spec.npz && python3 specan.py OUT/spec.npz OUT/specan.json` (colo / Thor, 8 threads);
  `runldlq.sh` (env `KS`, `HS`, `MAXJ`, `THR`; ~2.6 GB and ~17–25 min per arm at 2 threads), then
  `python3 emb2report.py data/emb2/ldlq/*.jsonl`.
