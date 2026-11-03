# w05-head2s — two-sided LDLQ on the 27B LM head with its exact softmax-Fisher diagonal (ranked.md E2)

**Question.** Does two-sided LDLQ (`ldlq_2hess`) with the head's exact output-side Fisher diagonal F_v = Σ_t p_t(v)(1 − p_t(v))
(`headsens.npz` `fisher_qad_gen`, already computed for the slab ranking) lower the head-only KL at equal bytes? If so, in which
row order: vocabulary order (no format change), a strided order (every 128-row block spans all sensitivity ranks), or the
sensitivity order the slab heads use? And how does it compare with the slab head at the same bytes (E1, #34)?

**Mechanism.** EXL3's output Hadamard mixes 128-row blocks, so diag(F) becomes a dense 128×128 block per output block in the
rotated basis. Two-sided LDLQ shapes the error inside each block toward the low-F rows. The bound on the gain is
Σ_b GM_b(d)/Σ_b AM_b(d), with d = F·sv² per row. Sorting rows by F (slab order) flattens every block and removes most of the
gain. The vocabulary order keeps the natural spread, and a strided order maximises it.

A dense F_out for the head (248k²) does not fit, so this needs a **block-diagonal path**: `quantize-hout-bd.patch` (exllamav3)
plus `ourq-hout-bd.patch` (`--hout-bd`). A `hout_diag` vector stays a vector, and `ldlq_2hess_bd` runs ldlq()'s row recursion
with 8 sub-steps per 16-row step: tile column t of every block, feedback inside the block. It has the same dependencies as the
dense `ldlq_2hess`.

## CPU evidence already in hand (`headpred-proxy.txt`; colo, real head and real calib `lm_head.input`)

The Fisher here is a **proxy**: next-token counts on gen rows of the qad pool. The exact F is only on iceman. The quantizer is
exllamav3's own code with a **scalar** stand-in for the trellis (`src/cpuq.py`). K2/K3, 8,192 head rows. The error measure is
Σ_v F_v (w−q)ᵀH(w−q), in-sample for F.

| order | GM/AM bound (σ_out 0.01) | emulated K2 | emulated K3 |
|---|---|---|---|
| vocabulary | 0.275 | 0.40 | 0.37–0.38 |
| sensitivity (slab) | 0.759 | 0.76–0.78 | 0.75 |
| strided | 0.046 | 0.15–0.16 | 0.14 |

For scale, the slab head at K3 bytes measured 0.76× uniform head-only KL (`lcurve.json`, s3-f2, exact headsim). If even half of
the vocabulary-order emulated gain transfers to exact head KL, the uniform two-sided head beats the slab head with no format
change. **Caveats:**
- The emulated gain is in-sample on a count proxy that is peakier than the softmax Fisher.
- The scalar stand-in is not the trellis.
- F·err drops the −ppᵀ cross terms.
- Error moves onto rows the qad pool barely saw. Hence the rare-token guard and the held-out dump.

## Arms (iceman, one card; settings = the head donor recipe: mean-row, σ_reg 0.005, mul1, qtf2, K3 unless noted)

| arm | quant |
|---|---|
| `one` | one-sided, vocabulary order (control) |
| `two-v<σo>` | two-sided block-diagonal, vocabulary order, `--hout fisher --hout-bd --sigma-reg-out σo` |
| `two-s<σo>` | the same in strided order (`--head-perm strided.npy`; serving needs the slab heads' row permutation) |
| `hd-v` | one-sided + Fisher output scales (existing `--hout-diag` path: t ∝ (σ²/F)^¼), vocabulary order |
| `one-s` | one-sided, strided order, seed 0 (diagnostic: is the order alone worth anything) |
| `slab3` | E1's budget-3.0 slab plan (same bytes as K3 + 1 MB), when E1 has run (one seed, report only) |

- **Stage 1** (seed 0): σo ∈ {0.01, 0.1, 1} for two-v and two-s.
- **Stage 2:** seeds 0/1/2 (`--seed-offset`) for one, two-v*, two-s*, hd-v.
- **Stage 3** (conditional): K5, seeds 0–2, `one` vs the K3 winner.

**Scoring.** `src/headarms.py`: headsim's exact head-only KL on the 32,768 calib-A head inputs (fit pool, held out from the
qad Fisher) with per-position KL files, the rare-token guard (bf16 argmax outside the 20k most sensitive rows), and a
unit-bootstrap. Units are 512-position blocks: even = **selection** half, odd = **report** half. If E4's gen-row dump exists, it
is scored too (documents as units; secondary).

## Draft pass rule (for the lead to commit before data)

1. **Stage 1 (selection half):** per family, σo* = the lowest-KL σo whose guard KL ≤ 1.3× `one`'s. If none qualifies, the
   family stops.
2. **Stage 2 (report half; `src/judge.py`, thr 0.90):** a family **passes** if all of:
   - mean KL over its 3 seeds / mean KL over `one`'s 3 seeds ≤ **0.90**;
   - the bootstrap 95% CI upper bound < 1;
   - every seed is below every `one` seed;
   - guard ratio ≤ **1.3**.

   Winner = the passing family with the lowest ratio.
3. **Stage 3 (K5, thr 0.95):** the same rule, which decides whether it extends above K4.

## What the result changes

- **two-v passes:** it becomes the head recipe for every uniform head donor (headremake.sh gets `--hout fisher --hout-bd`).
  There is no format change. knap4's head prices scale by the measured ratio (E1's curve anchors move with them). E2 moves
  above E1 in ranked.md: about −0.1 × the head share of KL, i.e. −2 to −5% total at 2.0–2.5 if ≈0.6× transfers.
- **two-s passes and beats two-v by ≥ 10%** (report half, every seed): strided order plus the row permutation, the same
  plumbing as slab heads (TODO "Plugin: slab head").
- **Winner vs slab3:**
  - If the winner ≤ 0.9× slab3: low-budget slab plans (E1) become secondary.
  - Otherwise: hybrid. Two-sided sorted-order donors for the slab curve (E1 rerun with `--hout-bd` donors; the predicted
    extra is ×0.75).
- **Nothing passes:** E2 closes. hd-v is reported alone (it is the existing #33-06 / #48 path and was negative on the trunk).

**GPU time (iceman, one card + exclusive host RAM per quant):**
- Stage 1: 7 quants.
- Stage 2: about 11 more.
- Stage 3: 6.
- Scoring: about 2 × 20 min.

Expect **~5–8 h**. A two-sided quant is slower than one-sided (8 sub-steps of 1,940 tiles instead of one of 15,520): measure on
stage 1. **Run:** `CARD=0 ./run.sh` on iceman. It needs both patches merged, `/data/yaqa/hs/headsens.npz` and
`/data/yaqa/headslab2/perm.npy`. It is idempotent (finished donors and scored arms are skipped). Results go to
`/data/yaqa/w05-head2s/result.txt`.

## Patches (lead merges)

- `quantize-hout-bd.patch` (forks/exllamav3-frac): `block_ldl_batched`, `prepare_H_out_bd`, `ldlq_2hess_bd`. The dispatch in
  `quantize_exl3` happens only for a **1-D** `quant_args["H_out"]`, which no existing caller passes.
- `ourq-hout-bd.patch`: `--hout-bd`. lm_head's vector is permuted by `--head-perm`. Note: any ourq.py change moves the code
  stamp, so resuming a run started before the merge needs `OURQ_MIXED=1`, as for any ourq change.

## Self-test (`selftest.txt`: `python3 src/selftest.py <orig quantize.py> <patched quantize.py>`, CPU, 2 s)

- **T1:** original vs patched quantize_exl3 is **bit-identical** in four configurations: one-sided, dense H_out, scaled
  Hessian, dense + scaled.
- **T2:** batched block LDL equals block_ldl.
- **T3:** the vector path equals the dense diagonal `ldlq_2hess` path exactly (also with the scaled Hessian).
- **T4:** synthetic log-normal F: two-sided/one-sided weighted error 0.30–0.35 at K2–4.

`headarms_test.txt`: `headarms.py` + `judge.py` on a synthetic head. Permuted storage and slab-plan composition score
identically to hand-built heads, and the pass logic fires on the right family.

The ourq part cannot run on the CPU (it needs the GPU extension). GPU no-op check for the lead (4070, ~2 min): one layer of
`q2b.sh 08 3 i8` with the patched ourq and without `--hout-bd` gives `wq/` byte-identical to the unpatched run.

Files:
- `src/cpuq.py`: the CPU harness. One harness-only change: `block_ldl`'s `H.cpu()` is cloned, because on the CPU it aliases
  the tensor it overwrites.
- `src/headpred.py`: the predictor and emulation.
- `src/headarms.py`, `src/judge.py`.

## Runtime cost (user rule, 2026-11-03)
none (head quantizer objective).
