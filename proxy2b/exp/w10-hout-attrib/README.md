# w10-hout-attrib: where does the diagonal-H_out gain come from? (ranked.md B3)

**Question.** The diagonal H_out (Σdy², `edy`, the 27B collector's primary estimator since #33-18/#33-20) beats base
on the 0.8B by **K3 −9.2%, K2 −6.5%** (3 seeds; seed-paired gain 0.092 ± 0.006 / 0.065 ± 0.006,
`selftest/real-summ-4070.txt`). Three things are open:
- Which families carry the gain? (GDN inputs, attention inputs, MLP gate/up, residual writers out_proj/o_proj/down)
- Which depth third carries it? (layers 0–7, 8–15, 16–23)
- Does it survive at K4 and K5?

The answers set the scope of the two-sided Phase-B requant, whether the 27B backward can be truncated, and which
levels need the "oursH" pricing family (F1).

**Mechanism / prior.** #33-09: the diagonal is very uneven for input projections (CV 1.5–4) and flat for residual
writers (CV ~0.2). A flat diagonal is close to the identity, and the identity H_out alone is *worse* than base
(#33-16). So I expect the writers to contribute little and the input projections most. YAQA's Table 8 has the gain
flat in K (−28/−30/−29% at K2/3/4), so I expect K4 to keep a real gain; K5 is the open case.

**Arms** (0.8B, i8, base settings + diagonal H_out at output damping 0.01, golden g08-search1):

| tag | H_out on | K | seeds |
|---|---|---|---|
| `k{3,2}h-gdn_in` | linear_attn.in_proj_qkv / in_proj_z (36 linears) | 3, 2 | 0, 1 (+2, 3 adaptive) |
| `k{3,2}h-attn_in` | self_attn.q / k / v (18) | 3, 2 | 〃 |
| `k{3,2}h-mlp_in` | mlp.gate / up (48) | 3, 2 | 〃 |
| `k{3,2}h-writers` | linear_attn.out_proj, self_attn.o_proj, mlp.down (48) | 3, 2 | 〃 |
| `k{3,2}h-d0 / d1 / d2` | all linears of layers 0–7 / 8–15 / 16–23 (50 each) | 3, 2 | 〃 |
| `k{4,5}edy` | every linear (the full Σdy² diagonal) | 4, 5 | 0, 1 (+adaptive) |
| controls (existing, reused) | `k{3,2}base*`, `k{3,2}edy*` (seeds 0–2), `k4base`, `k4base_s1`; new: `k5base`, `k5base_s1` | | |

- A subset arm is the full arm's `--hout` directory restricted to the subset by symlinks (`mksub.py`).
- ourq quantizes linears without a file exactly as base does (same seed, one-sided), so **outside its subset each
  arm is bit-identical to its seed's base arm**.
- Families and depth thirds never split a Hessian group (q/k/v, GDN qkv/z, gate/up); the self-test checks this.
- `--side-partial` is passed only if w01-failloud's ourq patch is merged.

**Why 2 seeds suffice for family / depth vs base:**
1. The comparison is seed-paired, and a subset arm differs from its seed's base only in the subset's own tensors.
   - Verified on real quants: ourq is bit-deterministic run to run (`selftest/real-determinism-4070.txt`).
   - `run.sh` checks the identity for every subset arm (`idcheck.py`).
   - So an arm's rounding noise comes from its subset alone, below the whole-model figure. Seed-paired log-ratio SD
     of the full arm is 1.04% (K3) / 1.06% (K2) per seed, i.e. ≤ 0.75% SE on a 2-seed mean, ≤ 0.08 in share at K3
     and ≤ 0.11 at K2. For a family holding a quarter of the rounding, it is about half that.
2. The questions are coarse: a share above or below 0.15, and a gain above or below 3% at K4/K5. The effects that
   matter are many SEs from these boundaries.
3. Pre-registered adaptivity covers the rest:
   - An arm gets a 3rd (then 4th) seed when its boundary lies within one SE.
   - If any identity check fails, every subset arm gets 3 seeds (`MINSEED=3`).

## Draft pass rule (for the lead to commit before data)

**Definitions.**
- Per arm X and K: l_s = ln(KL_X,s / KL_base,s) over the paired seeds; gain g_X = −mean l_s; SE = SD(l_s)/√n.
- Share s_X = g_X / G(K), where G(K) is `edy`'s gain over seeds 0–2. Delta-method SE.

**Rules.**
0. **Identity gate:** every subset arm is bit-identical to its seed's base outside the subset (`id-*.json`).
   Otherwise → 3 seeds for all subset arms (automatic), and the "why 2 seeds" argument is void.
1. **Adaptive seeds:** if |s_X − 0.15| < SE(s_X) (subset arms), or |G(K) − 0.03| < SE (K4/K5), add one seed, up to
   4 in total. After that, decide on the point estimate.
2. **Family f:**
   - **one-sided** in the Phase-B requant if s_f < 0.15 at both K3 and K2, or if f is *harmful* (g_f < 0 with
     z ≤ −2 at either K);
   - otherwise **two-sided**.
3. **Depth third:** same rule.
   - If **d0** (layers 0–7) is one-sided, the 27B collector may truncate its backward at a third of the depth
     (start layer 21 of 64). collect.py's truncation hook exists only for YAQA_FULL today; extending it to
     YAQA_DIAG is a small patch, not written here.
   - A one-sided middle or late third only drops those linears' diagonals; the backward is unchanged.
4. **K4 / K5:** if G(K) < 0.03, levels ≥ that K are requantized one-sided. Quarter levels between K3 and K5 follow
   log-linear interpolation of G between the measured K.
5. **Additivity** (reported, not a gate): Σ family gains / G and Σ depth gains / G. Outside 0.7–1.3, a "one-sided"
   family verdict needs an all-but-that-family arm before Phase-B adopts it (not run here).

## Run

```
proxy2b/exp/w10-hout-attrib/run.sh        # 4070; waits for any running ourq / evalq; idempotent, resumable
```
- Results go to `/data/yb/w10-hout-attrib/result.txt`. The final table and verdicts come from `summ.py`, which also
  writes JSON.
- **GPU time:** 34 quants + evals at ~2 min each ≈ 70–80 min. Adaptive seeds add ≤ 20 min.
- Needs only existing data: `c18/hdiag-snap1024-dy` and the base / edy arms in `/data/yb`.
- Run `../w10-longpos/run.sh` again afterwards: it re-scores every subset arm on the long golden (family × position)
  for ~40 more minutes of evals.

## What the result changes
- **Phase-B requant (V3):** families and levels marked one-sided are quantized without H_out. That saves
  ldlq_2hess time (B10 becomes unnecessary for them), and F1's "oursH" repricing applies only to the two-sided ones.
- **27B diagonal night (#33-13):**
  - d0 one-sided → backward truncated at ~layer 21, about a third less backward time.
  - A family with share < 0.15 → its diagonal is still collected (free in the same pass) but not used.
- **ranked.md:** B3 resolves. If the writers carry < 15% (expected), B6/B7's permutation ideas, which target
  gate/up and q/k/v output blocks, keep their place. If they carry most, B6/B7 rise.
- **INDEX:** a new #33-xx row.

## Files
- `mksub.py`: builds the subset `--hout` dirs.
- `idcheck.py`: the identity gate.
- `summ.py`: shares, verdicts, adaptive-seed list.
- `run.sh`: the run.

## Self-tests
- **CPU, `selftest/selftest.txt`: ALL PASS.**
  - mksub on the real c18 file list: families 36/18/48/48 and thirds 50/50/50 each partition the 150 linears, no
    Hessian group is split, an empty subset is refused.
  - idcheck: passes an identical outside, catches one changed weight.
  - summ: planted shares recovered exactly, verdicts and additivity right, the adaptive seed is listed for the arm
    at 0.155 and not for clear ones.
- **Real data** (4070 CPU, niced):
  - `selftest/real-determinism-4070.txt`: two ourq runs of one setting are bit-identical (150/150); base vs edy
    differs on 150/150.
  - `selftest/real-summ-4070.txt`: summ.py on the existing arms gives G = 0.0919 ± 0.0060 (K3) and
    0.0646 ± 0.0061 (K2).

No patches.

## Runtime cost (user rule, 2026-11-03)
none (attribution of the H_out gain).
