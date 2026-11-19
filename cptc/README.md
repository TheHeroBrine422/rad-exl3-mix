# cptc/ — CPTC CPU reference (arXiv 2611.22019)

Reference implementation + verification of Computed Permutation Trellis Coding (Balın, Amazon
Annapurna Labs, 2026-11-17) for the CPTC workstream (local/TODO.md, C1–C4). Workstream decision:
user 2026-11-19 — "get this working if it's real before working on dsv4.1f" (sequential, not
parallel with DS4.1).

## Key construction fact

CPTC's tail-biting bitshift trellis (Lmax=16, window q_t = [low K: code_t | high 16-K: first
16-K stream bits AFTER code_t]) is bit-identical to the **EXL3 state-after-shift of the reversed
value sequence** (v3/e3/viterbi.c convention: [low K: code | high 16-K: last bits BEFORE it]).
So:

    CPTC-encode(x) == viterbi_tiles_tb(reverse(x), cb = decode_table(a0, a1, Tlower))

reversed back. The 65536-entry `cb` is exact for the encoder and checks; the serving decoder is
the computed two-multiply form (Algorithm 1) with the 64-byte (2Q-entry) signed table — not this
LUT. No new encoder C code is needed beyond the `_st` state-output variant in v3/e3/viterbi.c
(used for the bit-exact Algorithm 2 cross-check).

## Protocol fidelity (paper refs; assumptions noted)

- Lmax = 16, Q = 6, T = 256, s = 0; coefficients = Table 6 L=16 row (per K), depth L = 16 −
  ctz(a0|a1) (asserted).
- Table: signed 64-entry lower half T↓ (cell j<64 decodes +T↓[j], i≥64 decodes −T↓[126−i]);
  mirror-symmetric pairs (j, 126−j), j ≤ 62; **center cell 63 = 0** (exact fold forces it).
  E2M5 = sign + 2 exp (bias 1) + 5 mantissa, subnormals, no inf, all-ones magnitude NaN
  (Sec 5.1); entries snapped to the 128-magnitude grid.
- Generation (Sec 3.3/A.2): Eq (2) conditional-mean init × (1+2/n)(1−4^−K), n = L−K+1; 30 Lloyd
  rounds + 20 damped rounds (damping 0.5 — **not stated in the paper, assumption**) at 2^22 fresh
  samples per round, best round by held-out Keff (2^18); 16 E2M5-snap Lloyd rounds; final
  single-entry grid hill-climb (one sweep).
- Tail-biting: QTIP construction (Cornell-RelaxML/qtip) — two-pass Viterbi (free run rolled by
  T/2 fixes the wrap group; constrained run pins it); viterbi.c `ntry` rolls. Fit rounds use
  ntry=1 (production EXL3 heuristic; encoder quality only affects cell assignment); the Keff
  evaluation uses ntry=16.

## Commands

    gcc -O3 -march=native -fopenmp -shared -fPIC v3/e3/viterbi.c -o v3/e3/libviterbi.so
    # double build (exactness checks): sed float->double (+ float.h include, DBL_MAX)
    python3 cptc/cptc.py prop1 [K]    # Proposition 1: scalar formats reproduced exactly
    python3 cptc/cptc.py table K      # full A.2 table generation -> cptc/tables/table-K{n}.npy
    python3 cptc/cptc.py keff K       # Keff on fresh i.i.d. N(0,1), 2^22 samples (Table 2 target)
    python3 cptc/cptc.py check K      # bit-exact: encoder vs CPTC's own Algorithm 2 window decode
CPTC_LIBDIR env overrides the lib directory (repo-relative by default).

## Targets (Table 2, L=16, E2M5, Gaussian, their protocol)

| K | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Keff | 0.950 | 1.948 | 2.943 | 3.934 | 4.913 | 5.802 |

Their seed spread: 0.0005 bits at K=4 (3.9334 ± 0.0005). Pass bar: within ±0.005 of the targets.

## Verified so far (2026-11-19)

- **Prop 1 PASS for K=1..6** (0/16384 mismatches, f64 DP): decode-table construction, reversal
  equivalence, and the tail-biting Viterbi plumbing all reproduce the paper's scalar special case
  exactly.
- **f32 note:** the production f32 Viterbi flips strictly-worse near-tie code choices in the
  fully-separable Prop-1 degenerate case (1 position per 16K; f32 accumulation ulp ≈ 1e-5 vs
  1e-6-level distance gaps). Harmless for real trellises (value depends on the full 16-bit state,
  paths are coupled); documented because it is surprising.
- Table-generation pipeline runs end-to-end (coarse 2^16/12-round K=4 table: Keff 3.39, well
  below target as expected underfit — scalar Lloyd–Max at K=4 is 3.36, so the trellis machinery
  is active and the fit just needs the full protocol).
- C2 full-protocol run: colo, K=1..6 in parallel (8 threads each), ETA ~2026-11-20 00:40 UTC.
