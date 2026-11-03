# w03-swiglu-diag — A9: forward-only analytic diagonal H_out for the MLP gate / up projections (0.8B)

**Question.** Can the Sketch B diagonal of H_out for `mlp.gate_proj` / `mlp.up_proj` be replaced by a closed form
computed from a forward pass alone, without losing the gain those two families contribute? If it can, the 27B
collection could skip the gate/up weight-gradient work (the 17,408-wide outputs: the largest accumulators and
`diag_gg` cost), and gate/up could get their diagonal from a forward-only pass (ranked.md A9; INDEX #33-12/13/16/18).

**Mechanism.** h = SiLU(g)·u, y = W_d h. The gradient on gate / up channel r is (W_d[:, r]ᵀδ)·SiLU'(g_r)u_r and
(W_d[:, r]ᵀδ)·SiLU(g_r), δ = dL/dy. If δ is isotropic and independent across tokens, Sketch B's
E‖Σ_t dy_t[r] x_tᵀ‖² equals exactly

    a9x:  w_gate[r] = ‖W_d[:, r]‖² Σ_t (SiLU'(g)u)² ‖x_t‖²       w_up[r] = ‖W_d[:, r]‖² Σ_t SiLU(g)² ‖x_t‖²

(`selftest.py` checks this identity by Monte Carlo). `a9d` drops the ‖x_t‖² weight. Two assumptions can break it:
(i) the residual gradient is far from isotropic (massive channels), (ii) per-sequence coherence (Sketch B sums dy·x
over the sequence before squaring; the closed form is the per-token sum). `a9cmp.py` separates them using c18's
measured extras: `cos_a9x_vs_xw` tests (i) alone (xw is the measured per-token Σ‖x‖²dy²), `cos_xw` vs B tests (ii)
alone, and `a9g` (isotropy replaced by down_proj's measured Σdy² diagonal) shows how much (i) costs.

## Files
- `swiglu_diag.py` — forward-only collector (same windows as collect.py / c18: calib qad pool, seed 0, 2048 tokens,
  all positions, fp32). Writes `a9x-snapN/`, `a9d-snapN/` (ourq `--hout` layout, gate/up only) and `raw-snapN/`.
- `a9cmp.py` — the CPU gate: cosine (and log-correlation) per layer vs the Sketch B diagonal.
- `a9mk.py` — builds `--hout` dirs: c18's B diagonal for every linear except gate/up (from the A9 dir, or none).
- `a9oracle.py` — the diagnostic behind the verdict (CPU, real backward): oracle token weights, a forward-only Fisher
  token weight, the full residual-gradient covariance, and the measured per-token form, all vs Sketch B.
- `seedsumm.py` — 3-seed means, per-seed SD, gains vs base, "every seed beats every seed".
- `run.sh` — the 4070 entry point; `selftest.py` / `selftest.out` — the algebra check; `cpu-gate.txt` — the CPU
  gate on real 0.8B data (colo CPU, see below).

## Arms (K3 and K2, rotation seed offsets 0 / 1 / 2, base settings, output damping 0.01, i8, g08-search1)
| arm | gate / up | other linears | source |
|---|---|---|---|
| `base` | one-sided | one-sided | existing (#33-16) |
| `eB` | Sketch B diag, 1K | Sketch B diag, 1K | yaqaFact7 (#33-18); built here with the same command if missing |
| `noGU` | **one-sided** | Sketch B diag, 1K | new: the gate/up share of the gain |
| `a9x` (or `a9d`) | **analytic** | Sketch B diag, 1K | new |

12 new quants. noGU − eB = the gain gate/up's measured diagonal is worth; noGU − a9 = what the analytic one recovers.

## Draft pass rule (for the lead to commit before data)
CPU gate (no GPU; on c18's 1K B diagonal):
- **C1:** a9x median cosine over the 24 layers ≥ 0.95 for gate and for up, and min ≥ 0.90. If a9x fails and a9d passes,
  run the GPU arms with `A9VAR=a9d`; if both fail, A9 fails at the CPU stage (the GPU arms may still be run for noGU,
  which answers the attribution question B3 for gate/up on its own).

GPU (per K; gain = mean over 3 seeds):
- **G0 (is there anything to recover):** gate/up gain G_GU = mean(noGU) − mean(eB). If G_GU < 1 base per-seed SD
  (~1% K3, ~2% K2) at both K, gate/up's diagonal is worth nothing measurable on the 0.8B → the 27B can drop gate/up from
  the two-sided path outright (cheaper than A9); A9 is moot.
- **A9 passes** if at both K: mean(noGU) − mean(a9) ≥ 2/3 · G_GU, and every a9 seed beats every noGU seed.
- Fail otherwise.

## What it changes
- A9 passes → the 27B collection can drop gate/up from the backward hooks (YAQA_DIAG over the other families only;
  gate/up diagonals from a forward-only pass of `swiglu_diag.py`, adapted to the 27B's two-card loader). Saves the
  gate/up `diag_gg` (two 17,408 × 5,120 weight gradients per layer, the largest of the per-linear products) but not the
  backward itself (the other families need it). Also gives a gate/up diagonal for
  any K-level requant with no backward at all.
- G0 fires → gate/up one-sided in the 27B requant (B3 / K4 scope), regardless of A9.
- Fail → 27B gate/up keep the Sketch B diagonal (status quo).

## Running
`bash proxy2b/exp/w03-swiglu-diag/run.sh` on the 4070 after #33-18 (c18 collection) has finished. Idempotent:
finished arms are skipped; results in `/data/yb/w03-swiglu-diag/result.txt`. Set `A9VAR=a9d` if the CPU gate says so.

**GPU time:** forward stats ~5 min (1,024 forward passes, no backward); then, as C1 fails, 6 noGU quants + evals
≈ 25 min (12 quants ≈ 50 min with `FORCE_A9=1`); plus up to 6 eB control quants (≈25 min) if yaqaFact7 has not made
them. **≈ 30 min** (≈ 1 h forced).

## CPU gate result (2026-11-02, colo CPU): A9 FAILS C1, and no forward-only fix found
All against c18's 1K Sketch B diagonal (#33-18; its own halves median 0.990 gate / 0.988 up).

| estimator (gate / up, median cos over 24 layers; min) | gate | up | file |
|---|---|---|---|
| **a9x** (analytic, 128 windows × 2048) | **0.864** (min 0.245) | **0.901** (min 0.648) | `cpu-gate.txt` |
| a9d (no ‖x‖²) | 0.864 | 0.901 | `cpu-gate.txt` |
| a9g (isotropy → down's measured Σdy² diagonal) | 0.867 | 0.913 | `cpu-gate.txt` |
| oracle token weight (true ‖δ_t‖², 40 × 1024, backward) | 0.888 | 0.918 | `oracle40.txt` |
| softmax-Fisher token weight 1 − Σp² (forward-only) | 0.855 | 0.905 | `oracle40.txt` |
| fullC (W_d[:,r]ᵀ C̄ W_d[:,r], mean residual-gradient covariance, 24 × 1024) | 0.864 (min 0.421) | 0.931 (min 0.759) | `oracle24-fullC.txt` |
| measured per-token Σ‖x‖²dy² (c18 xw, 1,024 × 2048) | **0.995** | **0.995** | `cpu-gate.txt` |

Reading:
- **Sequence coherence is not the issue.** The measured per-token form matches Sketch B at 0.995, so the per-token
  structure of the closed form is right.
- **What fails is the gradient model.** The per-token gradient size (oracle: +0.02), its diagonal (a9g) and its full
  mean covariance (fullC: gate unchanged; up +0.03) each recover little. What is left is the correlation, token by
  token, between the down-output gradient's direction and the channel's own SwiGLU activation. Any
  token-independent model misses it. It is worst at the last layer (L23 gate 0.24 for every model, 0.98 measured)
  and at L5/L6/L10/L22.
- So the gate/up diagonal needs the backward. On the 27B, the cheapest valid option stays the per-token Σ dy² or
  Σ‖x‖²dy² sum, which c18 measures at 0.995 vs Sketch B. Whether those keep the KL gain is #33-18's A2 question.

**Recommendation:** A9 rejected at the CPU stage. `run.sh` checks C1 itself, skips the a9 arms (`FORCE_A9=1` overrides)
and runs only `noGU`: 6 quants, ~25 min. noGU still answers G0, i.e. how much of the diagonal-H_out gain gate/up
carry (ranked.md B3 scope for the 27B requant). If G0 fires (gate/up worth < 1 seed SD), the 27B collector can drop the
gate/up hooks, which is the saving A9 was after, without any approximation.
`oracle40.txt` is from the first version of `a9oracle.py` (before the fullC column); the file as committed produces
both tables.