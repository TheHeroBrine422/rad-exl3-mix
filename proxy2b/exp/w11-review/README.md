# w11-review: skeptical review of #87 (sibling asym), #89 (frozen-code healing), #85 (fixed-code scale refits)

Reviewed 2026-11-03, finished ~10:15 AM CST. Code read in full; read-only checks on the 4070 (CPU only, niced). No GPU used,
no shared file edited.

**Verdict.** The three gains are real for what they measured. No leakage, the math is right, the eval matches what a
pack decodes, and the pairing is right. One bug would distort a queued verdict (#89-02 / heal1). Two findings change
how #87 can be used at 27B and how its stacks read.

## Findings, by severity

### A: would change a conclusion

**A1. d1103's heal steps score the healed arms without their folded norms.**
- `d1103/run.sh:17-18`: `ev()` always runs `evalq.py $M` (the stock checkpoint).
- The heal step (`d1103/run.sh:118`) and run2's heal1 (`run2.sh:36`) therefore drop `fold.py`'s `OUT/model`: the trained
  q_norm / k_norm / final-norm gains.
- `w06-heal/run.sh:22` does it right (`$d/model` when `model/.done` exists), so #89's own numbers are fine.
- Size: in #89's k*all params, max |log g| on the norms is 0.038 (K3) and 0.10 (K2); the final norm is 0.029 / 0.059. The
  linear factors were trained jointly with those gains, so scoring without them can understate the heal or even hurt.
- **Not run yet:** d1103 was in its `scale` step at 10:07 AM CST, and heal comes after allasym and k45.
- **Fix:** `d1103-heal-norms.patch` (adds an optional MODEL argument to `ev` and `hm DIR`, used at both heal call sites).
  `patchtest.sh` checks it: it applies, parses, old call forms are unchanged, the heal uses `OUT/model` when present, and
  run2's sed-import picks up `hm` → `patchtest.txt`: PASS.
- **Caution:** do not edit a running bash script in place (bash reads it by offset). run.sh's running process will keep
  the old code. Run2 is still in its wait loop and has not sourced run.sh yet; restart it after applying the patch.
- For run.sh's own heal step (or if nothing is restarted), run `healfix.sh` after the heal steps. It re-scores every
  `h-*` arm with its folded model into `/data/yb/w11-review/` and writes the pairs to `healfix.txt`.

### B: might change a conclusion (the 27B plan, and how the stacks read)

**B1. #87's correction mostly fits one rounding of the siblings.** (`fsim.py`)
- (F − I) for the same tensor, built on seed 0 / 1 / 2 siblings: cosine 0.02–0.10 (diagonal parts 0.1–0.4), at L3 o_proj,
  L14 down_proj and L1 out_proj.
- K3 vs K2 siblings: 0.06–0.16. base vs edy at the same seed: 0.15–0.18 (shared rotation signs).
- So the 14–18% gain is GPTAQ-style error compensation for those particular sibling weights. That is legitimate: each
  arm was scored with exactly its own siblings. But the F is invalid whenever a sibling changes afterwards:
  - the knapsack picks another K, seed or H_out variant for q/k/v, GDN qkv/z or gate/up;
  - scale refits or healing touch the siblings.
- For #58 ("per-level library item", `w07-seqsib/README.md:73-77`): the writer requant must run **last**, on the final
  shipped siblings. It cannot be a per-level library entry that is mixed freely.
- Test: `xseed.sh` (4 evals, ~4 min, no quantization; assembly checked on CPU → `xseed-dry.txt`). It puts seed-0 asym
  writers on seed-1 siblings and compares against seed-0 plain writers on the same siblings. Prediction from the
  cosines: ratio ≈ 1.

**B2. run2's `full` stack is a stale-F combination.** (`d1103/run2.sh:25-26`)
- swj (gate/up output scales) and α* (svh of every family) are applied **after** the asym writers were fit. At K3, α* sets
  attn_in −1 and gdn_in / gate_up 2, i.e. it changes o/out's and down's siblings.
- `refitq.py:85,105-106` also measures each writer's row error against the original W, not the W·F target, so α*
  partly undoes F's deliberate shift.
- "Scale refits don't stack on asym" would therefore be a false negative. Fix: siblings' refits first, then the moments
  and the writer requant; or leave the 48 writers out of α*.

### C: cosmetic / bookkeeping

- **C1. Moment pass vs calib Hessian.** `hsub_vs_calib` (mkdenoise's own check) is median 0.14, max 0.40 (L14 down,
  L6 out, L7 o).
  - This is not a hook error. The moments use 2,048-token windows of 1M qad tokens (`moments.py:61-63`); h08/calib uses
    all 10.26M calib tokens in 8,192-token chunks (`hcollect.py:32-34`). That gives 4× more window starts, which carry
    the massive first-position activations, and these sit mostly in the down inputs.
  - G uses calib's H (`mkdenoise.py:329`), F the subset's own (`:343`). Each is self-consistent, but the metric and the
    target come from slightly different distributions.
  - At 27B, collect the moments with the same windowing as the Hessians the metric uses.
- **C2. Ledger.** `eval/pools/ledger.jsonl` has no entries for: the moment collections (705 qad docs), heal training
  (912 qad docs), heal's 48 early-stopping search docs, or the c18 / snap H_out collections (qad). AGENTS.md requires
  them.
- **C3. Silent `--hout` skip.** `ourq.py:249-250` skips `--hout` without a word when a key's file is missing. The stack is
  fine (proxy.jsonl: hout 48/48 for k3edyasym, k3edyasym_s1, k2edyasym_s2), but `qo` should assert it.
- **C4. No-op gate coverage.** #87's no-op gate covers K3 seed 0 only. K2 and seeds 1/2 rely on the rs controls
  (ratios 0.99–1.00), which is consistent.

## Checked and cleared

**Leakage** (`idcheck.py`, outputs of all CPU checks in `checks.txt`)
- calib-meta order = calib.jsonl order: 0 / 4,789 source mismatches.
- g08-search1: 173 docs, all in the search pool. g08-fit1: 87 docs, all in fit. Tokens are identical to
  calib-meta[orig], and the two goldens share no doc.
- Moment windows: 705 docs, all qad, none in either golden.
- Heal: trains on qad only. Its held-out set is 48 search docs, none in g08-search1 (datacheck.txt). The oracle arms are
  in-sample by design.
- swiglufit / α* fit on g08-fit1 (fit pool) and are judged on search. swiglufit's own held-out half matches in-sample
  (K3 0.01544 vs 0.01523, K2 0.06134 vs 0.06036).
- H_out collections (c18, snap8192 → dv8192) read qad windows only (`v3/yaqa/collect.py:99`).

**Math** (`recompute.py`)
- mkdenoise's G and Ft match an independent numpy recomputation to ~3e-8 relative (f32 storage), at L3 o_proj and
  L14 down.
- The objective identity holds to 1e-13, and the gradient at W·F is ~0.
- So the target is the right GPTAQ / QEP minimiser for ρ = 0, and correctly interpolated at ρ = 0.5.
- F is ~97% off-diagonal (diagonal mean 1.001–1.003; ‖WF‖/‖W‖ 0.99–1.02). It is not mainly a gain fix, so it overlaps
  little with #85's shrink restore. (The negative tr_C does show the siblings shrink the writers' inputs.)

**`--wpre`** (`ourq.py:183-195`)
- W·F is applied in the checkpoint basis, before `quantize_exl3`'s rotation, regularisation and scales.
- Upstream `refit_scales` refits against that same target (`weight_orig`) in the `--hin` metric.

**Eval = pack** (`packcheck.py`)
- The saved `wq`, un-rotated with the pack's own f16 suh / svh, lands on the i8 codebook grid: residual RMS ≈ 0.015
  steps, max < 0.1. That holds for base, asym, edy+asym and K2 asym (3 tensor types).
- So no eval-only scale effect exists, and the asym weights pack like any other quant.

**Assembly** (all 20 `wqall` dirs of #87 and the stack)
- Each has 150 files: the 48 writers → the arm's own `wq`, the other 102 → the matching-seed base.
- Stamps differ from the base only in `--hin` / `--wpre` / `--only` (and the controls' seed offset).
- The stack carries `--hout` (same dir, `--sigma-reg-out 0.01`) on all 48.

**Comparability**
- Same golden, model and evalq for every arm and control.
- `evalq.py` / `common.py` are unchanged since 2026-11-02 09:10 UTC, before both goldens and every eval.
- The base score files have g08-search1's row count. Per-seed pairing in `summ.py` is correct.

**Healing fold storability**
- r → suh (per input channel), c → svh (per output channel): exact up to f16.
- Per-tensor suh already differs inside the q/k/v and gate/up groups in the base packs (shared signs, magnitudes ±7–13%),
  so the kernels already take per-tensor suh.
- The q_norm / k_norm / final-norm gains go into the checkpoint norms (bf16, (1+w)·e^g − 1).
- The gauges (down r ≡ up c; v c ⊂ o r) are exact.
- No trained parameter is unstorable. (The 27B converter must take the modified norms.)

**Scale refits** (`fsutil.py`, `refitq.py`)
- Rescaling rows and columns of the dequantized weight is exactly suh *= r, svh *= c (`edit_shards`).

## Files

| file | what |
|---|---|
| `checks.txt` | raw outputs of the four CPU checks below |
| `idcheck.py` | pool / golden / moment-window provenance (CPU) |
| `packcheck.py` | saved wq vs the pack's suh / svh + the i8 grid (CPU) |
| `recompute.py` | independent G / F recomputation + objective identity (CPU) |
| `fsim.py` | (F − I) similarity across sibling seeds / K (CPU) |
| `d1103-heal-norms.patch`, `patchtest.sh`, `patchtest.txt` | fix for A1 and its no-op / behaviour test |
| `healfix.sh` | post-hoc re-score of d1103's heals with their norms (GPU, lead) |
| `xseed.sh`, `xseed-dry.txt` | B1's transfer test (GPU, lead); CPU assembly check |
