# w01-genmask: Fisher loss on the scored positions only (ranked.md A1)

**Question.** The 27B diagonal-H_out collector (`collect.py` YAQA_DIAG, #33-13/#33-17) sums the sampled-label CE over
**every** position of a 2048-token qad window. That includes user turns, system prompts and tool outputs, which the
challenge never scores (`gen.u8` = 0). Does a loss restricted to the scored positions give a better diagonal at equal
sequences? Decide before the 27B night (hdiag27.sh).

**Mechanism.** H_out ≈ E[G Gᵀ] of the loss we care about. The score is forward KL on gen rows only, so the Fisher of
non-gen predictions is weight on outputs that never count. With sampled labels, the cross-position terms of G_b G_bᵀ
have zero mean, so masking the loss is a clean restriction to scored rows. Context still flows: every position runs,
and its activations feed the later scored ones. Counter-risk: fewer loss positions means a noisier diagonal at a fixed
sequence count. In the 1,024 windows the 0.8B and 27B runs draw, **74.2% of loss positions are scored**, and
**105 of 1,024 windows have none** (`../w01-windows`). So masking drops 26% of positions, mostly prompts and tool
output in agentic, SWE and chat documents. Against the corpus's gen mix, the drawn loss positions are at TV 0.291
unmasked and 0.261 masked. Examples: swe-rebench and swe-smith positions are 3.3× their corpus gen share, wildchat
2.65×, smoltalk2/longalign 70×.

**Implementation** (`genmask.patch` on `v3/yaqa/collect.py`):
- `YAQA_GENMASK=1`, together with YAQA_DIAG=1, builds a per-token scored flag from `<golden>/gen.u8` (next to
  `tokens.i32`; `score_from` respected) and multiplies the chunk CE by it.
- Labels are drawn at every position exactly as before (same RNG), so the masked run's scored positions get the
  unmasked run's labels.
- A window with no scored position is skipped but counted, so snapshot n covers the same windows as an unmasked run.
- The files carry `genmask: "1"` in their metadata.
- Off: unchanged; see the self-test.

## Arms (0.8B, i8, base settings + diagonal H_out, output damping 0.01, g08-search1)

| tag | H_out source | seeds |
|---|---|---|
| `k{3,2}gm{,_s1,_s2}` | **masked** 1K diagonal (`/data/yb/w01-genmask/cmask/hdiag-snap1024`) | 0 / 1 / 2 |
| `k{3,2}eB{,_s1,_s2}` (control) | unmasked 1K diagonal `c18/hdiag-snap1024`: same windows, seed and labels (yaqaFact7, #33-18; reused) | 0 / 1 / 2 |
| `k{3,2}base*` (reference) | none (existing) | 0 / 1 / 2 |

Seed s of the masked arm is paired with seed s of the control (same rotations; the two quants differ only in H_out).

## Draft pass rule (for the lead to commit before data)

Ratio = mean KL over 3 seeds, masked / unmasked.
- **PASS** (27B collection uses `YAQA_GENMASK=1`): at K3, ratio ≤ 0.975 **and** every masked seed below its
  seed-matched control; at K2, ratio ≤ 1.00.
- **FAIL** (keep the all-position loss): K3 ratio ≥ 1.00, or K2 ratio > 1.02.
- **Otherwise inconclusive:** keep the all-position loss (status quo; the masked loss has 26% fewer loss positions).
- **Reported beside it, not part of the rule:**
  - each arm vs base;
  - halves.json median / p5 of the masked diagonal at 1,024, vs c18's. If PASS and the masked median is < 0.995,
    amendment 10's stopping rule cannot be met at 1,024 on the 27B, and the qad pool has only 1,350 windows
    (`../w01-windows`).
- **GPU no-op gate (step 0):** the patched collector with the mask off reproduces c18's 128-sequence snapshot to
  rel L2 ≤ 1e-6 (bit-exact expected: #33-17's resume was), or nothing runs.

## Run

```
proxy2b/exp/w01-genmask/run.sh       # on the 4070; after genmask.patch is merged and yaqaFact7's c18 collection is done
```
- Results: `/data/yb/w01-genmask/result.txt` and `summary.json`.
- Idempotent: it skips finished steps, the collection resumes from `hdiag-state.pt`, and quants skip when their
  `.f32` exists.
- It reuses yaqaFact7's `k*eB*` / `k*base*` scores from `/data/yb`, and quantizes any that are missing into its own
  folder.

**GPU time:**
- no-op gate: ~3 min
- masked collection: ~25 min (1.4 s/seq, ~10% skipped)
- 6 quants + evals: ~30–40 min (~1 h total)
- +30–40 min if yaqaFact7's eB arms are not there yet

## What it changes
- PASS → `v3/yaqa/hdiag27.sh` exports `YAQA_GENMASK=1` for the 27B diagonal night (#33-13). Nothing else changes:
  same windows, same cost per sequence (skipped windows are free).
- FAIL / inconclusive → hdiag27.sh unchanged. A1 is closed for the 27B collection; healing (H2) and head statistics
  (E4) use their own masks anyway.
- Either way: the masked diagonal's halves agreement shows whether any gen-restricted estimator needs more than 1K
  sequences.

## Files
- `genmask.patch`: against the attached `v3/yaqa/collect.py` (39e3cd3). Applies with or without
  `../w01-failloud/collect-check.patch`, in either order, with the same result.
- `run.sh`, `summ.py`.
- `selftest/cpu_e2e.py` → `cpu_e2e.out` (**ALL PASS**). CPU end-to-end on colo with a tiny random Qwen3.5 (4 layers)
  and a synthetic golden, using the same CUDA→CPU text shim on both copies:
  - mask off: every tensor bit-exact with the original collector (hdiag, snapshots, -xw / -dy, halves.json).
  - all-ones mask = unmasked (rel L2 0).
  - mixed mask: the expected zero-gen windows are skipped and counted; per-window scored counts match the golden;
    the result differs from unmasked; metadata is correct.
  - It also runs the A8 checks of `../w01-failloud/collect-check.patch`.
- `selftest/masked_ce.py` → `masked_ce.out` (**ALL PASS**). The masked, chunked, checkpointed CE has exactly the
  gradient of the CE over the scored positions alone, including an empty mask.
- `selftest/summ_synth.py` → `summ_synth.out`: a dry run of `summ.py` on synthetic scores.

## Runtime cost (user rule, 2026-11-03)
none (changes the H_out collection only).
