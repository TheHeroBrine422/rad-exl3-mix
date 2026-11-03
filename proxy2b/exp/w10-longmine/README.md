# w10-longmine: mine the existing 27B long-trace evals for family × position effects (ranked.md I1, measure L3)

**Question.** V1/V2 scored about 17 containers and mixes on the two 250k verify traces: `golden-long2` (calib 905
and 2675) and its capped copies `golden-long2-cCAP`, per-row scores in iceman's `/root/longscores/`. Does a mix's
KL grow with context position faster when its attention or its GDN tensors get fewer bits than the rest of the
trunk? If so, a long-position family probe on the 27B (I2 / measure L1, ~1 card-h) and a position term in kfit2 are
worth building.

**Mechanism.** GDN state error and attention-input error accumulate over context. The question is whether the
allocation needs a long-context term at all, given that selection only sees ≤ 2k positions. #52 / B4 found
whole-mix ratios flat by position, but its quants did not separate families.

**Method (`longmine.py`).**
- **Pairing.** Rows are keyed by (orig doc, position), so capped and full goldens pair. Each arm's KL is taken
  relative to a reference arm (default: the first line, e.g. turboderp sc4.00-e4).
- **Per arm:**
  - the ratio to the reference by band (0–2k / 2–8k / 8–32k / 32–64k / 64–128k / 128k+);
  - the compounding index c = ln(ratio at ≥ 32k / ratio at < 8k);
  - per-family bits from the arm's .mix (or uniform container) and V2's per-tensor byte table
    (`data/v2/radls.jsonl`; ours-K / hd-K containers at K).
- **Across arms:** c ~ trunk bits + (attention − trunk bits) + (GDN − trunk bits), least squares.
  - CIs come from a paired block bootstrap: 1,024-position blocks resampled within each document's
    short / mid / long region, the same resample for every arm.
- **`mkmanifest.py`** builds the arm list from the score files:
  - the golden by row count;
  - the bits source as a mix of the tag's name first, then a uniform container whose name ends the tag;
  - anything else is listed as UNRESOLVED and never guessed.
- **Public scores are excluded on purpose:** `/root/fullscores` holds many more long traces, but public only
  reports and never informs a decision.

## Draft pass rule (for the lead to commit before data)
- **Flag** the attention or GDN contrast if its exp(β) per bit has a 95% CI excluding 1 **and** a point estimate
  above ×1.25 or below ×0.8 (ranked.md: "CI excludes 1 by > 25%").
- **Any flag** → I2 / measure L1 (the 27B long-position probe) moves up; the 0.8B family × position table from
  `../w10-longpos` (w10-hout-attrib's subset arms on the long golden) is the cross-check before the card-hour is
  spent.
- **No flag** → I2 is not justified by I1. Given the power limit below, that is **not** evidence of no effect.
- Fewer than 6 arms with long rows → no verdict.

## Power: why I expect this to be weak, measured before any data (`designcheck.py` → `designcheck.txt`)
- **Contrast spread.** Over the 26 arms whose bit layouts are in the repo (16 turboderp/plain containers, 10 V2
  knap mixes), attention − trunk bits has SD **0.17 bits** and GDN − trunk **0.15 bits** (correlation +0.38). The
  plain containers have both contrasts at 0 or +0.57.
- **Minimum detectable effect.** With c's per-arm noise σ_c, SE(β) ≈ σ_c / (0.16 · √n) ≈ 1.6 σ_c for 16 arms.
  - The B4 per-document spreads (e.g. V2 3.00 at 3.92× vs 2.73× the reference in one bin on the two docs) put
    σ_c around 0.05–0.1.
  - That gives a 95% half-width of ±0.15–0.3 in β, so only effects of about ×1.35–×1.8 per bit are detectable.
- **Documents.** There are two (905, 2675); the block bootstrap covers within-trace noise, not trace-to-trace
  variation.
- **Recommendation:** run it (CPU, a minute), but **rank I1 down**. The informative version is the 0.8B
  family × position test: designed single-family arms at fixed codes, 22 documents, 2–64k positions
  (`../w10-longpos` + `../w10-hout-attrib`, ~45 min of 4070 evals).

## Run (iceman, CPU only, marked)

```
# iceman is down at writing (ssh refused, 2026-11-03); when it is back, copy this folder to /root/w10-longmine, then
bash /root/w10-longmine/run-iceman.sh      # CPU only (CUDA_VISIBLE_DEVICES=, nice 19, 4 threads), takes /root/ram.lock
```
- **First run:** writes `/root/w10-longmine/manifest.tsv` and stops if any score file is unresolved. V2 release
  containers (`v2-<size>`) need their mix path written by hand: longverify.sh does not record the mix.
- **Second run:** writes `result.txt` and `longmine.json`.
- Env overrides: `LM_SCORES`, `LM_GOLDENS`, `LM_MIXDIRS`, `LM_RADLS`, `LM_REF`.
- **GPU time: none.** The score files are not reachable from here, so their existence after the iceman crash
  (#58) is unverified.

## What the result changes
- **Flag:** I2 / L1 runs (27B, ~1 card-h, 13 long evals), and kfit2 gets a family × position factor (measure L2).
- **No flag:** nothing changes, and I1 is closed. Long-context sensitivity is then judged by w10-longpos's 0.8B
  family × position table.

## Self-test
`selftest/selftest.txt`, **ALL PASS**: synthetic 2 × 250k traces, 15 mixes over the real container byte table.
- **Planted attention effect β = 0.3:** flagged at ×1.343 (planted ×1.350); GDN not flagged.
- **Capped arm:** aligned by (orig, position).
- **β = 0:** nothing flagged. Its attention CI excludes 1 (×0.995), but the ×1.25 margin keeps it unflagged, as
  intended.
- **mkmanifest:** golden by row count; a V2 mix wins over the `sc4.00-e4` substring in its name; `v2-3.00` and an
  odd row count are listed UNRESOLVED.

No patches.

## Runtime cost (user rule, 2026-11-03)
n/a (analysis).
