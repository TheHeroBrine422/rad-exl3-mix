# w10-longpos: does the 0.8B diagonal-H_out gain hold past 2k positions? (ranked.md I4)

**Question.** Every 0.8B H_out result so far (#33-10 … #33-21) was scored on g08-search1, whose documents are capped
at 2,048 tokens. The diagonals themselves come from 2048-token qad windows that start with an empty state. The
private set is long: ~35% of the public scored positions sit in long agentic traces, and a 27B night is about to be
spent on this estimator. Does the gain hold at context positions > 2k, or does it fade where the state and KV carry
long context?

**Mechanism.** Errors in the GDN state and attention inputs accumulate over long contexts. A diagonal fitted on
short windows may weight the output channels that matter early in a document and miss the ones that matter deep in
it (STEPQuant / LeapQuant: GDN error grows with horizon). Against that, #52 / B4 found the KL ratio of 27B quants to
a reference flat by position (2 docs, quants that did not isolate anything).

**Design: fixed codes, new golden.**
- **New golden `/data/g08-long1`** (mkgold.py, the 0.8B bf16 reference):
  - every **search-pool** document of ≥ 8,192 tokens: 22 docs, 20 whole (8k–30k tokens), plus the two agent
    traces (ripgrep 1241, cobra 4029) cut at 65,536;
  - ~420k rows, **236k scored**: 32k at 0–2k, 100k at 2–8k, 81k at 8–32k, 24k at 32–64k (`glong-docs.table`);
  - document list frozen in `glong-docs.txt` (`sel.py`);
  - search pool: disjoint from the qad windows the diagonals came from, and from fit.
- **No new quants for the main question.** The existing seed-paired arms are re-scored on it: `k{3,2}base` and
  `k{3,2}edy` (Σdy², the 27B primary), seeds 0–2; `k{3,2}eB` (Sketch B) as a second estimator.
  - Seed s of an arm is paired with seed s of base (same rotations).
  - Only the golden is new, so no extra rounding noise enters beyond what #33-18 already had.
- **Family × position (measure L5) comes free.** Once w10-hout-attrib has run, its subset arms are re-scored too, so
  posgain.py prints each family's and depth third's gain by position band.

**Statistic (`posgain.py`).**
- **Bands** by context position: 0–2k ("short") and ≥ 2k ("long"; also 2–8k / 8–32k / 32–64k).
- **(a) Within-doc (primary).** It separates position from category: short and long positions inside the same
  documents.
  - Uses only documents with ≥ 500 scored rows in both bands. The agent traces have < 500 scored rows below 2k, so
    they enter only (b).
  - Per doc: ln(Σ KL_arm / Σ KL_base) in each band; averaged over docs and seeds.
  - Reports g_short, g_long, D = g_long − ⅔ g_short and D3 = g_long − ⅓ g_short.
- **(b) Pooled (what the score sees).** All 22 docs, traces included. The ≥ 2k gain is compared with the
  g08-search1 gain of the same arms and seeds.
- **SE:** documents bootstrapped (2,000 replicates, all seeds of a replicate share the documents), plus the seed
  spread SD/√n, added in quadrature.

## Draft pass rule (for the lead to commit before data)

Per K ∈ {3, 2}, primary arm `edy`:
- **DECAYS** if D3 < −2 SE (the long gain is detectably below a third of the short gain), or if the pooled long
  gain is < ⅓ of the g08-search1 gain with z ≤ −2.
- **HOLDS** if D ≥ −SE (not detectably below ⅔), and g_long ≥ 2 SE, and pooled long gain / g08-search1 gain ≥ ⅔.
- **PARTIAL** if D < −2 SE but not DECAYS: the gain holds at a measurably reduced size.
- **INCONCLUSIVE** otherwise.

**Combined over K:** DECAYS if either K decays. Otherwise INCONCLUSIVE if either is inconclusive, then PARTIAL if
either is partial, else HOLDS.

**INCONCLUSIVE (pre-registered):** two more seeds of `edy` (3, 4; their base seeds exist after yaqaFact12), then
decide on point estimates: DECAYS if D3 < 0; HOLDS if D ≥ 0 and the pooled ratio ≥ ⅔; else PARTIAL.

`eB` gets the same verdict, reported but not decisive. Family arms get the band table only.

## Run

```
proxy2b/exp/w10-longpos/run.sh       # 4070; waits for running ourq / evalq / mkgold; idempotent, resumable
```
- Results: `/data/yb/w10-longpos/result.txt`. Score files are `<tag>.glong.f32` next to it.
- **Golden build:** mkgold over ~420k positions, ~10–20 min. It is also the memory gate for the 65k traces under
  the 7.2 GB cap: on failure the script rebuilds with the traces cut at 32k (`glong-docs-c32k.txt`) and logs it.
  The 32–64k band then disappears; the rule is unchanged.
- **Evals:** 18 at ~1–1.5 min ≈ 25 min.
- **Extra seeds if inconclusive:** 4 quants + 8 evals ≈ 15 min.
- **With w10-hout-attrib's arms:** +28–36 evals ≈ 45 min.
- **Total:** ~1 h (2 h with the family arms).
- Disk: ~1.8 GB for the golden.
- **Ledger:** add `g08-long1` (0.8B reference, 22 search-pool docs, role "search / diagnostic", positions 0–65k of
  the two agent traces). Nothing about the 27B is selected on it.

## What the result changes
- **HOLDS:** the 27B collector's 2k fresh-state windows are adequate. Long-context collection variants (measure
  L6/L7, carried-state windows) drop in rank. The 27B swap test can be judged on short slices plus the usual final
  long check.
- **PARTIAL:** the 27B swap test (K1 in ranked §2) must include a long-position read (ver3-long, from 8k) with its
  own rule. L7 (carried state across windows, same cost) gets one 0.8B test: collect.py windows deep in long qad
  docs vs fresh ones.
- **DECAYS:** the diagonal's value is a short-context artefact. Before the 27B night, collect at deep positions (L7)
  and re-test here. If the 0.8B long gain cannot be recovered, the 27B pass rule must weight long positions by
  their share of the score.
- **Family × position (with w10-hout-attrib):** a family whose long-band share differs from its short-band share by
  > 0.15 at both K points to where long-context curvature differs. That replaces I1's weak 27B regression (see
  ../w10-longmine) as the input to I2 / measure L1.

## Files
- `sel.py`: document selection, from the published calib kld.json, pools-v3 and gen.u8.
- `glong-docs.txt` / `glong-docs-c32k.txt` / `glong-docs.table`: the frozen lists and gen counts.
- `posgain.py`: the statistic and verdict.
- `run.sh`: the run.

## Self-test
`selftest/selftest.txt`, **ALL PASS**. Synthetic golden (16 docs) with planted seed-paired gains:
- 9% everywhere → HOLDS (band gains recovered);
- 9% → 1% past 2k → DECAYS;
- 9% → 5% (0.56 of it, precisely measured) → PARTIAL;
- 9% / 6.5% with an 11% short-slice gain → INCONCLUSIVE, with exactly the pre-registered extra seeds listed.

`sel.py` ran on the real calib metadata (copied read-only from the 4070): 22 docs, 236,425 scored rows.

Not tested without a GPU: whether the 65k-token documents fit the 7.2 GB cap (the run's first step is the gate,
with the 32k fallback).

No patches.

## Runtime cost (user rule, 2026-11-03)
none (evaluation of H_out quants at long positions).
