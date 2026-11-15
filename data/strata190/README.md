# data/strata190 — #190 supply check for the 2× cooldown slices (w169, 2026-11-14 night)

Question (RULES-cooldown §1): can the calib c side and the xcorp x side give six 2× slices s190a–f (c ~90k scored positions,
x 260 pieces, N share 15.5–19 % c / 15–20 % x, ≥ 6 scripts, ≥ 1.9× the matching s186 part)? Answer: **no. The x side gives
one x part, so no pair can be built, and nothing was built** (RULES-cooldown: "build as many full pairs as possible (≥ 2)
and report"; never shrink a slice).

## Files
| file | what |
|---|---|
| `supply190.json` | counts only: the exclusion sizes per source, the c pool (free docs, N docs by script, N gen), the x pool (unused pieces by batch, N pieces, N gen, mean gen per piece) and the per-2× x part arithmetic |
| `draw190.err.txt` | the dry draw's summary (stderr): the six c parts, the one x part, x_short, calib stats |
| `plan190-dryrun.json` | the dry draw's plan (indices only: calib documents and xc8k-b1 pieces); **not built, not for use** |
| `check190.out` | check190 on the dry plan: 37 PASS, 8 FAIL (the x count / x_short, and the six size checks) |

Code: `search/v3/run190/` (`supply190.py`, `draw190.py` = draw186 with the 2× targets, the exclusion set and a stop on x
shortage, `check190.py` = check186 with the 2× sizes). Run on ds8 with `/data/q27/venv/bin/python3 -I`.

## Supply (ds8; snapshots of the live registry and the ledger taken 11-14 11:19 PM CST)
- **Exclusion (c):** 922 calib documents (s171a–d, s173a–f, s186a–f, t183, t186n, golden-final750 c1/c3, and every
  ready / active / staged registry c doc). Exclusion (x): the xsupply186 used flags + s186a–f x (780 pieces) + t186n x (2).
- **c side:** 1,134 fit + search documents, 425 free. Non-Latin (N) in the merged pool: 64 documents, of which **49 are already
  used** (t186n monitor 17, s186 c parts 16, s17x c parts 15, t183 1). **15 free N documents** (12,775 scored N gen positions,
  6 scripts; ARABIC 6 docs; largest script share 32 %). Calib has 223 N documents in all; 159 sit in verify / qad (not
  search pools).
- **c arithmetic:** a 2× c part needs ~13.2k N gen positions at 17 % (or 12.8k at the 15.5 % floor). The free N pool is
  exhausted, so the six parts can only be built by **reusing the same 15 N documents in every part** (dry run: 127 distinct
  c documents, 341 repeated uses). A part then tops out near 94k scored positions at the 15.5 % floor.
- **x side:** xc8k batches with a golden and metadata (b1, b2-c0/c1/c2): 2,794 unused pieces, of which 52 N (b1 41 with
  50,240 N gen, exact; b2 11 with ~14k, scored-position upper bound). A 2× x part at 15–16 % needs ~40 N pieces (~49k N
  gen) because the L pieces average ~1,270 gen. **Only one x part fits** (dry run: s190a, 260 pieces, 39 N, N share 15.7 %,
  7 scripts; s190b would have 2 N pieces left).
- **Not used:** tr32k batches (= our pi-sessions, never for selection), xc8k pieces outside the exclusion set's flags.

## Size check (≥ 1.9× the matching s186 part)
s186 c parts built at 59–69k scored positions (`data/strata186/check186.out`), so 1.9× needs 112–130k. The N band caps a c
part near 94k (above). The RULES-cooldown "~90k" target and the 1.9× check cannot both hold with this N pool.

## Re-run (when new N supply exists)
1. Draw: `nice -n 19 /data/q27/venv/bin/python3 -I /data/q27/w169/code/search/v3/run190/draw190.py PLAN.json 190` (ds8; reads the
   snapshots in `/data/q27/w169/in/`; refresh them from the live registry and `eval/pools/ledger.jsonl` first, and only with
   the driver stopped or between writes).
2. Check: `check190.py PLAN.json calib-kld-strata.json xsupply.jsonl gm-b1.u8` (all six parts, x_short null, sizes).
3. Build (not written yet): a `build190.py` from `build186.py` (its t186n block and the 130-piece assumptions removed).
