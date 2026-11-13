# search/v3/picks145 — RULES-picks implemented + the final-pick slice plan (w145, 2026-11-13)

Rule: `../RULES-picks.md` (d0149de, user 11-13 12:30 PM CST; not changed here). Code: `../seqrule.py`, `../driver.py`,
`../slices.py`, `../pickstat.py` (new), `../finalpick.py` (rewritten), `../test_picks.py`; hosts:
`../earlystop-picks.patch`. Replay data: `data/searchv3/w145/`. CPU only; no card time, **no ver4 read**.

## What the code does
- **In-search (driver + hosts):** every per-part sum in seqrule is the **capped statistic (b)**: per-document weights
  (rule key `"w"`) = each document's incumbent KL capped at 20 % of its part's *whole-slice* incumbent total (heavier
  documents scaled down to the cap, in A and B alike; exact water-filling). The weights depend only on the incumbent,
  so they are fixed before the candidate's first document and the hosts' early stop runs on (b) unchanged
  (`score_watch` already passes the rule through; only `rxrun/earlystop.py` = seqrule.py must be restaged:
  `earlystop-picks.patch`, checked to reproduce seqrule.py byte for byte). Accept = the existing rule (size-adjusted,
  equal-weight per-part mean, every part better, B p2 w5 early stop) **and the (c) guard**: in every part the
  size-adjusted per-document mean log(B_d / A_d) has the sign of the size-adjusted (b) (no CI). A guard reject is
  recorded as `"decision": "reject", "why": "guard"`. CONF `cap` (0.2) / `guard` ("docmean"); null = the #171 rule.
  Decision events also carry `pooled_a` and `docmean` per part.
- **Monitor:** each read vs the size's best so far: (b) and (c) with paired bootstrap CIs (10,000 draws; event
  `"stats"`, the pooled (a) beside, never alone). The "best" read and the alert use the size-adjusted capped score
  (each read's (b) vs the size's first read in the mode, on its log KLD).
- **Rotation per slice** (CONF `rotate_unit` "slice", default): 25 decisions on a slice, all sizes together
  ("size" = the #171 rule). **s171b retired** in `rad-local/workers/w131/run/registry.json` (`slices.py retire`; backup
  `rad-local/workers/w145/registry.json.bak-pre-retire`).
- **finalpick.py** (RULES-picks "Final picks"): decision on (a) pooled over both parts, size-adjusted with the frozen b;
  guard: the pick's (c) must not be significantly worse (95 % CI of the size-adjusted difference wholly > 0) than any
  other finalist in either part, else the next best (a) is checked; FLAG any document > 20 % of a part's KL; output per
  band: KLD per part / pooled, (a) pooled and (a) / (b) / (c) per part vs the pick with CIs, the pick vs the runner-up
  and vs SPEC `"default"` (the re-solve's pick). CIs by paired bootstrap over units: SPEC `"units"` per part (a label
  list or the golden's kld.json: label = source:orig, so a trace's chunks resample together), the same draws for every
  pair. (b) in a pair takes its weights from the reference (the other candidate).

## Readings to confirm (lead)
1. The (c) guard is compared **size-adjusted** (c_p + off vs d_p + off), like (b). Raw (c) would be a weaker guard for
   bigger candidates.
2. finalpick's (a) is pooled **over both parts** (the board's estimator over all the slice's positions; parts sized
   below), as the old finalpick; per-part (a) is reported.
3. Final guard vs **every other finalist** (as the old calib veto), not only vs the runner-up / default.
4. After retiring s171b the registry's next ready slices are **s171c / s171d**, which #172 / #178 already used for
   24 decisions each (w141: retire both): a restarted driver would activate s171c. Build fresh slices first.

## Tests (`python3 -I search/v3/test_picks.py`, 9 / 9 PASS, seconds)
capped weights = readout.py's where its loop converges (exact elsewhere) and bootstrap copies act as repeated
documents; no `w` / `guard` = the old earlystop.py's numbers exactly (30 random runs); one dominant document (50 % of
A's KL, B −10 % on it, +3 % on the other 59): (a) better, (b) / (c) worse → the old rule accepts, the new one rejects
(also as a two-part slice); heavy-tail gain under the cap: (b) better, (c) worse → guard rejects, off → accepts, a
size offset that flips (c) → accepts; short-document noise (240 short noisy + 60 long docs 2 % better): (a) / (b) CIs
< 0, (c) CI 3× wider and covers 0 → finalpick picks the better one, no flag; finalpick two parts: D wins (a) through
a 35 % x document → FLAG + guard passes it over (x (c) significantly worse), E picked, vs default CI < 0; size
adjustment +0.05 bpw at b 1 flips the pick; traces as units (equal copy counts per trace, CI 1.3× wider than per
chunk); early stop through the JSON early-stop block = the driver's rule; rotation per slice + retire.
CPU stand-in (`standin.py` si145 on colo, 401 evals, cap + guard + per-slice rotation on): **STANDIN PASS** (7 / 7: true KL −27.6 / −35.6 / −12.2 % at 2.0 / 3.0 / 4.5; 0 of 48 accepts truly worse; 240 of 280 rejects stopped early, 11.7 vs 21.8 card-h; fit corr 0.994; 5 rotations; restart; 48 monitor reads with (b) / (c) CIs). `data/searchv3/w145/standin-si145.out.gz`.

## Replay on #171 (`data/searchv3/w145/replay145.json`; first-order: the incumbent chain #171 actually had)
The #171 rule recomputed reproduces all 527 recorded decisions. Under RULES-picks' in-search rule:

| #171 decision | n | under (b) + guard |
|---|---|---|
| accept | 71 | **38 accept**; 29 guard rejects (c part 15, x 7, both 7; failing (c) + off +0.0002 … +0.026), 3 (b) not better on a part, 1 stopped early |
| accept, (b) without guard | 71 | 67 accept (w141's capped count) |
| reject, whole slice | 102 | 1 becomes an accept (6.5 solve-00612: x (a) +0.0034, (b) −0.0028, (c) −0.015) |
| reject, stopped early | 354 | 292 still stopped; 62 undetermined (the capped statistic would have kept scoring) |

Changed accepts by kind: pair 14 / 34, explore 9 / 16, block 4 / 11, partial 3 / 5, thompson 2 / 3, solve 1 / 2. Report
only: nothing is decided from this.

## Final-pick slice plan (not run; `plan145.py`, ledger + manifests only, ver4 not opened)

| part | what | units | scored positions | provenance |
|---|---|---|---|---|
| c1 | `golden-ver4` (ds8, frozen): 90 short docs (16k cap) + trace 4723's first 16k | 91 | ~205k (ledger: 189,191 + ~16k) | role final, never read |
| c2 | trace 4723 (requests) from 16384 in ≤ 32k chunks (`eval/traces` manifest boundaries), **first ~4 chunks** | 1 with c1's 4723 doc | ~120k rows (gen-masked fewer) | ledger `golden-ver4-long` (final); needs its bf16 golden (TP2 on ds8, ~15 min) |
| c3 | reused verify short docs: no 27B (v3 / v3.1) use, not ver3 / ver4 / headx / reserve / fit (`plan145.py --c3 150000`, seed 145) | 123 | 150,527 (cap 8,192) | prior uses: V1 / V2 goldens (calB/calBbig, ver1, ver2) and 0.8B g08 checks only; cut from the published calib golden (no bf16 run); ledger role final |
| x | fresh xcorp: `pickx.py` from night1105 `xc8k-b2-c2` (416 free pieces, 503k positions heavy-capped), heavy categories cut to 2,048, exclusions = w129/excl + hd122-x + s129-x + s171a–d x (as mkslices171) | ~210 | ~255k | never scored; registry / corpora note (non-calib) |
| total | | ~425 units | **~750k** (c ~475k, x ~255k) | |

- **Why c2 is cut:** calib's 8 traces are 19.5 % of its positions (public ~29 %). The whole 4723 (c1's 16k + 229k)
  would be ~33 % of a 750k slice in ONE resampling unit: its CI would be the trace's. ~4 chunks keep the trace ~18 %.
  The lead may prefer the whole trace (then ~860k, still ≤ 1M).
- **c3 pool:** 421 verify docs (628k positions at 8k) have no 27B use; 48 untouched reserve docs (43k) stay unspent
  (audit's "28 untouched" counted 0.8B-checked ones as used); ver3 (104, spent), w119-headx-verify (100) and touched
  reserve (88) are excluded.
- **x caveat:** b2 has almost no agentic / long pieces left free (3 / 1); `xc8k-b1` (3,389 pieces, 5.0M) has no meta on
  ds8. If heavy categories should be in the x part, draw part of it from b1 where its meta lives.
- **Cost (ds8, A8, one card per candidate):** measured A8 scoring on ds8 = ~658 positions/s (w132: ver3-8k 148,808 in
  226 s, s171b 218,601 in 332 s, incl. load) → 750k ≈ 19 min + ~0.5 min mix; the 32k chunks are unmeasured on ds8
  (allow 0.7–0.9× rate: +1–2 min) → **~20–22 min ≈ 0.35 card-h per candidate**. Top 5 + the default per band ≈ 2 card-h;
  12 bands ≈ 25 card-h ≈ 3.2 h on ds8's 8 cards. Expected precision ~±0.55 % paired SE pooled (100k ≈ ±1.5 %), wider
  where the trace unit dominates.
- Ledger before the first read: rows for c2 / c3 (role final) via `ledger_add.py` (roles first), x pieces in the
  slice's docs file; the frozen b (`driver.py --freeze-b`) before any ver4 score.
