# data/strata190cd — #190-02 pair s190c (search) / s190d (confirm) (w176, 2026-11-15)

**Status: built, checked, verified, copied (ds8 and box2, md5 identical), ledger rows added, staged registry and config
written. Nothing live was touched.** The lead decides when to activate (see `search/v3/run190/SWITCH-190cd.md`).

## The rule (RULES-cooldown 1'', 5:15 AM CST; widened by the 5:40 AM CST amendment, option A)
- **Latin:** fresh (outside every earlier slice, s186b/d/e/f included), 2× the matching s186 part's L gen (s190c ↔ s186a,
  s190d ↔ s186b), least used first; the xcorp L pieces are unused xc8k-b1 pieces.
- **N, calib (5:40):** documents of the retired slices s173a–e, s186a, s186c and of the retired s186b, s186d, s186e, s186f.
  Never a document held by s190a / s190b (live), s173f (staged), s171a–d, t183, t186n, golden-final750, verify, or any other
  non-retired registry slice. Least used first. Within the pair: distinct.
- **N, xcorp:** the 5:15 rule, unchanged: retired s173a–e / s186a / s186c pieces, and the 11 unused xc8k-b2 N pieces (gen masks
  by `genmask181.py make`, all three batches built). The 5:40 amendment names calib documents, so the x parts are the ones built
  on 11-15 at 11:17 AM CST. Their plan is identical on the current registry (checked piece by piece).
- Each N part: ≥ 4 non-Latin scripts **with N gen**, no script above 1/2 of the part's N gen, N gen near s190a/b's (c 6,388).
- **Choice (w176):** an N unit needs N gen > 0. The first draft had calib doc 520 (CYRILLIC, 4,095 rows, **0 gen**) in s190c:
  it adds scored rows and no N signal, so it is out, and scripts count only with gen. This changed the c draw (5 N docs in
  each part, not 6 and 7).

## Result (`plan190cd.json`, `check190cd.out` ALL PASS 41/41, `verify190cd.json` no mismatch)
| part | docs / pieces | N | L | L gen (match) | N gen (share) | N scripts (gen) | max script / N gen |
|---|---|---|---|---|---|---|---|
| s190c-c (search) | 112 calib | 5 | 107 | 100,564 (2.01× s186a 50,052) | 6,368 (6.0 %) | 5 | 38 % |
| s190d-c (confirm) | 63 calib | 5 | 58 | 86,941 (2.00× s186b 43,420) | 6,693 (7.2 %) | 4 | 42 % |
| s190c-x | 201 xc8k pieces | 24 (5 fresh b2 + 19 reused) | 177 | 244,586 (2.01× s186a) | 28,152 (10.3 %) | 7 | 29 % |
| s190d-x | 193 xc8k pieces | 24 (6 fresh b2 + 18 reused) | 169 | 216,672 (2.01× s186b) | 22,191 (9.3 %) | 8 | 28 % |

Mixes (goldmix, seeds 1900 / 1901): `golden-s190c` 313 docs, 421,500 rows (gen 379,670 = planned); `golden-s190d` 256 docs,
367,369 rows (gen 332,497 = planned).

**Pool facts (`supply190cd.json`, `draw190cd.err.txt`):** the fit + search pool has 64 N documents: 46 held by an excluded
slice (s190a/b 15, s173f, s171, t183, t186n, final750) and **18 eligible** (17 with N gen; 520 is the zero-gen one). Under
the strict 5:15 rule only 2 were eligible, which is why run 1 was blocked (`run1-*`). Holders of the 17 pool documents:
s186a 11, s186c 9, s186e 7, s186d 7, s186f 7, s186b 6, s173d 1, s173e 1 (a document can be in several).

**Within the pair:** c documents 175 uses, 175 distinct; x pieces 394, 394 distinct; the c N sets (5 + 5) are disjoint.
Latin c documents are outside every earlier slice (0 held). x disjointness is by xdoc (check190cd), not by the id string
(38 x id strings also occur in other slices as different pieces, as with s190a/b).

## Verify (`verify190cd.py`, the same checks on w169's built s190a-c / s190b-c first: 0 mismatches, so the c check is validated)
- x: tokens, top-k bytes and rows.f32 identical to the xcorp batch per piece; gen = plan; labels = xsupply table.
- c: each document = the calib document's first min(tokens, 4096) tokens (score_from equal); tokens, top-k bytes, rows.f32
  and gen.u8 equal the calib golden's rows; stratum / script = calib strata; gen sums = plan (N 6,368 / 6,693).
- mix: documents keyed (c, orig) and (x, batch, orig); rows and gen = the parts' planned gen.

## Copies and ledger
- `md5-golden-s190cd.txt`: 36 files (the 3 golden directories of each part, 5 files each, plus `golden-s190{c,d}-c.jsonl`,
  `golden-s190{c,d}-x.jsonl`, `golden-s190{c,d}-x.docs`). Identical on ds8 and box2 (`/data/q27/w176/slices/`, box2 free 305 GB after the copy).
- Ledger: 175 rows (role `select`, line v3, date 2026-11-15; roles in `search/v3/run190/ledger-roles-190cd.json`), 27,930 → 28,105
  lines; the old prefix is byte-identical (`eval/pools/ledger.jsonl`).
- Staged: `workers/w176/registry-staged.json` (live registry at md5 `fa61256a` + s190c staged, s190d staged role confirm; no other key
  changed) and `workers/w176/driver-staged.json` (live driver `f01e3797` with `confirm.pairs` + `["s190c","s190d"]`; nothing else).

## Registry notes for the lead
- In the live registry s186b, s186d, s186e and s186f are still `staged` (roles confirm / reselect), not `retired`. The amendment treats
  them as retired. Checked in the driver log: the last confirmation on s186b is 11-14 21:51 CST and on s186d 11-15 00:20 CST; no
  tentative waits on either. The live config still lists `[s186a,s186b]` and `[s186c,s186d]` (harmless while idle).
- s190c's GREEK part is 243 gen (small). It passes the ≥ 4 scripts rule as written; flagged, not changed.

## Files
- final: `plan190cd.json`, `check190cd.out`, `verify190cd.json`, `supply190cd.json`, `draw190cd.err.txt`, `build-cd.out`,
  `build.log`, `md5-golden-s190cd.txt`.
- run 1 (superseded, kept for the record): `run1-plan190cd-strict.json` (blocked), `run1-plan190cd-optA.json` (a narrower option A: its
  pool missed the s186b/d/e/f-only documents, and its code did not exclude the staged s173f; none of its N documents is in s173f, so the
  plan itself is unaffected; the zero-gen doc 520 is in it), `run1-check190cd.out`,
  `run1-draw190cd-strict.err.txt`, `run1-draw190cd-optA.err.txt`, `run1-supply190cd-strict.json`, `run1-verify190cd-x.json`,
  `run1-build-x.log`, `run1-md5-golden-s190cd-x.txt`, `run1-md5-x-box2.txt`. The x parts of the final plan are the run-1 x parts.
- Code (search/v3/run190/): `draw190cd.py`, `check190cd.py`, `pickx190cd.py`, `build190cd.py`, `verify190cd.py`, `stage190cd.py`,
  `ledger-roles-190cd.json`, `SWITCH-190cd.md`. draw190.py, build190.py, check190.py and the other w169 files are unchanged.
- ds8 `/data/q27/w176/`: `strata/` (plans, masks, logs), `slices/` (the 36 files), `in/` (registry snapshot `registry-live-fa61256a.json`,
  ledger snapshot), `code/` (the run190 tree).
