# data/strata190 — #190 cooldown pair s190a (search) / s190b (confirm), built (w169, 2026-11-14 11:40 PM CST restart)

RULES-cooldown amendment (11-14 11:40 PM CST): one 2×-Latin pair matched to s186a / s186b. Built on ds8 (CPU, nice), copied to
box2, staged (not ready, not live). Nothing was registered in the live registry, no driver was touched.

## Pair (from `plan190.json`; indices only, no text)
| part | docs / pieces | N | L | N gen | L gen (match) | N share of gen | scripts (N) | max script / N gen | scored positions |
|---|---|---|---|---|---|---|---|---|---|
| s190a-c | 104 calib | 7 | 97 | 6,488 | 101,174 (s186a 50,052: 2.02×) | 6.0 % | 5 | 30.6 % | 125,174 (s186a 64,863) |
| s190b-c | 103 calib | 8 | 95 | 6,287 | 87,588 (s186b 43,420: 2.02×) | 6.7 % | 4 | 39.2 % | 111,031 (s186b 64,478) |
| s190a-x | 195 xc8k-b1 | 24 | 171 | 25,526 | 245,112 (s186a 121,926: 2.01×) | 9.4 % | 6 | 22.2 % | — |
| s190b-x | 214 xc8k-b1 | 17 | 197 | 24,714 | 216,502 (s186b 108,030: 2.00×) | 10.2 % | 6 | 33.1 % | — |

Mixes (goldmix, x / c interleaved): s190a 299 docs / 422,278 rows (378,300 gen rows); s190b 317 docs / 375,453 rows (335,091 gen rows).
c and x are distinct within each part and across the pair (207 calib docs, 409 pieces). N sizes: c 15 free N documents split 7 + 8
(12,775 N gen in all, the whole free pool); x 41 unused b1 N pieces split 24 + 17 (50,240 N gen in all).

## Supply (`supply190.json`; the restart's run gives the same numbers as the 11:25 PM run, every field)
- Calib: 1,134 fit + search documents, 922 excluded (s171a–d, s173a–f, s186a–f, t183, t186n, final750 c1/c3, every ready /
  active / staged registry c document), 425 free: 15 N (ARABIC 6, CJK 4, CYRILLIC 2, HEBREW, HANGUL, BENGALI), 410 L.
- xcorp: xc8k-b1 2,542 unused pieces, 41 of them N (CYRILLIC 9, HEBREW 5, ARABIC 15, CJK 8, GREEK 2, THAI, HANGUL); 2,501 L.
  The 11 unused N pieces in the b2 batches are not used (no gen mask / pickx path on ds8 for them yet).
- Exclusion: calib from every earlier golden's `orig` (s171, s173, s186, t183, t186n, final750 c1/c3) and the registry; xcorp from
  xsupply186's used flags and every earlier x golden's `xdoc` (global identity).

## Checks (`check190.out`, final run on the current registry: ALL PASS)
- Two c parts, two x parts; c and x distinct within and across the pair; every c document fit or search, outside the exclusion set;
  every x piece xc8k-b1, outside the exclusion set (xdoc); the N documents / pieces of the pair are exactly the free / unused N ones,
  each once.
- ≥ 4 non-Latin scripts per part (5, 4, 6, 6); no script above half of a part's N gen (max 39 %).
- L gen of each part ≥ 1.9× the matching s186 part's (2.02, 2.02, 2.01, 2.00).
- N share: printed, not banded (amendment); 6.0 / 6.7 % (c), 9.4 / 10.2 % (x).
- `verify190.json` (after the build): every built golden holds exactly the plan's documents; every document carries stratum and
  script; gen.u8 sizes match the rows; the mixes hold each part's documents once. bad = [].

## Build and copies
- Build: `build190.py` (w164's `build186.py` without the t186n monitor; `pickx186.py` from run186w, unchanged). Built on ds8 with
  `/data/q27/venv/bin/python3` (the system python has no transformers; the first attempt failed on that, its c part was kept).
- Copies: `/data/q27/w169/slices/golden-s190{a,b}{,-c,-x}` on ds8 and box2 (same path, rsync from ds8, `dbox` key). md5 of every
  file (36 per side) identical on both: `md5-golden-s190.txt`. Disk after the copy: ds8 488 GB free, box2 329 GB free (6.2 GB copied).
- `build.log`: the build log (the first failed run and the rerun, both shown).

## Ledger and registry (staged)
- Ledger: 207 rows added to `eval/pools/ledger.jsonl` (`ledger_add.py`, roles `ledger-roles-190.json`, date 2026-11-14): golden-s190a-c
  104 rows, golden-s190b-c 103, role `select` (the ledger's search role; the slice's search / confirm role is in the note).
  `ledger-roles-190.json` is not merged into `eval/pools/ledger-roles.json` (lead's call: needed only for a mkledger rebuild).
- Registry: `/workspace/rad-local/workers/w169/registry-staged-190.json` = the live registry (md5 8366d096…, 11:46 PM CST) +
  s190a (status staged) + s190b (status staged, `"role": "confirm"`); live slices and doc_decisions unchanged. Calib doc ids
  checked against every other slice (none). x doc ids are `x:<source>:<orig>`; five of them also occur as strings in s173c / s173d /
  s173e / s173f for DIFFERENT pieces (xdoc differs, e.g. `x:fineweb-edu:474`: 41799 in s173c, 2281 in s190a). The x identity is the
  xdoc (checked above); the registry's doc_decisions for x ids are counts only (no draw reads them).
- Candidate config: `/workspace/rad-local/workers/w169/driver-190.json` = the live driver.json (md5 4a87755c…) with
  `confirm.pairs` + `["s190a", "s190b"]` (now `[s186a,s186b], [s186c,s186d], [s190a,s190b]`); nothing else changed.
- Switch procedure: `search/v3/run190/SWITCH-190.md`.

## Reruns
`run190/README.md` (rerun order, paths). Plan inputs as of 11:46 PM CST; `draw190.py` re-run on the newer registry gives the
same plan (the s186c decision counts moved; no c or x selection changed).

## attempt1/ (the first run, superseded)
The six-pair dry run of 11:25 PM CST (supply check, draw, check, README): `attempt1/`. Its checks failed (x parts: 1 of 6; the size
checks); nothing from it was built or registered.
