# data/strata190ef — #190-03 pair s190e (search) / s190f (confirm) (w179, 2026-11-15)

**Status: built, checked (42 PASS, 0 FAIL), verified (no mismatch), copied (ds8 and box2, md5 identical), ledger rows added, staged
registry and config written. Nothing live was touched.** The lead decides when to activate (`search/v3/run190/SWITCH-190ef.md`).

## The rule
RULES-cooldown amendments 2026-11-15 **4:25 PM CST** (pair s190e / s190f: the N and x rules of 1'' / 5:40, the calib Latin shortfall
filled in the x parts so each slice's total L gen is 2× the matching s186 slice's c + x) and **4:40 PM CST** (Latin reuse for s190e:
retired search slices' and V2-era fit + search documents, least used first by V3 decisions; s190f keeps the freshest Latin).

- **Pair:** s190e (search, Latin 2× s186c) / s190f (confirm, Latin 2× s186d). The second s186 pair (a/b were used for s190c/d).
- **Latin, calib:** s190f takes the fresh documents first (outside every earlier slice, t183, t186n, final750, s171, s173f, verify,
  the live pair). s190e takes the rest of the fresh pool, then the reuse pool (retired search docs: s173a–e, s186a, s186c, s190a; V2-era:
  ledger lines containing `v2`, fit + search only), least used first by V3 decisions (ledger lines v3 / v3.1 plus `doc_decisions`).
  Nothing from the live pair s190c / s190d, t183, t186n, final750, s171, s173f, or any non-retired registry slice.
- **Latin, xcorp:** unused xc8k-b1 L pieces (fresh), stratified by category, grown until the slice total is 2× (the c shortfall or overshoot
  is in the x part).
- **N, calib:** documents of the retired slices s173a–e, s186a–f, s190a, and s190b (no pending confirmation: `pending190b.json`), least
  used first; at least 4 non-Latin scripts per part with N gen; no script above half of a part's N gen. Targets: c 6,388, x 25,120 N gen
  (the same as w176's, so the N amount is as s190c/d's).
- **N, xcorp:** pieces of the same retired slices (wide reading, see below); the 11 unused b2 N pieces were all used by s190c / s190d.
- **Seeds:** draw 1903; goldmix 1903 / 1904.

## Supply (`supply190ef.json`, `supply190ef.err.txt`; computed on ds8 before the draw)
| source | count | gen (positions) | note |
|---|---:|---:|---|
| calib Latin, fresh (fit + search, outside every earlier slice) | 53 docs | 65,290 | the s190f pool |
| calib Latin, reuse for s190e (retired search 364 + V2-era; union) | 493 docs | 524,196 | `supply190ef.json` `L_reuse_for_s190e` (of the 318 V2-era fit + search documents, 129 are in this pool: L, ≥ 16 scored positions, not held by an excluded slice) |
| xcorp Latin, unused xc8k-b1 | 1,787 pieces | 2,228,632 | enough for both x parts' shortfalls |
| calib N (retired-ok slices, 4:25 pool incl. s190a / s190b) | 22 docs | 21,985 | 7 scripts |
| xcorp N, wide (retired search / confirm / reselect + s190a/b) | 160 pieces | 191,634 | 9 scripts; as built |
| xcorp N, narrow (retired search only) | 60 pieces | 64,437 | 8 scripts; `supply190ef-xnarrow.out` |
| xcorp N, unused b2 | 0 | 0 | all 11 used by s190c / s190d |

## Result (`plan190ef.json`, `check190ef.out`, `verify190ef.json`)
| part | docs / pieces | N docs | L docs | N gen (share of part) | N scripts (max share) | L gen | own 2× target |
|---|---:|---:|---:|---:|---:|---:|---:|
| s190e-c (search) | 104 calib | 9 | 95 | 7,255 (7.5 %) | 4 (45 %) | 89,791 (2.10×) | 85,736 |
| s190f-c (confirm) | 60 calib | 7 | 53 | 7,581 (10.4 %) | 4 (28 %) | 65,290 (1.39×; all fresh) | 93,668 |
| s190e-x | 237 pieces | 15 reused | 222 | 26,509 (8.6 %) | 7 (26 %) | 283,423 | 280,149 (slice fill) |
| s190f-x | 313 pieces | 19 reused | 294 | 23,778 (6.3 %) | 8 (34 %) | 355,168 | 355,148 (slice fill) |

Slice totals (c + x L gen, 2× the matching s186 slice): s190e 373,214 vs 369,940 (2.018× of s186c's 184,970); s190f 420,458 vs
420,438 (2.000× of s186d's 210,219). Mixes: `golden-s190e` 341 docs, 450,276 rows, gen 406,978 (= planned); `golden-s190f` 373 docs,
513,651 rows, gen 451,817 (= planned). Mix-level N share 8.3 % / 6.9 % (s190c/d: about 9 %).

**Checks:** `check190ef.out`: 42 PASS, 0 FAIL, ALL PASS (the pair names and policy, the x pieces unused / not in any non-retired slice /
distinct, L fresh or allowed (s190e) and fresh (s190f), N only from retired-ok slices, ≥ 4 scripts, the slice totals, the calib Latin order).
**Breakage test** (`breakage-check.out`): a document of the live s190c put into s190f's list: 3 FAILs, exit 1.
**Verify** (`verify190ef.json`): bad = []; x pieces' tokens, top-k bytes and rows.f32 equal their xcorp batch; c docs equal the calib
golden's first 4,096 tokens and rows; gen sums equal the plan; mix rows / gen equal the parts'.

## Copies, md5, ledger, staged files
- `md5-golden-s190ef.txt`: 36 files (5 per directory, the c / x `.jsonl` and `.docs` sidecars). **Identical on ds8 and box2**
  (`/data/q27/w179/slices/`; box2 free after the copy 205 GB; ds8 567 GB).
- Ledger: 164 rows (`golden-s190e-c` 104: fit 67, search 37; `golden-s190f-c` 60: fit 55, search 5; role select, line v3, date
  2026-11-15; `search/v3/run190/ledger-roles-190ef.json`). 28,105 → 28,269 lines; the old prefix is byte-identical.
  (An earlier draw with fresh-only Latin for s190e was run, ledgered and then removed: superseded by the 4:40 PM rule. Its files are
  in `/data/q27/w179/strata/run1-fresh-only/` on ds8.)
- Staged: `workers/w179/registry-staged.json` (live registry snapshot `32b691db…` + s190e staged, s190f staged confirm) and
  `workers/w179/driver-staged.json` (live driver `2c384964…` with `confirm.pairs` + `["s190e","s190f"]`; nothing else).
  The stage refused nothing (`stage190ef.out`: 37 x xdoc strings also occur as strings in other slices, different pieces, as in
  stage190).
- s190b pending check (`pending190b.json`, `workers/w179/pending190b.py`): 90 tentatives on s190a, 90 confirms on s190b, 0 pending.

## How to rerun (on ds8, CPU, `nice -n 19`; the code tree is `/data/q27/w179/code/search/v3/`)
```
cd /data/q27/w179/code/search/v3/run190
python3 -I draw190ef.py  PLAN.json 1903            # or --supply for the counts only
python3 -I check190ef.py PLAN.json /data/q27/w164/strata/calib-kld-strata.json /data/q27/w164/strata/xsupply186b.jsonl
/data/q27/venv/bin/python3 -I build190ef.py PLAN.json WORK /data/q27/w164/strata/calib-kld-strata.json /data/q27/w164/xview
python3 -I verify190ef.py PLAN.json WORK
python3 -I stage190ef.py LIVE_REG C_KLD C_DIR D_KLD D_DIR OUT_REG LIVE_DRIVER OUT_DRIVER S190B_PENDING
```
Inputs (snapshots on ds8 `/data/q27/w179/in/`): the registry (`registry-live.json`), the ledger (`ledger.jsonl`, 28,105 lines), the driver
(`driver-live.json`), the s190b pending file. The draw reads the pending file at import: a pending confirmation drops s190b's documents.

## Files
- here: `plan190ef.json`, `check190ef.out`, `verify190ef.json`, `supply190ef.json`, `supply190ef.err.txt`, `supply190ef-latin.py`
  (the Latin reuse supply), `supply190ef-xnarrow.py` / `.out` (the x N readings), `draw190ef.err.txt`, `build190ef.out`,
  `stage190ef.out`, `breakage-check.out`, `pending190b.json`, `md5-golden-s190ef.txt`.
- code (`search/v3/run190/`): `draw190ef.py`, `check190ef.py`, `build190ef.py`, `pickx190ef.py`, `verify190ef.py`, `stage190ef.py`,
  `ledger-roles-190ef.json`, `SWITCH-190ef.md`. The cd scripts (`*190cd.py`) and `draw190.py` / `check190.py` etc. are unchanged.
