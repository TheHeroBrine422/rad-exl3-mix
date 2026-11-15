# data/slices195 — #195: the fresh non-Latin xcorp batch and the 15 slices of RULES-slices195 (w180, 2026-11-15)

**Status: batch goldened; 15 slices built, checked (68 PASS), verified (no mismatch), on ds8; ledger rows added; registry staged. Nothing live.**
Pipeline and the rules as implemented: `search/v3/run195/README.md`.

## Part 1 — the non-Latin batch `xn8k-b195`
- **Supply** (`supply195-summary.json`, `supply195.jsonl`; xcorp v1, 53,587 documents, first ≤ 8,192 tokens of each): 2,911 non-Latin documents; held by an
  earlier corpus: xc8k-b1 186, b2 204, b3 185, the QAD corpus xq40 1,237, its skip list 389 (= b1 + b2). **Free: 1,187 pieces, 1.45M tokens, 1.35M gen rows.**
  The multilingual pieces (fineweb2-hq, wikipedia) are plain text, so every row is a gen row under the board's mask; the 99 wildchat pieces carry 63 % gen rows.
  **No self-generation was needed** (the selfgen branch of the task was not run).
- **Batch** (`xn8k-b195.meta.jsonl`: xcorp doc, source, category, tokens, script, gen per piece): the free pieces with ≥ 128 gen rows, MATHEMATICAL left out:
  **1,164 pieces, 1,365,926 tokens, 1,364,762 golden rows, 1,343,809 gen rows (98.5 %), 11 scripts**; 85 chat pieces, the rest plain text.

  | script | pieces | gen rows | | script | pieces | gen rows |
  |---|---|---|---|---|---|---|
  | ARABIC (arb, fas) | 305 | 227,132 | | HANGUL | 49 | 122,874 |
  | CYRILLIC (rus, uk) | 216 | 263,521 | | HEBREW | 49 | 90,042 |
  | CJK | 207 | 135,624 | | DEVANAGARI | 25 | 91,418 |
  | HIRAGANA | 113 | 80,987 | | BENGALI | 20 | 94,594 |
  | GREEK | 101 | 109,047 | | KATAKANA | 9 | 3,178 |
  | THAI | 70 | 125,392 | | | | |

  The batch shares no xcorp document with xc8k-b1 / b2 / b3, xq40 or any earlier x golden (checked on colo by document id and again by `check195.py`).
- **Golden:** box2 cards 0 + 1, the lead's gold3.sh command (radiance `rad10-perf`, TP2, bf16 KV, `--kld-top 512`: top-512 + tail), 20 min, 0 errors
  (`gold195.log`). Two more chunks were recorded the same way for fresh Latin: **xc8k-b2-c3** (741 pieces, 1,001,696 rows) and **xc8k-b2-c4** (665 pieces,
  994,217 rows), 14 min each on cards 3 + 2. They were cut on 11-05 with b2 but never recorded; all 1,406 documents are in the xq skip list and none is in xq40 / b1 / b3.
- **Gen masks, labels:** `genmask181.py make` per batch (`gm.u8` beside each golden on ds8 `/data/q27/w180/xnew/<batch>/`); labels and used flags in
  `xsupply195.jsonl` (`xsupply186.py`; 2,570 rows for the three batches). The golden's own tokens label all 1,164 pieces N.
- **Copies:** box2 `/data/q27/w180/batch/gold/<batch>` = ds8 `/data/q27/w180/xnew/<batch>` (md5 identical: `md5-batch-ds8.txt`, `md5-batch-box2.txt`).

## Part 2 — the 15 slices (`plan195.json`, md5 11717a08; seed 195)
| slice | role | match | c docs | c L gen (× own c) | V3 decisions of c docs (min–max, mean) | x L pieces | N pieces | L gen (× match) | N gen | N share | scripts | max script | rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s195s1 | search | s186a | 48 | 51,607 (1.03×) | 168–496, 265 | 263 | 43 | 344,181 (2.001×) | 37,333 | 9.8 % | 10 | 10.2 % | 428,859 |
| s195s2 | search | s186b | 50 | 51,840 (1.19×) | 168–573, 281 | 241 | 36 | 303,572 (2.004×) | 33,336 | 9.9 % | 10 | 10.1 % | 380,517 |
| s195s3 | search | s186c | 53 | 50,701 (1.18×) | 169–573, 264 | 271 | 27 | 370,186 (2.001×) | 40,513 | 9.9 % | 10 | 10.2 % | 456,962 |
| s195s4 | search | s186d | 71 | 50,628 (1.08×) | 169–502, 283 | 326 | 42 | 420,911 (2.002×) | 46,056 | 9.9 % | 10 | 10.2 % | 510,056 |
| s195s5 | search | s186e | 45 | 50,707 (1.03×) | 170–573, 290 | 248 | 37 | 361,575 (2.020×) | 39,565 | 9.9 % | 10 | 10.2 % | 441,436 |
| s195c1 | confirm | s186a | 59 | 53,920 (1.08×) | 1–167, 67 | 260 | 42 | 351,641 (2.045×) | 38,694 | 9.9 % | 10 | 10.1 % | 433,073 |
| s195c2 | confirm | s186b | 55 | 51,410 (1.18×) | 5–166, 69 | 201 | 32 | 309,637 (2.044×) | 33,709 | 9.8 % | 10 | 10.2 % | 376,758 |
| s195c3 | confirm | s186c | 55 | 49,967 (1.17×) | 8–164, 65 | 279 | 31 | 370,988 (2.006×) | 40,097 | 9.8 % | 10 | 10.3 % | 457,558 |
| s195c4 | confirm | s186d | 49 | 49,987 (1.07×) | 8–164, 73 | 289 | 40 | 420,978 (2.003×) | 46,209 | 9.9 % | 10 | 10.1 % | 511,580 |
| s195c5 | confirm | s186e | 59 | 54,007 (1.09×) | 8–168, 70 | 241 | 45 | 359,060 (2.006×) | 39,115 | 9.8 % | 10 | 10.2 % | 446,391 |
| s195c6 | confirm | s186a | 53 | 50,991 (1.02×) | 9–154, 71 | 236 | 43 | 344,180 (2.001×) | 37,785 | 9.9 % | 10 | 10.1 % | 422,423 |
| s195c7 | confirm | s186b | 50 | 51,935 (1.20×) | 9–162, 75 | 204 | 46 | 303,115 (2.001×) | 32,978 | 9.8 % | 10 | 10.2 % | 370,165 |
| s195c8 | confirm | s186c | 40 | 50,682 (1.18×) | 9–162, 72 | 280 | 40 | 371,507 (2.008×) | 40,487 | 9.8 % | 10 | 11.8 % | 452,243 |
| s195c9 | confirm | s186d | 44 | 49,963 (1.07×) | 9–165, 78 | 292 | 53 | 421,303 (2.004×) | 46,097 | 9.9 % | 10 | 18.0 % | 508,877 |
| s195c10 | confirm | s186e | 51 | 50,055 (1.01×) | 4–158, 58 | 257 | 45 | 358,302 (2.001×) | 39,387 | 9.9 % | 10 | 20.2 % | 449,800 |

"L gen (× match)" = the slice's Latin gen positions and their ratio to the matching s186 slice's c + x Latin gen (rule: ≥ 1.9). Totals: 6,646,698 rows,
6,002,497 gen positions (L 5,411,136, N 591,361); 782 calib documents (each in one slice), 4,490 xcorp pieces (each in one slice; 602 N, 68 of them cut to a prefix).

- **Rule check, measured on the built mixes** (`verify195.json`): N share 9.75–9.91 % (rule 9–12 %), 10 scripts in every slice (≥ 5), largest script
  10.1–20.2 % of N gen (≤ 1/3), Latin 2.001–2.045 × the matching s186 slice (≥ 1.9).
- **The one place the rule's size is not met part by part: the calib c parts are ~1.0–1.2 × their s186 c part, not 2 ×.** The allowed calib Latin pool is
  782 documents / 768,400 gen; fifteen 2 × c parts need 1,395,630. Each c part has an equal share (~51k gen) and the x part carries the rest of the slice's
  2 × Latin total (the 4:25 PM RULES-cooldown precedent). The x parts are therefore 2.25–2.44 × their s186 x part.
- **Freshness:** every allowed calib document already has V3 decisions (none has 0); the confirm slices hold the least used two thirds (1–168 decisions,
  means 58–78), the search slices the rest (168–573, means 264–290). Confirm xcorp Latin is fresh (2,539 pieces: b1 1,132, b2-c0..c2 218, b2-c3 628, b2-c4 561);
  search xcorp Latin is reused (1,349 pieces: 835 from retired slices — s190b 170, s171c/d 220, s186b/d/e/f 387, s173a/b/e 58 — and 514 flag-only pieces
  of the search27 / s129 era). All N is fresh.
- **N by script over the 15 slices** (gen): ARABIC 58,348, BENGALI 57,731, CJK 58,004, CYRILLIC 67,169, DEVANAGARI 53,761, GREEK 58,350, HANGUL 59,668,
  HEBREW 59,357, HIRAGANA 59,520, THAI 59,453. Categories: 575 multilingual, 26 chat, 1 code piece.
- **Left over:** N batch 553 pieces / 647,554 gen (ARABIC 233, CYRILLIC 144, CJK 112, GREEK 42, HANGUL 8, THAI 7, HIRAGANA 4, HEBREW 3; **BENGALI and DEVANAGARI are spent**);
  fresh xcorp Latin 293 pieces / 372,703 gen; reusable retired Latin 1,189 pieces / 1,493,202 gen.

## Checks
- `check195.out`: **68 PASS, 0 FAIL** (names, pairing, roles; calib documents distinct, in the pool, Latin, outside the never set, the whole allowed pool used,
  confirm ≤ search in V3 decisions; xcorp pieces distinct by (batch, idx) and xdoc, outside the never set, N fresh from the batch, confirm Latin fresh, search
  Latin from retired slices only; caps; gen recount from the masks; per slice: L ≥ 1.9 ×, N share, scripts). A plan with one s190e document put into s195c3 gives 3 FAIL.
- `verify195.json`: `bad = []` over 45 goldens (x: tokens, top-k bytes, rows.f32 identical to the batch per piece, gen = plan, labels = xsupply; c: the calib
  golden's rows, gen and labels; mix: the two parts' documents once, gen = plan).
- **History:** draw 1 (4:55 PM CST) read w179's s190e / s190f goldens while w179 was still rebuilding them, and the registry before the rotation; draw 2 used the
  registry after the rotation (s190c retired, s190e active); draw 3 (final) fixed the retired x pieces' use count for s171a–d (their decisions were read as 0).
  The c parts of draws 2 and 3 are identical. The superseded plans are on ds8 `/data/q27/w180/strata/draw1`, `draw2`; their goldens were deleted.

## Copies, ledger, staging
- **Goldens:** ds8 `/data/q27/w180/slices/golden-s195{s1-5,c1-10}{,-c,-x}` (+ `-c.jsonl`, `-x.jsonl`, `-x.docs`): 270 files, 51 GB; `md5-golden-s195.txt`.
  The box2 copy was made and compared (270 files identical, `copy195.out`), **then deleted on the lead's 5:25 PM CST note** (box2 disk; ds8 only). box2 free 197 GB, ds8 502 GB.
- **Ledger:** 782 rows (role `select`, line v3, 2026-11-15; `search/v3/run195/ledger-roles-195.json`); the old prefix is byte-identical. xcorp pieces have no
  ledger rows (as every earlier x part: the c part's note names them).
- **Staged:** `rad-local/workers/w180/registry-staged.json` = the live registry at 5:25 PM CST (md5 f83c1d69, copy in `registry-live-at-stage.json`) + 15 keys
  (`s195s*` staged, `s195c*` staged with role confirm, each with `"pair"`); every existing key unchanged (`stage195.out`). No driver config was written.

## Files
`supply195-summary.json`, `supply195.jsonl`, `xn8k-b195.meta.jsonl`, `gold195.log`, `prep195.log`, `xsupply195.jsonl`, `draw195.err.txt` (supply and per-slice
lines), `plan195.json`, `check195.out`, `build.log`, `verify195.json`, `copy195.out`, `md5-golden-s195.txt`, `md5-batch-{ds8,box2}.txt`, `stage195.out`.
