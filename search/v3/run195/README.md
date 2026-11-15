# run195 — the fresh non-Latin batch and the 15-slice set of RULES-slices195 (#195, w180, 2026-11-15)

Rule: `search/v3/RULES-slices195.md` (committed 4ead2983, before any of this). Numbers and raw files: `data/slices195/README.md`.
Nothing here touches the live driver, registry or config; the slices are staged in `rad-local/workers/w180/registry-staged.json`.

## Pipeline (rerun order)
| step | where | script | output |
|---|---|---|---|
| 1 supply | colo, CPU | `supply195.py XCORP_META TOK OUT.jsonl b1=… b2=… b3=… xq40=… xqskip=…` | every non-Latin xcorp v1 document (first ≤ 8,192 tokens): script, gen rows, which earlier corpus holds it |
| 2 batch | colo, CPU | `mkbatch195.py XCORP_META TOK SUPPLY OUT.jsonl` | `xn8k-b195.jsonl` (+ `.meta.jsonl`): the free pieces with ≥ 128 gen rows, a radiance `--kld-corpus` |
| 3 goldens | box2, 2 cards a run | `gold195.sh A B` (xn8k-b195), `gold195b.sh A B CHUNK…` (xc8k-b2-c3 / c4) | bf16 goldens, the lead's gold3.sh command (TP2, bf16 KV, top-512 + tail); card and ram locks held, non-blocking polls |
| 4 prep | ds8, CPU | `prep195.sh BATCH…` | goldens pulled to ds8, gen masks (`genmask181.py make`), `xsupply195.jsonl` (labels by `strata186`, used flags), the xview |
| 5 draw | ds8, CPU | `draw195.py PLAN.json 195` (`--supply`: counts only) | `plan195.json` |
| 6 check | ds8, CPU | `check195.py PLAN.json` | `check195.out` (68 PASS; an independent recount) |
| 7 build | ds8, CPU | `build195.py PLAN WORK CALIB_STRATA XVIEW` (absolute paths) | `golden-s195{s1-5,c1-10}{,-c,-x}` |
| 8 verify | ds8, CPU | `verify195.py PLAN WORK` | `verify195.json` (bad = []) |
| 9 copy | ds8 → box2 | `copy195.sh` | the same path on box2; md5 lists of both sides compared (270 files identical). **Then removed from box2** on the lead's 5:25 PM CST note (box2 disk; box2 is not in the search): the slices live on ds8 only. Do not rerun this step unless box2 joins the search |
| 10 ledger | VM | `eval/pools/ledger_add.py search/v3/run195/ledger-roles-195.json eval/pools/pools-v3.json eval/pools/ledger.jsonl --date 2026-11-15 C_KLD_DIR…` | 782 rows (the c parts; x parts read no calib) |
| 11 stage | VM | `stage195.py LIVE_REG KLD_DIR /data/q27/w180/slices OUT_REG` | `registry-staged.json` |

ds8 layout: `/data/q27/w180/{code,in,xnew,xview,strata,slices}`; `in/` holds the registry and ledger snapshots the draw read
(refresh both before a redraw). box2: `/data/q27/w180/batch` (the three batch corpora and their goldens, 13 GB; md5-identical copies are on ds8 in `xnew/`). colo: `/data/work/w180/`.

## Rules as implemented (RULES-slices195 §2; decisions marked **(w180)**)
- **Names, pairing:** search `s195s1–5`, confirm `s195c1–10`; search `s195s((k−1) mod 5 + 1)` pairs with confirm `s195ck`. Slice `s195s<i>` and its two
  confirm slices match `s186{a..e}[i]` **(w180)**: a slice's Latin target is 2 × (that s186 slice's c + x Latin gen).
- **Never:** t183, t186n, golden-final750 / final-x / final-c3, s173f (staged), s190e / s190f (live), s190d (still a confirm slice at the draw), verify / public
  (outside the pool). s190c was retired by the time of the final draw (registry snapshot md5 3865eba5, ~5:15 PM CST) and counts as a retired search slice.
- **Calib Latin:** the allowed pool is 782 Latin documents, 768,400 gen positions at the 4,096 cut. Fifteen 2 × c parts need 1,395,630 (the ten confirm
  parts alone 930,420), so 2 × c parts cannot be built. **(w180)**, following the 4:25 PM RULES-cooldown amendment for s190e/f ("fill the shortfall with
  more xcorp Latin in the x parts, same total L gen per slice; never shrink a slice"): every c part takes an equal share (~51k gen, 1.0–1.2 × its s186 c part);
  the least used two thirds of the pool (by V3 decision count: ledger rows of a v3* line + the registry's doc_decisions) go to the ten confirm slices,
  the more used third to the five search slices; no document is in two slices. The x part carries the rest of the 2 × Latin total.
- **xcorp Latin:** confirm slices take fresh pieces only (no used flag, in no golden): xc8k-b1, b2-c0..c2 leftovers and the two chunks goldened for this
  (xc8k-b2-c3 / c4, never recorded before; in the xq skip list, so not QAD training data either). Search slices take pieces of retired slices
  (s171a–d, s173a–e, s186a–f, s190a–c) and of the older search27 / s129 x parts, least used first (the holders' decisions; s171 from the ledger;
  the flag-only pieces count 10), the least used prefix stratified by category. Heavy categories cut at 2,048 scored positions, as pickx.
- **N:** only the fresh batch xn8k-b195, each piece in one slice. Target = Latin gen / 9; equal quotas over ten scripts; whole pieces in a seeded order
  while they fit, then one piece cut to the quota (a prefix; rows stay exact). BENGALI (20 pieces) and DEVANAGARI (25) are rationed
  (pieces_left // slices_left a slice); where they fall short in s195c8–c10 the script with the most supply (CYRILLIC) fills in. KATAKANA (9 short pieces) is unused.
- **Gen masks / labels:** every x part gets the board mask by `genmask181.py make` on the built part and labels from its own tokens (`strata186.label_golden`);
  c parts inherit the calib golden's gen rows and calib labels (as build190).

## Pairing note (for the lead)
- `confirm.pairs` to add when the set goes live: `[s195s1,s195c1] … [s195s5,s195c5]` for the first cycle, `[s195s1,s195c6] … [s195s5,s195c10]` for the second.
  A search slice has **two** confirm partners, one per cycle; the staged registry entries carry `"pair"` (search: both partners in order; confirm: its search slice).
  The driver's pair list is static, so at the start of the second cycle the first-cycle pair of that search slice must be dropped (or the confirm slice retired)
  before `s195s<i>` is active again; I did not check how `driver.py` picks among two pairs with the same search slice.
- A confirm slice is used for one rotation period, then retired (RULES-slices195 §2); after `s195c10` the set is spent on the confirm side. A third cycle needs
  new confirm slices: 553 N pieces (648k gen, but no BENGALI / DEVANAGARI and few HEBREW / HIRAGANA / THAI / HANGUL pieces) and 293 fresh Latin pieces (373k gen) are left.
- The x doc-id strings (`x:<source>:<orig>`) are not unique across batches (as in s190a–f): 812 strings of the new slices also occur in other registry slices
  for different pieces; identity was checked by xdoc. The registry's `doc_decisions` for those strings are shared counts.
- The live registry changes on every decision: `registry-staged.json` is a copy of the live file at staging (md5 in `data/slices195/README.md`) plus the 15 keys;
  re-run `stage195.py` (seconds) before merging, or merge only the 15 `s195*` keys.
