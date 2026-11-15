# run186w — the #186 stratified slices s186a–f and the t186n tracker (w164, 2026-11-14)

Built from `search/v3/RULES-strata.md` (committed before any data: f0db8aa5). Everything runs on ds8 (CPU, `nice`); the
raw numbers are in `data/strata186/` (plan, checks, build log, built-part shares, supply summary).

## Pipeline (rerun order; ds8 paths)
| step | script | input | output |
|---|---|---|---|
| 1 labels | `search/v3/strata186.py label GOLDEN TOK OUT` | calib golden (4,789 docs) | `calib-kld-strata.json` (labels per calib doc; `data/strata186/calib-labels.tsv`) |
| 2 supply | `xsupply186.py TOK OUT.jsonl XVIEW GOLDEN_X...` | the xcorp goldens, their meta, every earlier x part (the exclusion set) | per-piece labels and `used` flags; summary in `data/strata186/xsupply-summary.json` |
| 3 draw | `draw186.py PLAN.json 186` | labels, gen masks (`genmask181 make` on the b1 golden: `gm-b1.u8`), registry, ledger, t183 c part | `plan186.json` (`data/strata186/plan186.json`) |
| 4 check | `check186.py PLAN ...` | the plan, the labels, the gen masks, the registry; an independent recount | `check186.out` (ALL PASS) |
| 5 build | `build186.py PLAN WORK CALIB_STRATA XVIEW` | the golden's files | `WORK/golden-s186X-{c,x}`, `golden-s186X` (goldmix), `golden-t186n*` |
| 6 shares | (inline, `data/strata186/built-shares.json`) | the built goldens' `gen.u8` | N share of gen positions, scripts, max script share |
| 7 ledger | `eval/pools/ledger_add.py ROLES POOLS LEDGER --date 2026-11-14 KLD...` | the c parts' kld.json | 300 rows (roles in `ledger-roles-186.json`, merged for the call) |
| 8 refs | `refs186.py T183G T183R T186G T186R REFS_T183 TOK OUT` | the rows files of each reference on t183 and t186n | `refs186.jsonl`: L, N183, N (t186n), board per reference |
| 9 refs lane | `refs186.sh` (one reference at a time through `ref186.sh`; `LIST`, `CARDS`, `LOCAL_DIRS`) | containers (local, or `hf download` and delete) and golden-t186n | `{name}-t186n.json.rows` (+ `.json.part` report) |
| 10 CI | `refs186ci.py GOLDEN ROWS_A ROWS_B --name-a A --name-b B` | two references' rows on golden-t186n | paired per-document bootstrap (4,000 replicates, documents resampled together) of the N-stratum ratio B/A and difference |
| 12 tails | `refs186tails.py GOLDEN ROWS_DIR` | the rows of step 9 and golden-t186n's gen mask | `data/strata186/refs186-tails-t186n.tsv`: gen-position mean KLD, p99, top-1 per reference |
| 11 plot | `search/convplot4.py STATE OUT.png ... --track results.jsonl --track-refs refs-t183.jsonl --track-strata refs186.jsonl` | `refs186.jsonl` | the tracking panel; with `--track-strata`, IQR-V4 per band: N (dotted) and board (dash-dot) geo-means of the ratio to V2 (log-linear in bpw); the t183 band lines are full t183 |

`xview/` is a scratch view of the b1 golden with colo's copy of its corpus meta (`/data/work/gold27/xc8k-b1.jsonl.meta.jsonl`
on colo; 3,389 lines, tokens and source identical to the golden line by line). ds8's `xc8k-b1` has no meta.

## Rules as implemented (RULES-strata §3, §7)
- **c parts**: calib documents of the merged fit + search pool, least used first (ledger decision counts + the registry's
  own decisions; ties by a seeded draw), never a document of a ready / active slice (the registry's c parts: 146 docs) or
  of t183's c part (48). Each part: 6–11 N documents (built: 11, 6, 9, 7, 7, 7), one per script
  first (at least six scripts), then N documents balanced over the scripts by gen positions, then L documents until ~45k
  scored positions; a repair step adds documents until the N share of gen positions is in [15.5 %, 19 %] (built:
  17.7–19.0 %), >= 6 scripts, and no script above a third of the part's N gen positions. Parts end at 40–70k positions
  (the repair adds documents, so the parts are larger than 45k). Documents cut at 4,096 tokens (as draw171). **Decision (w164):** the six c parts draw from one
  pool, so documents recur across parts once less used ones are spent (75 repeated uses over 205 documents; the
  least-used-first rule is kept; each part has distinct documents). Without this the N pool (58 free documents) cannot
  give six parts six scripts each.
- **x parts**: 130 pieces each, from xc8k-b1's unused pieces (3,324 unused; 181 N, 9 scripts). N pieces one per script
  first (the scripts with fewer than 20 unused pieces — THAI, HANGUL, GREEK, HEBREW, DEVANAGARI — only once per part, so the
  rest is left for t186n), then balanced by gen positions; L pieces stratified by category (largest remainder over the
  unused L pieces); heavy categories (agentic, tools, long, reasoning) cut at 2,048 scored positions as pickx.
  **Decision (w164):** the x parts use xc8k-b1 (the lead's `goldens/xcorp/*`). That batch was also the H_out / Fisher
  collection of w96 / #127 (all of it except search27b's 65 pieces): training / statistics use, allowed for xcorp; the
  pieces are not disjoint from it.
- **Sizes**: the N share is measured on gen positions of the part as scored (documents cut at 4,096 for c, the cut
  pieces for x). Built shares: c 17.7–19.0 %, x 15.2–16.4 %, max script share of N gen ≤ 0.33 (`data/strata186/built-shares.json`).
- **t186n** (§7): 20 calib N documents (merged pool, not in any c part, not busy, not t183) + 2 xcorp N pieces (b1, not in
  any x part): 22 documents, 32,117 gen positions, 10 scripts; role monitor in the ledger; never decides.
- **Labels**: a calib document's label is its calib label (full document, `strata186.label_docs`); an xcorp piece's label
  is from its own tokens (`label_golden`); both agree with the supply table (checked on the built x parts).
- **Registry**: NOT written to the live registry (the driver rewrites it on every decision); the six entries are staged in
  `/workspace/rad-local/workers/w164/registry-staged-186.json` with `"status": "staged"` (see `search/v3/SWITCH-strata.md`).

## Checks (run on the plan before the build; `data/strata186/check186.out`)
All PASS: x pieces distinct and unused by every earlier x part (xsupply flags: all golden-s17*-x, t183-x, final-x, s129-x,
w129's search27 excl files); c documents in fit + search, not busy, not t183; c and x N shares in [0.15, 0.20] of gen
positions; >= 6 scripts; no script above 1/3 of N gen; t186n >= 6 scripts, ~30k gen positions, every document N.
The built goldens reproduce the plan's numbers (`data/strata186/built-shares.json`).

## Reruns
`python3 -I search/v3/run186w/draw186.py PLAN.json 186` (CPU, seconds) → `check186.py` → `build186.py` (restartable;
a finished golden is skipped). The draw is seeded (186) and reads the registry and the ledger as they are: rerun it only
before a build, and only after the registry has been merged, or the plan changes.

## Reference lane and plot (step 9-11, 11-15, w164)
- **Cards.** The lane takes only the cards in `CARDS` under their `flock` (the lead's rule: ds8 card 7 for workers; box2 cards 2 and 3,
  shared with w163 / w160). The first pass (from 23:50 UTC on 11-14) ran V2 1.50-4.00 on ds8 cards 0, 3, 5 and 6, while the search
  was live, before the lead's card-7 rule; those six rows are in `refs-lane-ds8.log`. After the restart the ds8 lane used card 7 only
  (`lane2` section: V2 4.50 / 5.00), then was stopped.
- **Downloads.** A ds8 `hf download` ran at ~3 MiB/s under the search's load (load average 48-72); after 8 minutes the V2 6.00 file
  was at 0.36 GB and was stopped (its partial deleted). w162's note measured the same slow relay through the VM (1-5 MB/s), while ds8 <->
  box2 over 10 GbE is fast. Box2's direct download runs at 12-15 MiB/s per stream, so the rest runs there in two lanes
  (`lane-box2-v2.log`: V2 6.00 / 7.00; `lane-box2-iq.log`: IQR iq3r / iq5r / iq6r). Containers are deleted after scoring;
  box2 keeps > 100 GB free.
- **Equivalence (the 1e-5 bar, cross-machine).** Same radiance build (md5 `d07114d0`) and golden-t186n on both boxes (md5 of
  `kld.json` and `rows.f32` match). iq2r on t183 scored on box2 (golden copied from ds8, 10 GbE) against the lead's ds8 row file:
  the rows are bit-identical (181,488 rows, 4 columns, max |diff| 0.0), the gen-masked mean 0.067851394415 = the lead's
  refs-t183 value. Raw: `data/strata186/equiv-iq2r-t183.txt`.
- **Plot.** `convplot4.py` with `--track-strata refs186.jsonl` draws the IQR-V4 band geo-means of the N and board ratios (V2
  log-linear in bpw). The t183 band lines stay full t183 (3.8 % of its positions are N); the board uses t183's L-only reads.
  Not drawn: the search's incumbents on t186n (`--track-n`, 62 incumbents, each needs a t186n read; the board line for them also
  needs an L-only t183 row per incumbent). Not run: a GPU decision for the lead.
- **Paired intervals** (`refs186ci.py`; 21 of the 22 t186n documents have gen positions: doc 9, calib orig 966, has none under the mask and adds nothing, a defect of the t186n build, not fixed here): IQR-V4 / V2 at the nearest bpw, N stratum,
  95 % per-document bootstrap. Results in `data/strata186/refs186ci.jsonl`; none differs from 1 at 95 %.
- **Reruns.** `CARDS=7 ./refs186.sh` on ds8 (default list), or on box2 `LIST=... CARDS="3 2" LOCAL_DIRS=... ./refs186.sh`
  (commands in the script header). `refs186.py` then needs the t183 rows (`/data/q27/srch183/refs`) and the bf16 tokenizer.
