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
