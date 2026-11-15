# run190 — the s190a / s190b pair (#190, RULES-cooldown amendment 11-14 11:40 PM CST; w169)

One 2×-Latin pair of slices matched to s186a / s186b, built with w164's pipeline (run186w) and checked by check190.py.
Results and numbers: `data/strata190/README.md`. Switch: `SWITCH-190.md`.

| step | script | input | output |
|---|---|---|---|
| 1 supply | `supply190.py OUT.json` | the calib labels, the registry and ledger snapshots, the golden exclusion set | counts only (`data/strata190/supply190.json`) |
| 2 draw | `draw190.py PLAN.json 190` | the same, plus xsupply186b and the b1 gen mask (`gm-b1.u8`) | `plan190.json` (indices only) |
| 3 check | `check190.py PLAN.json CALIB_STRATA XSUPPLY GM_B1.u8` | the plan and the same inputs | `check190.out` (ALL PASS) |
| 4 build | `build190.py PLAN.json WORK CALIB_STRATA XVIEW` | the calib golden, the b1 golden (`XVIEW`), the tokenizer | `WORK/golden-s190{a,b}{,-c,-x}` |
| 5 verify | `verify190.py PLAN.json WORK` | the built goldens | `verify190.json` (bad = []) |
| 6 copy | rsync ds8 → box2 (same path), `md5sum` of every file on both | the built goldens | `data/strata190/md5-golden-s190.txt` (identical) |
| 7 ledger | `eval/pools/ledger_add.py ledger-roles-190.json eval/pools/pools-v3.json eval/pools/ledger.jsonl --date 2026-11-14 C_KLD_DIR_A C_KLD_DIR_B` | the two c parts' kld.json | 207 rows (done) |
| 8 stage | `stage190.py LIVE_REG A_KLD A_DIR B_KLD B_DIR OUT_REG LIVE_DRIVER OUT_DRIVER` | the live registry and driver, the mix kld.json files | `registry-staged-190.json`, `driver-190.json` (in `workers/w169/`) |

## Where it runs
- ds8, CPU only (`nice -n 19`). The code tree there mirrors this folder: `search/v3/{strata186.py, slices.py}`, `search/v3/run190/`
  and `search/v3/run186w/pickx186.py` (the scripts resolve these relative to their own folder, so keep that layout).
- Use `/data/q27/venv/bin/python3 -I` for steps 4 and 5 (transformers, numpy); the others run with the system python3 -I.
- Inputs: `/data/q27/w164/strata/` (calib labels, pools, xsupply186b, gm-b1, plan186 for the matching sizes),
  `/data/q27/w164/xview/xc8k-b1` (the b1 golden with colo's meta), `/data/q27/w169/in/` (snapshots of the registry and ledger:
  refresh them from the live files before a draw; the ledger is `eval/pools/ledger.jsonl`).

## Rules as implemented (RULES-cooldown amendment)
- c: the free calib N documents (merged fit + search pool, outside `exclusion()`: s171–s186 goldens' `orig`, t183, t186n, final750
  c1/c3, the registry's ready / active / staged / confirm / reselect / tracking docs) are split between the two parts; a script with two
  or more documents goes to both parts; the rest to the part with less N gen. L documents: least used first (ledger decisions + the
  registry's own), each part until its L gen reaches 2× the matching s186 part's L gen (cut at 4,096). No document in both parts.
- x: the unused non-Latin xc8k-b1 pieces (outside `exclusion()`: xsupply186's used flags, every earlier x golden's `xdoc`) split the
  same way; L pieces stratified by category, grown until their L gen reaches 2× the matching s186 part's (heavy categories cut to
  2,048 scored positions, as pickx). No piece in both parts.
- No monitor, no minor-script reserve (RESERVE_MINOR = False: not needed for ≥ 4 scripts per part). The N share is not aimed at a band.
- The first run (six pairs, `attempt1/` in `data/strata190/`) is superseded. Its code is kept here: `draw190-6pair-attempt1.py`
  (standalone, not used) and `check190-6pair-attempt1.py` (imports the pair draw190, so it does not run as it is).

## Files here
`draw190.py` (the pair draw), `check190.py` (the pair checks), `build190.py` (w164's build without t186n), `verify190.py`
(built-golden checks), `stage190.py` (staged registry and candidate config), `supply190.py` (supply count; unchanged from the first
run), `ledger-roles-190.json` (roles for ledger_add), `SWITCH-190.md`, `draw190-6pair-attempt1.py`, `check190-6pair-attempt1.py`.
