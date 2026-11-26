# v3/qad232: goldens for the V3.1 finals (w232, 11-26)

Part A: `xg-fin` = xgap-v1 training documents, WHOLE (<= 131,072 tokens), bf16 goldens (top-512, bf16 KV). Part B: the final-read slices cut from the published calib golden.

| step | script | what |
|---|---|---|
| A1 | `sel232.py XGAP TOK OUT --exclude META...` (colo) | draws non-holdout xgap docs per cell (seeded, round-robin over sources) to ~5.4M scored (gen) rows, cell shares 30/20/15/12/8/10/5, shortfall spread pro rata; whole docs, excludes `xg_id`s of xg-scr1 + the gap judge; writes `xg-fin.c<k>.jsonl` chunks (deepest first, docs > 100k alone in chunk 0), `xg-fin.chunks.json`, `xg-fin.stats.json`. Counts: `data/qad232/`. |
| A2 | `run232.sh` -> `chain232.py` (box2) | TP2 bf16 recording, lane A cards 0,1 (locks held, `only-card0/1` = NONE meanwhile), lane B cards 2,3 only when both locks are free; TP4 fallback at the end for a chunk that does not fit TP2; restartable per chunk. |
| A3 | `merge232.py`, `check232.py`, `report232.py` | chunk goldens -> `QROOT/ext/gold/xg-fin` (+ `ext/genmask/xg-fin.u8` by genmask181, `ext/xg-fin/xg-fin.jsonl`), checks (rows, mask vs meta, tokens vs tokenizer for every doc, corpus hash, top-512 sanity), final numbers. |
| B | `mkv31.sh IDX` (box2), `checkv31.py` | `slices/golden-v31ver` (422 non-trace `verify_v31` docs), `slices/golden-v31tr` (traces 905 / 2675), cut with `eval/golden_pick.py` (whole docs); `checkv31.py` compares every byte with the calib golden. |
| tests | `selftest232.py`, `chaintest232.py`, `fake_radiance232.py` | no GPU (colo): merge + check incl. corruption detection; the whole chain with a fake radiance (TP2 fails above 20k -> TP4 fallback; both lanes). |

Part C (lead 11-26, replaces golden-final750's lost x part): `slices/golden-v31x` = FRESH non-calib verification part from xgap-v1 HOLDOUT docs not in the gap judge, whole docs <= 64k tokens, ~400k scored rows in the gap judge's cell mix by scored rows
(`sel232.py ... --holdout --mix data/qad232/jgap-mix.json --total 400000 --maxtok 65536`, counts `data/qad232/v31x.*`), recorded by the same chain (`run232x.sh`, own root `/data/q27/w232x`, cards 0,1 only), gen mask by genmask181 (inside the slice as gen.u8).
Cards 2,3: lane B is OFF by default (`W232_LANE_B=1` enables it; lead arbitration 11-26, w233 needs 2,3). `after232.sh PID` = the hand-off that restarted the xg-fin chain with lane B off and then ran part C.

Score prefixes (`v3/qadq/qscore.sh`, `qread.sh`): `JV31-*` -> golden-v31ver (MML 101376), `JT31-*` -> golden-v31tr (262144), `JX31-*` -> golden-v31x (MML 67584; max doc 62,104 tokens).
Read each ONCE per final decision.

Rerun (box2): `cd /data/q27/w232; setsid nohup code/v3/qad232/run232.sh < /dev/null > /dev/null 2>&1 &`; log `/data/q27/w232/chain.log`; done = `chain232 done`.
