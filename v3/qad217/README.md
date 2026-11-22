# v3/qad217 — does a larger Adam eps fix the scale-only QAD noise? (#217, w217)

Rules: `RULES217.md` (committed before any step-2 run). Report: `rad-local/workers/out/w217.md`. Raw results: `../../data/qad217/`. Runtime: none.

Scale-only QAD (trainable suh / svh log multipliers, frozen codes; w214's recipe) on the released EXL3-V3 picks at 2.50 / 3.00 / 4.00, with Adam's `eps` raised.
Step 1: a held-out judge slice (j217) and the scale-gradient noise at the unhealed releases. Step 2: 18 arms (size × lr × eps), budget 0.5 M rows with a 0.25 M branch,
served reads on j217. box2 `/data/q27/w217` (inputs from w214's `/data/q27/w214`: `lv/`, `gold/`, `genmask/`, `rel/`, `chunks214.txt`, `pools214.json`).

| file | what |
|---|---|
| `../qad200/qad200.py` | the trainer (w214's), + `--eps` (Adam eps), `--compile-ew` (w215), `--gradnoise N` (gradients of the first N plan steps at the start state, no optimizer step), w215's strict / deterministic `--compile-scales` |
| `mkjudge217.py` | the judge slice j217 (qad-pool documents in no w214 / w217 training set, not read by the release's fits); cut with `eval/golden_pick.py` |
| `gradnoise217.py` | readout of `--gradnoise`: median |mean gradient|, median across-batch std, their ratio, per scale class |
| `env217.sh`, `run217.sh`, `q217.sh`, `card217.sh`, `mkqueue217.py` | box2 runners: one training job a card (retrying), a card's sequential queue (train + read lines), the 18-arm queue builder |
| `score217.sh`, `serve217.sh`, `reader217.sh` | judge: plain radiance `--kld-ref` on j217; swap (`../qad214/swap214.py`) + score of one checkpoint; a reader queue |
| `table217.py` | per-size tables: ratio vs the unhealed release at 0.25 / 0.5 M and the step, paired per document (`../qad214/cmp214.py`'s arithmetic) |

## Rerun (box2 `/data/q27/w217`; code = this dir + `v3/qad{,181,191,200,214}` shipped to `code/`, `code/pipeline` from w214)
```
python code/v3/qad217/mkjudge217.py $CALIB calib.jsonl pools-v3.json ledger.jsonl $W14/pools214.json 150000 8192 217 > judge217.args
python eval/golden_pick.py $CALIB calib.jsonl slices/golden-j217 slices/j217.jsonl $(cat judge217.args)
code/v3/qad217/q217.sh gn-2.50 0 2.50 5e5 --gradnoise 40            # gradient noise (3.00: card 1, 4.00: card 2)
python code/v3/qad217/gradnoise217.py runs/gn-2.50 40
code/v3/qad217/score217.sh ctl-2.50 $W14/rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad 3         # the unhealed controls
python code/v3/qad217/mkqueue217.py eps217.json .; code/v3/qad217/card217.sh 0 q217-card0.txt # the 18 arms (cards 0-3)
python code/v3/qad217/table217.py scores slices/golden-j217 eps217.json --json table217.json
```
