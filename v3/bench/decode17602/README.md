# #176-02 decode speed table from e2e on the mad kernels (w159)

The knapsack's decode term λ_d needs a decode price per (tensor, option). Rule: speed tables come from e2e runs. This
dir measures them on ds8 with the release build (qc-exl3-v3 120132a, `/data/q27/rad10-e157b`), f16 decode activations
(`QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0`, the user default), bf16 KV, batch 1 through the chat API (w138's `bench.py`).
Pre-registration: `RULES.md` (ed41023). Results, report and the price table: `data/decode17602/`.

## Method

A batch-1 decode step runs its ops in order on one queue: RADIANCE_AQL_STAMP's device timestamps of a served pass (741
dispatches, no overlap) show every op; the k-th exl3i GEMV launch is `menu.trunk_names()[k]` (then the head when it is
exl3i). So one stamped served run of a uniform level gives the device time of every trunk tensor at that level, a head
pack the head plan's, an embedding / a-b pack theirs. End-to-end server ms per decode step is then fitted as
`F_ctx + α_ctx · Σ stamped menu time` on the uniform packs (two unstamped ABBA rounds), and the table's price of an option
is `α_ctx × its stamped time`. Held-out mixes (random per-tensor levels, five heads, T4 / T5, trellis embeddings, with
and without act lines) check the sum against measured e2e (< 2 %).

## Files

| file | what |
|---|---|
| `run17602.sh` | ds8 runner: `run17602.sh CARD LIST...`, one server session a row (TAG PACK [stamp] [ctx=..] [env=..] [wait]); `QUIET=1` for e2e rounds (no pack build runs beside them) |
| `mk17602.sh` | ds8 pack builder: head packs (rad-convert --reuse of the K1.125 pack + an R16 head line), trellis embedding and a/b packs (radmix on the K1 pack), held-out mixes, a rolling head builder with gc |
| `st17602.py` | reduces a stamp CSV (decode passes: per tape record kernel, median µs) |
| `mkheld.py` | the 5 held-out mixes (seed 17602): radmix files + assignment JSON (`held/`) |
| `fit17602.py` | `fit` (prices + uniform fit + held-out check), `annotate` (menu options get `dec_us` / `dec_us_8k`), `predict` (ms / step of an assignment) |
| `fetch.sh` | copies ds8's small results into `data/decode17602/` |
| `bench.py` | w138's client (`v3/bench/decode176/bench.py`), copied unchanged |

## Rerun

```
# on ds8 (/data/q27/w159; tools/ = this dir)
bash tools/run17602.sh 7 recipes/L-uni-s.txt                      # stamped uniform round (ctx 0)
QUIET=1 bash tools/run17602.sh 7 recipes/L-uni-aq.txt recipes/L-uni-bq.txt   # e2e rounds, K descending then ascending
bash tools/mk17602.sh roll 3 u1 u1.125 ... & bash tools/run17602.sh 7 recipes/L-parts.txt   # heads / emb / a-b
QUIET=1 bash tools/run17602.sh 7 recipes/L-held.txt               # held-out mixes (built by mk17602.sh hd / mix)
# on the VM
bash fetch.sh && python3 fit17602.py fit ../../../data/decode17602/res ../../../data/decode17602/decprice17602.json \
  --report ../../../data/decode17602/report.txt
python3 fit17602.py annotate ../../../data/decode17602/decprice17602.json MENU.json MENU-dec.json
```
