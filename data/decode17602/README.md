# #176-02 decode speed table (w159, 2026-11-14) — data

Scripts and method: `v3/bench/decode17602/` (README there; rules `RULES.md`, ed41023 + amendments). All rows: ds8,
radiance qc-exl3-v3 120132a (`/data/q27/rad10-e157b`), f16 decode activations (`QWEN35_EXL3_AQ_XB=0 EXL3I_MAD_GEMV=0`),
bf16 KV, batch 1, greedy, chat API. Card HIP 6 (R9700S, 210 W cap) for everything in `res/`; `c7/` = HIP 7 before the
10:08 AM CST reboot (pilot, first stamped round, part of round a; cross-card check only, not fitted).

| file | what |
|---|---|
| `decprice17602.json` | **the price table** (`fit17602.py fit`): `ctx0` / `ctx8k` → `F_ms`, `alpha`, `trunk[type][level]`, `head[plan]`, `emb[option]`, `ab[bf16/T4/T5]` in µs a decode step; `raw` = stamped device µs; `fit` = uniform fit; `heldout` = the check |
| `menu173m-v1b-dec.json.gz` | w158's menu `menu173m-v1b.json` (ds8 `/data/q27/w158/menus/`) with `dec_us` / `dec_us_8k` on every option (`fit17602.py annotate`); 31 trellis embedding options interpolated (`dec_interp`) |
| `report.txt` | the fit's report (min of runs): uniform rows K by K, components, held-out |
| `report-stat-med.txt`, `report-stat-gap.txt` | the same with the other two e2e statistics (RULES amendment 1: all reported) |
| `res/u<K>-{a,b,c}.jsonl` | uniform e2e rounds (a: K descending, b: ascending, c: amendment 2 reruns of the 9 disturbed packs) |
| `res/u<K>-s.jsonl`, `res/st-u<K>-s.json` | stamped round (ctx 0) and its reduced stamps (st17602.py) |
| `res/h-<plan>*`, `res/e-<opt>*`, `res/ab-T*` | head / trellis embedding / a-b rows (stamped, ctx 0) |
| `res/m-h<n>-{a,b,s}*`, `res/u2.5-c*`, `res/u4-c*` | held-out mixes (a, b, stamped s) and the drift reference rows |
| `recipes/` | runner lists, held-out radmix files + assignments (`m-h<n>.assign.json`) |
| `logs/run.log`, `logs/mk.log` | runner / builder logs (HIP 7 lines = pre-reboot / tests) |

Headline (ctx 0 / 8k): step = F + α·Σ stamped menu time with F 3.557 / 4.597 ms, α 0.974 / 0.979; uniform residual rms
0.27 / 0.26 % (42 levels); held-out errors (pred vs measured) h1 −1.35 / −1.37 %, h2 +0.13 / −0.28 %, h3 −0.40 /
−0.54 %, h4 −0.82 / −1.17 %, h5 −0.54 / −0.92 % → PASS (< 2 %).
