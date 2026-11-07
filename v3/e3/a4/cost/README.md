# a4/cost · #46-03 steps 1, 3, 4 on the 27B (w72, 2026-11-07)

Rules: `../README.md` (steps 1–4) and its **amendment 1** (K1, the real kernel instead of the E1b hook, additivity and
K-interaction tests, the knap5 input), written before any data.
**Runtime cost:** A4 is a speed option (prefill `gemm_us`); this folder only prices it (no format or kernel change).

## Pipeline

| where | file | what |
|---|---|---|
| gpubox | `build.sh` | plugin home `/data/a4cost/home`: radiance 1.17.0 (`/data/install-117`) + exl3i / qwen35_exl3 / exl3quant / exl3r4d / exl3dense from radiance-hero `qc-exl3-v3` 42960d5 (A4 rows, per-weight act modes, the mul1i4 importer), host tests |
| gpubox | `q.sh` | the card queue (`/data/a4cost/tools/q.sh`, log `/data/a4cost/q.log`): GPU smoke (test_a4, exl3i_gpu_test), lane Q quantizes i4 K2 → K3 → K1 (ourq `--codebook i4`, ladder settings, 16-layer chunks; every card step = one ~10–15-min hold, lock order: quants and packs `conv.lock` (a 27B quant beside any rad-convert runs gpubox out of RAM) → `turn.lock` → `card0.lock` → `ram.lock` (exclusive for the pack, else shared); scores `turn.lock` → `card0.lock` → `ram.lock` shared; scores in batches of 4, so the queue takes about half of card 0 beside q2), lane S packs (`exl3i codebook=mul1i4`), mixes into V2 3.00 with every trunk weight `a8g128` (census: 336 exl3i mul1i4 a8g128, every other tensor = V2's bytes), scores the baseline and one arm per family × band at `a4g64`, the combined gate_up + down arm, the K2 f16 control and V2 3.00 itself, then the same arms at `a4g128` |
| gpubox | `cmp.py` | one paired readout (gen-masked mean, dKLD, per-document bootstrap SE = swap27's paired.py, ratio, p99, top-1) |
| gpubox | `analyze.py` | `table.jsonl` + tables + additivity + K-interaction + the `A4COST VERDICT` line (after the g64 phase and at the end) |
| colo | `mkcost.py` | knap5a's `KN_A4COST` file: E1c a8g128 cost + the cell's dKLD (floored at 0.5 SE, scaled to the board by V2 3.00's KLD) shared over its tensors by E1c's A4 rel_out |
| colo | `knap5a.py` (`knap5a.patch` vs `v3/levels/reprice/knap5.py`) | knap5 + `KN_A4COST` (measured per-tensor A4 costs, log-linear in K) + `KN_A4X128` (g128's speed ratio, w34 ×1.51) |
| colo | `mkjobs.py`, `run6.py` (= run5.py running knap5a, nice 19) | fast variant at 2.0 / 2.5 / 3.0 (and −0.03 / −0.06 bpw for the bytes test), 14 λ: arms noa4 (A8 only), a4e1c (E1c proxy A4 costs), a4m-&lt;ver&gt; (measured) |
| colo | `step4.py` | the README's step-4 rule: time at equal KLD vs noa4's fast pick, bytes test; `STEP4 VERDICT` |
| VM | `knapwait.sh` | waits for the gpubox verdicts (g64 phase, final), runs mkcost / knap / step4 on colo, copies results to `data/` |

## Run

```
# gpubox (once): bash /data/a4cost/tools/build.sh
setsid nohup bash /data/a4cost/tools/q.sh > /data/a4cost/q.out 2>&1 < /dev/null &      # gpubox; restart = same line
setsid nohup bash v3/e3/a4/cost/knapwait.sh > /dev/null 2>&1 < /dev/null &             # VM, from the repo root
```
`touch /data/a4cost/STOP` stops the gpubox queue between steps. Scores: `/data/a4cost/scores/`, paired lines
`/data/a4cost/results.jsonl`, logs `/data/a4cost/logs/`.
