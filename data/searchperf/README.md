# data/searchperf -- #161 search performance pass (w123, 11-12)

Code: `search/perf/` (README there: results and the recommended protocol). LOG: #161.

| path | what |
|---|---|
| `timing.json` | cost model (`timing.py` over `box2/scores.jsonl` + `box2/radmix.jsonl`): seconds per prefilled token, fixed seconds per radiance run, radmix seconds |
| `box2/scores.jsonl` | tscore.sh phase times: V2 3.00 base + 13 V2 3.50 group moves (radmix over 3.00) on golden-search27, box2 cards 2 / 3 |
| `box2/radmix.jsonl` | radmix wall / bytes per move (ext4: full copies) |
| `box2/settings.jsonl` | scoring settings on h2 (half of golden-search27): time and rows IDENTICAL / DIFFER vs the default run |
| `box2/packs.jsonl` | rad-convert packs: the k3 ladder level, stock (2,083 s) vs `--no-error` (140 s), cmp IDENTICAL, under conv.lock + ram.lock shared (`search/perf/box2/packtest4.sh`); radmix max RSS; the two `rc 1` lines of 11-12 ~19:10 UTC are a synthetic-level attempt (setup bug: no config.json), void |
| `box2/pack4-{stock,noerr}.mem` | 10 s samples: epoch, VmRSS, RssAnon, RssFile (kB), host MemAvailable (kB) |
| `box2/pack4-summary.txt` | the two packs' rad-convert summary lines |
| `decbench.jsonl` | rad-convert's error-report decode of one K3 gate_up-sized exl3i weight, 1 vs 4 threads (colo) |
| `se/pairs.jsonl`, `se/cross.jsonl`, `se/scale.json` | `slicese.py` over the stored rows: per pair x part SE (bootstrap, delta), caps, cap20, cross-slice z, SE vs docs |
| `se/rows.index.json` | which stored reports went into `rows.npz` (gpubox w90-w114 / swap27 / a4cost / ... and w123b2); the npz itself (112 MB) stays on colo `/data/work/w13/w123/rows.npz` -- rebuild with `rowpack.py` from the listed rows files |
| `se/x2-x6.out`, `se/x7.out` | `search/perf/adhoc/x*.py` outputs: cross-slice calibration, cap efficiency, x vs c, noise vs effect size, the move runs, heavy documents |
| `protocol/proto6a.out` | `protocol.py ... runs_per_eval=2` (today's costs: two runs per slice, serial radmix), sections 1-3, 5 |
| `protocol/proto6b.out` | `protocol.py ... runs_per_eval=1 mix_s=0` (goldcat + overlapped mix), sections 1-5 incl. the real-document check |
| `protocol/{noise,frac,resample}.json`, `decide-{before,after}.json` | the tables behind them |
