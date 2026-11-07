# reprice · knapsack re-pricing of the V3 ladder with the 2026-11-06 facts (w58, 2026-11-07)

Report: `../reprice.md`. Data: `data/levels/reprice/out/` (one JSON line per solve), tables `data/levels/reprice/report.out`.
**Runtime cost:** none (pricing only; no format, kernel or quantizer change).

## Files

- `knap5.py`: `v3/nf/knap4.py` plus the switches in its docstring (`KN_V3R`, `KN_TP`, `KN_SWJ`, `KN_CD`, `KN_HEAD16`,
  `KN_G8/G4`, `KN_NOA4`, `KN_PICKS`, `KN_R8_MID`, `KN_PHI_INTERP`) and a sliced DP update (`KN_FASTDP`, default on, same
  arithmetic). `knap4-to-knap5.patch` = the diff. knap4 itself is untouched.
  - Checks: knap5 with no new switch = knap4 byte for byte (V3 config, 2.5 bpw, lambda 0 and 3e-6), with and without
    the sliced DP; the sliced DP gives the same V3R output line as the full one (2.5 bpw, both lambdas).
- `twopass-m2048.jsonl`: prefill times per shape, f16 (T2) and A8 (two-pass, a8g128), µs per layer at M 2048, K4,
  from the A8 worktree's TWOPASS.md (`bench_twopass --m 2048`, median of 20). Layer mean 14,474 / 9,444 µs (×1.53).
- `mkjobs.py OUTDIR`: the arm list (one job = arm × size × lambda) → `jobs-cpu1.txt` (colo), `jobs-cpu2.txt` (Thor).
- `run5.py JOBS MAXPAR`: restartable queue runner (skips finished jobs, adopts solves still running from a previous
  runner, `logs/MAXPAR` re-read, `logs/STOP` pauses). Prints `FAIL ...` and `W58 QUEUE DONE`.
- `report.py DATADIR`: all tables of `reprice.md`.

## Run (as done 2026-11-07 00:29 UTC)

On each box, `/data/work/w58/tree` holds the repo paths knap4 reads (`v3/nf`, `v3/e1b`, `v3/e3/data`, `v3/bench/data`,
`data/v3`) plus `knap5.py` in `tree/v3/nf/` and `twopass-m2048.jsonl` in `tree/`; then
`cd /data/work/w58 && setsid nohup python3 run5.py jobs.txt N > logs/queue.log 2>&1 < /dev/null &`
(Thor N = 40, colo N = 20…28; single-threaded solves at nice 10, ~0.1–0.3 GB each; 1.5 min a solve at 2.5 bpw, 15–30 min
at 6.5–7.0 under load). History: started 00:29 UTC with the full λ grid on every arm; at 01:22 UTC the sensitivity
arms were trimmed to λ ≥ 3e-7 and h16 to accurate only (mkjobs.py as committed reproduces the trimmed lists; the few
already-finished low-λ solves of those arms are in out/ too); phifloor 2.0–4.5 and nolow added at 02:14, dupm1 /
noi8k7 / noi8hi (`jobs-cpu2b.txt`, Thor `jobs2.txt`) at 03:10. Last solve 03:22 UTC. No failures. Collect:
`tar` the two `out/*.jsonl` sets into `data/levels/reprice/out/`, then `python3 report.py data/levels/reprice/out`.
