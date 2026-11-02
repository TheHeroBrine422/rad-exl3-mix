# YAQA / H_out on the Qwen3.5-0.8B proxy (2026-11-02), raw results

Experiment numbers refer to `docs/experiments/INDEX.md` (#33 and its parts); pass rules in `proxy2b/YAQA08.md`.

| file | experiment | content |
|---|---|---|
| `result.txt` | #33-08 | Sketch B dose 1K–8K: per-arm KL / top-1 and paired comparisons (ratio, diff, se, z) |
| `objcheck.txt` | #33-08 | per linear, one-sided vs two-sided objective for base / ldlq / yb1024 |
| `houtstruct.jsonl` | #33-09 | per linear: H_out spectrum, diagonal share, split-half and convergence cosines, H_in vs calib |
| `result-fact.txt` | #33-10 | factorial: Sketch B H_in vs H_out vs diagonal |
| `result-fact2.txt` | #33-11 | H_out at K2, and stacked on base's settings |
| `result-fact3.txt` | #33-12 | mixed / diagonal-only H_out, dose of the full H_out |
| `result-fact4.txt` | #33-13 | diagonal dose 1K / 2K, hout_diag file-path identity check |
| `prectest.jsonl` | #33-14 | split-precision (x2, bf16) vs fp32: per-linear error of the summed diagonal |
| `result-fact5.txt` | #33-15 | split-precision collectors judged by KL |
| `result-fact6.txt` | #33-16 | seed replicates, identity H_out |

Times in the result files are UTC. Golden: g08-search1 (173 search-pool docs, 166k generated rows).
