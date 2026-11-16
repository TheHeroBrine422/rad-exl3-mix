# data/qad198 — QAD soup (#198, RULES198; w186, box2)

Raw results of the #198 readout. Report: `/workspace/rad-local/workers/out/w186.md` (headline); LOG entries 11-16 3:06 AM CST (main) and 3:25 AM CST (follow-up: arm SH, RULES198 amendment 1).
Code: `v3/qad198/` (soup198.py, analyse198.py, supp198.py, run198.sh, p198.sh, env198.sh, README.md).

| path | what |
|---|---|
| `analysis/tables.md` | the readout (RULES198 arms, SH arms of amendment 1, and contrasts): per arm KL on x181g / fe38 / pooled, every contrast with its per-document bootstrap 95 % CI and verdict. A superset: the 21 contrasts of the first readout reproduce exactly (same bootstrap draws) |
| `analysis/analysis.json` | the same numbers as JSON (arm paths, rows sha256 per set, contrasts, bootstrap reps 2,000, seed 0) |
| `analysis/perdoc.json` | per arm and judge set: per-document KL sums and gen-row counts (reproduces every interval) |
| `analysis/supp.md` | supplementary contrasts against the MEAN of each soup's members (not RULES198 contrasts); reproduces the pre-registered S2s-vs-ref interval exactly (checked in the script) |
| `raw/unit/report.json` | soup of one = L0B0T0Q0D0, files: state 720 / 720 bitwise, scale values 6,799,360 / 6,799,360 bitwise (exp on the GPU) |
| `raw/unit/shrink-identical.json` | amendment 1's unit check: a shrink soup of N identical members (D0 × 2 and × 4) equals the plain mean soup bitwise (state 720 / 720, scale values 6,799,360 / 6,799,360) |
| `raw/unit/report-cpuexp.json` | the same check with exp on the CPU: 96.6 % of scale values bitwise, max relative difference 1.2e-7 (kept: it is why the soups were redone) |
| `raw/soups/<SO1,S2,S4,C2,S2s,SH4,SH2s>/soup.json` | soup manifests: members (sha256 of params.pt, steps, rows, seed, mix50, a8), start state, sign-agreement between members, distance of the soup to each member; SH: the shrink ratio m²/(m²+v/N) over the moved parameters (SH4 mean 0.64, median 0.75; SH2s mean 0.53, median 0.56) |
| `raw/soups-cpuexp/<SO1,S2,S4>/soup.json` | the same manifests for the first (CPU exp) soups |
| `raw/served/<REF2,SO1,S2,S4,C2,S2s,A-s1,SH4,SH2s>/` | eval196 result.json, radiance score summaries (score-*.json), census.txt, eval.log |
| `raw/served-cpuexp/<SO1,S2>/` | the CPU-exp served results (side arms: S2cpu, SO1cpu in the table) |
| `raw/runs/A-s1/` | the training run: run.log, plan.json (order sha1 88dccb60bdb0), train.jsonl, mon.jsonl, fixed.jsonl, fixed-eval-docs.json, ckpt mon.json / .done |
| `raw/rows.sha256` | sha256 of every served rows file (per-token KL, float32 Nx4). The rows stay on box2 `/data/q27/w186/served*/` (about 3 MB a set; the per-document sums in `analysis/perdoc.json` are what the intervals use) |
| `raw/p198.log`, `raw/run198.log` | the driver's log (step times, gates, lock waits) |

Checkpoints (soups and A-s1, `params.pt` 23 MB each, scales/ 400 files) stay on box2 under `/data/q27/w186/soups*/` and
`runs/A-s1/ckpt-1e+06/`; `soup.json` records their sha256.

Reproduce (box2, `/data/q27/w186`): `CARD=2 setsid nohup bash ./code/v3/qad198/p198.sh` (every step skips what is done);
the readout: `analyse198.py analysis ...` (arguments in `p198.sh` step 5), then `supp198.py analysis`.
