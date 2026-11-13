# #178-02 results — T4 a/b served (w147, 2026-11-13)

Code and how to run: `search/v3/ab17802/README.md`. Container: the 2.5 incumbent `v3s171-2.5-block-00626`, A8 trunk
(`act a8g128` on all 336 trunk weights; w142's arm B form). B = plain bf16 a / b; T4 = the same with the 48 fused
`ssm_ab` (w142's K4 trellis). box2, one card, rad10-perf install + this branch's plugins; slice golden-s171d (54 calib +
130 xcorp docs, 210,258 positions).

## The container

| | B | T4 |
|---|---|---|
| weights | 787 | 739 (48 added, 96 dropped) |
| bytes | 10,509,639,680 | 10,478,772,224 (**−30,867,456 = −0.00918 bpw**) |
| static VRAM as served | 9.77 GiB | 9.74 GiB |
| census (full byte check vs donors + base, `--expect-act a8g128 --expect-codebook mul1i8 --max-version 3`) | PASS | PASS |

- The donor's 48 ssm_ab planes are **byte-equal** to w142's trellis source (`ds8/abcheck-donor.txt`).
- Decoded by the oracle and rounded to bf16, the T4 weights equal w142's simulated values in 93.0 % of entries. The rest
  differ by one bf16 ulp (max |d| 2⁻⁹, rel. Frobenius 1.5e-3). w142's path rounded through fp16 (exllamav3's decode)
  before bf16.
- Negative census runs fail as intended: the T4 container checked with B's mixfile (ssm_ab "not named"), and B checked
  with T4's mixfile ("dropped, still in the container").
- radmix with the add / drop patch writes a byte-identical B from B's mixfile.

## KLD (`box2/scores/cmp147.out`; mean KL over the golden's generated positions)

| row | what | KLD | c part | x part |
|---|---|---|---|---|
| B132 | w132's score of B (ds8, rad10) | 0.204430 | 0.054100 | 0.238458 |
| B | B, this build (box2) | 0.204430 | 0.054100 | 0.238458 |
| T4sim | w142's T4 arm (bf16 splice, ds8) | 0.205950 | 0.054076 | 0.240327 |
| T4s | the same splice built here (box2) | 0.205950 | 0.054076 | 0.240327 |
| T4o | B + the oracle's bf16 decode of the served records | 0.205694 | 0.054114 | 0.240005 |
| **T4** | **served: exl3i ssm_ab, A8 (AQ plane 3)** | **0.205516** | 0.054023 | 0.239808 |
| T4aq0 | T4 with `QWEN35_EXL3_AQ=0` (ab quantises h itself) | 0.205516 | 0.054023 | 0.239808 |

Identities (byte-identical rows):
- B = B132: the plain a / b path is unchanged and the hosts agree.
- T4s = T4sim: the splice reproduces across hosts.
- T4aq0 = T4: the fused weight reads its own plane of the shared site, so the wiring is right.

Differences (d = pmean over parts of log ΣKL ratio, 95 % paired document bootstrap stratified by part):

| pair | KLD diff | mean \|row diff\| | d [95 % CI] | c | x |
|---|---|---|---|---|---|
| T4 vs T4sim | −4.3e-4 | 0.0189 | −0.16 % [−0.42, +0.11] | −0.10 % | −0.22 % |
| T4o vs T4sim (1-ulp decode rounding only) | −2.6e-4 | 0.0190 | −0.03 % [−0.25, +0.26] | +0.07 % | −0.13 % |
| T4 vs T4o (A8 activation of a / b) | −1.8e-4 | 0.0201 | −0.13 % [−0.53, +0.18] | −0.17 % | −0.08 % |
| T4 vs B (the served change) | +1.09e-3 | 0.0209 | +0.21 % [−0.21, +0.50] | −0.14 % | +0.57 % |
| T4sim vs B132 (w142's arm) | +1.52e-3 | 0.0212 | +0.37 % [+0.03, +0.61] | −0.04 % | +0.78 % |

**The ≤ ~1e-5 KLD target vs w142's arm is not met (−4.3e-4), and no faithful served form could meet it on this slice.**
The served weight's own decode rounding (T4o) moves the mean by 2.6e-4: those are 1-ulp differences in 7 % of the a / b
values. Any a / b perturbation moves this slice's rows by ~0.02 KL a position on average, the same scale as T4 vs B. The
pieces that can be exact are exact: planes byte-equal, wiring byte-identical, host and B byte-identical. The served T4
is no worse than the simulated arm: −0.16 % vs it, CI over 0.

## Time (`box2/perf/perfsum147.out`; 2 rounds, medians; profile shares are with extra synchronisations)

| | B | T4 |
|---|---|---|
| 8k prefill-only (pq.py, s) | 3.414 / 3.400 | 3.398 / 3.403 |
| TTFT 8k (ms) | 3438 / 3427 | 3437 / 3436 |
| decode step ctx 0 (ms) | 32.74 / 32.57 | 32.75 / 32.72 |
| decode step ctx 8k (ms) | 33.56 / 33.59 | 33.80 / 33.74 |

- **Prefill** (profile, 2 × 8k):
  - a / b: 960 `gemm_nt` N 48, 38.9–41.3 ms → 480 `exl3i_dense` N 128, 26.7 ms (−12…−15 ms).
  - The GDN norms' AQ sites: 2 → 3 planes, ~70 → ~87 ms (+15…+17 ms).
  - Net ≈ 0, as e2e shows.
- **Decode** (`box2/perf/pdec-*.optable`, ctx 0, 256 steps, profiled):
  - bf16 `gemm_nt` N 48: 7.97 µs a call, two a layer (0.765 ms a step).
  - `exl3i_dense` N 128 K4: 14.90 µs, one a layer (0.715 ms a step).
  - The fusion saves ~0.05 ms a step, not w142's ~0.45 ms estimate: one K4 trellis N 128 product costs nearly what two
    bf16 N 48 GEMVs did.
  - e2e decode does not improve (+0.08 ms at ctx 0, +0.2 ms at ctx 8k, both within ~0.2 ms of run-to-run spread).
- **Follow-up (kernel, not done):** exl3i_dense at N 128 is latency-bound (327 KB in 15 µs). A small-N decode path, e.g.
  split-K over 5120, would turn the fusion into the ~0.4 ms a step w142 estimated.

## Files

`box2/scores/*.json` (radiance KLD reports), `box2/scores/cmp147.{out,json}`, `box2/perf/` (bench, e2e jsonl, perfsum,
decode op tables), `box2/run147.log`, `box2/logs/build147.log` (box2's host tests), `ds8/` (census, mixfiles,
mix logs, donor log and recipe, abcheck, ds8's build log), `parts-s171d.json`. The rows files stay on box2:
`/data/q27/w147/scores/`.

On the hosts:
- box2 `/data/q27/w147/mix/2.5-T4.rad` and ds8 `/data/q27/w147/mix/2.5-T4.rad`: the packed 2.5 incumbent with T4 a / b.
- ds8 `/data/q27/w147/donor-ab-K4.rad` (6.4 GB, the donor), `absrc-K4/` (16 MB source), `radmix147`.
- box2 `/data/q27/w147/home`: the serve home.
- The B containers and box2's golden copy are deleted.
