# #179 whole-step decode / prefill pass (w152, 2026-11-13 / 14) — results

Box2 card 1 (R9700, 210 W cap), rad10-perf core (ROCm 10.1) unless a name says 714 (the frozen 7.14 install). Scripts:
`v3/bench/step179/` (README there). Fork branch radiance-hero `qc-exl3-v3-step` (7359986, b73675c, 1f4ecbc, 027fd91);
write-up `plugin/qwen35_exl3/PERF.md` on that branch.

Homes (row names): `base` = qc-exl3-v3 179220f; `step` / `step2` = the fusions (step2 = 7359986's content); `mbase` =
qc-exl3-v3-mad e054e71 (mad, interface 4); `mstep` = its merge with the fusions (b02aa9a, now tag archive/w152-step-b02aa9a);
`mstep3` = b73675c; `mres*` = 1f4ecbc's tree (mres3 = final); `mfix` = qc-exl3-v3 9f0a059 + kctr-key.patch; `mord` = the
same with exl3i's rad_kernel_concurrent 0 (every launch ordered: the race-free reference); `fstep` / `fres` / `fres2` =
b73675c / 1f4ecbc (fres: before the batched last-arriver write) + kctr-key.patch (`fbuilds.txt` in the scripts dir has the
stash commits). `dg0` / `dgskip` = the timing diagnostic (diag/exl3_diag.h: every non-GEMV op of the plugin skipped, outputs
garbage). Packs: mul1i8 `v3-k2.5-r16k2.5-a8`, `v3-k4.125-r16k4-a8` (w138); mad `q27v3-635bac02-k2.5-mad-a8` (w151's level,
w150's pack), `q27v3-144dd6e1-k4.125-mad-a8` (w150); IQR-V4 iq2r drafter off.

| file | what |
|---|---|
| `stamp-*.txt` | RADIANCE_AQL_STAMP per-kernel tables (stsum.py): µs a step / launches, dispatches, idle; d0 ctx 0, d8 ctx 8k |
| `smallops-k2.5-d0.txt` | GEMV / head / a-b / attention / small-op split of the stamped steps |
| `abba-k2.5-mul1i8-stamps.txt`, `ab-*.clock.json` | ABBA base / step2 / step2 / base (1200 tokens) with cardclock's SMU samples and the engine's step counters; the mbase / mstep ABBA ran on the racy mad decode (garbage tokens) |
| `gemvpos-k2.5-base-step2.txt` | every GEMV launch matched base vs step2, by predecessor kernel: +1.5–2.3 % alike (the power cap) |
| `e2e.txt`, `e2/` | w138's bench.py rounds (e2sum.py); `*-mbase-r4` = the racy mad decode (garbage), not used |
| `prefill-profile-*.txt` | --profile-ops of an 8k prefill, base vs step (per op, µs) |
| `kld-rows.txt`, `kld-means.txt` | whole-model KLD rows compared between homes (rowcmp.py): all byte-identical |
| `gdec.txt` | greedy decode with top-20 logprobs (gdec.py): bit-exact comparisons; the mad builds without the patch finish 0 / 8 requests |
| `device-tests.txt` | exl3i device suites (ctall.sh) on 10.1 and 7.14 for mres2 (test_aq's 18 xb cases failed on unwritten plane padding in the test: fixed, mres3 test_aq PASS on both) |
| `queue-log.txt` | the card-1 queue's log (UTC) |
